//! The peer-wait clock over injected progress: silence reports at a quarter of the bound and expires at it, counters
//! that move never reach it, a closed round has no deadline at all, and one expiry prints one named line whose reason
//! rides the FAIL frame to the peer.
const std = @import("std");
const sock = @import("sock.zig");
const fate = @import("fate.zig");
const bootstrap = @import("bootstrap.zig");
const peerwait = @import("peerwait.zig");

const Clock = peerwait.Clock;

var exits: std.atomic.Value(u32) = .init(0);

fn recordExit(code: u8) void {
    std.debug.assert(code == fate.exit_code);
    _ = exits.fetchAdd(1, .acq_rel);
}

const base: u64 = 1_000_000;

/// The clock the injected `now` reads: a test steps it, so a bound is measured, not raced.
var now_t: std.atomic.Value(u64) = .init(base);

fn injectedNow() u64 {
    return now_t.load(.monotonic);
}

fn advance(ns: u64) void {
    now_t.store(base + ns, .release);
}

/// What this rank's progress counters say, as the session's sampler would read them.
const Src = struct {
    p: fate.Progress = .{ .transport = .host },

    fn read(ctx: *anyopaque, out: *fate.Progress) void {
        const s: *Src = @ptrCast(@alignCast(ctx));
        out.* = s.p;
    }
};

/// The lines the clock reported, verbatim.
const Cap = struct {
    lines: [16][512]u8 = undefined,
    lens: [16]usize = undefined,
    n: usize = 0,
    close_clock: ?*Clock = null, // set: the round ends inside its own report line

    fn line(ctx: *anyopaque, text: []const u8) void {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        if (c.n < c.lines.len) {
            const m = @min(text.len, c.lines[c.n].len);
            @memcpy(c.lines[c.n][0..m], text[0..m]);
            c.lens[c.n] = m;
            c.n += 1;
        }
        if (c.close_clock) |clk| clk.endRound();
    }

    fn at(c: *@This(), i: usize) []const u8 {
        return c.lines[i][0..c.lens[i]];
    }
};

/// Each clock starts its injected `now` at `base`: `advance` steps this test's own clock, not the one the test
/// before it left running, so a bound is measured from a known instant. The tests speak nanoseconds (a bound is pinned
/// to the nanosecond), while `Settings` carries the knob's units, so the allowance is built here: `nanos` rounds to the
/// nearest ns, which is what makes these bounds come back exactly the number the test wrote.
fn clock(rank: u32, silence_ns: u64, per_row_ns: u64, cap: *Cap, src: ?*Src) Clock {
    now_t.store(base, .monotonic);
    return .{
        .settings = .{
            .rank = rank,
            .allowance = .{
                .base_s = @as(f64, @floatFromInt(silence_ns)) / 1e9,
                .prefill_tps = if (per_row_ns > 0) 1e9 / @as(f64, @floatFromInt(per_row_ns)) else 0,
            },
        },
        .now = injectedNow,
        .sink = .{ .ctx = cap, .call = Cap.line },
        .progress = if (src) |s| fate.ProgressSource{ .ctx = s, .read = Src.read } else null,
        .exit_fn = recordExit,
    };
}

const grace = 50 * std.time.ns_per_ms;

/// Two connected ranks' fate channels, started, with recorded exits (fate_test's harness, kept local).
fn fatePair(fates: *[2]fate.Fate) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var fds0: [fate.max_ranks]sock.Fd = @splat(-1);
    var fds1: [fate.max_ranks]sock.Fd = @splat(-1);
    fds0[1] = b;
    fds1[0] = a;
    for (fates, [2][fate.max_ranks]sock.Fd{ fds0, fds1 }, 0..) |*f, fds, r| {
        f.* = fate.Fate.init(.{ .world = 2, .rank = @intCast(r), .grace_ns = grace, .report_ns = 20 * std.time.ns_per_ms }, fds);
        f.exit_fn = recordExit;
    }
    for (fates) |*f| try f.start();
}

fn waitExits(want: u32) !void {
    const t0 = sock.nowNs();
    while (exits.load(.acquire) < want) {
        if (sock.nowNs() - t0 > 5 * std.time.ns_per_s) return error.Timeout;
        sock.sleepNs(std.time.ns_per_ms);
    }
}

test "silence reports at a quarter of the bound and expires at the bound" {
    exits.store(0, .release);
    var src: Src = .{};
    var cap: Cap = .{};
    var c = clock(1, 400, 0, &cap, &src);
    c.beginRound(7);

    advance(99);
    try std.testing.expect(!c.tick());
    try std.testing.expectEqual(@as(usize, 0), cap.n);

    advance(100);
    try std.testing.expect(!c.tick());
    try std.testing.expectEqual(@as(usize, 1), cap.n);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), "[dsv41 peer-wait]") != null);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), "round 7") != null);
    try std.testing.expect(!c.expired());

    advance(300); // mid-way between the first report and the bound
    try std.testing.expect(!c.tick());
    try std.testing.expectEqual(@as(usize, 1), cap.n);

    advance(400);
    try std.testing.expect(c.tick());
    try std.testing.expectEqual(@as(usize, 2), cap.n);
    try std.testing.expectEqual(@as(u32, 1), exits.load(.acquire));
    try std.testing.expect(c.expired());

    advance(5_000_000);
    try std.testing.expect(!c.tick()); // expired once: nothing else is printed, no second exit
    try std.testing.expectEqual(@as(usize, 2), cap.n);
    try std.testing.expectEqual(@as(u32, 1), exits.load(.acquire));
}

test "counters that keep moving never reach the bound" {
    exits.store(0, .release);
    var src: Src = .{};
    var cap: Cap = .{};
    var c = clock(1, 400, 0, &cap, &src);
    c.beginRound(7);
    for (0..40) |i| {
        advance(i * 300); // past the bound every round, but the counters moved at this very tick
        src.p.plans += 1;
        src.p.completed += 1;
        try std.testing.expect(!c.tick());
    }
    try std.testing.expectEqual(@as(usize, 0), cap.n);
    try std.testing.expect(!c.expired());
    try std.testing.expectEqual(@as(u32, 0), exits.load(.acquire));
}

test "a closed round has no deadline, and neither has the knob at zero" {
    exits.store(0, .release);
    var src: Src = .{};
    var cap: Cap = .{};
    var c = clock(1, 400, 0, &cap, &src);
    c.beginRound(7);
    const armed = c.deadline();
    try std.testing.expect(armed != null);
    try std.testing.expect(armed.? >= injectedNow() + 400);
    c.endRound();
    try std.testing.expect(c.deadline() == null);
    advance(10_000_000);
    try std.testing.expect(!c.tick());
    try std.testing.expectEqual(@as(usize, 0), cap.n);
    try std.testing.expect(!c.expired());

    var off = clock(1, 0, 0, &cap, &src); // TF_DSV41_PEER_WAIT_S=0: the deadline is null forever, HEAD's path
    off.beginRound(7);
    try std.testing.expect(off.deadline() == null);
    advance(20_000_000);
    try std.testing.expect(!off.tick());
    try std.testing.expect(!off.expired());
    try std.testing.expectEqual(@as(u32, 0), exits.load(.acquire));
}

test "admitted rows widen the bound by rows / prefill-tps" {
    exits.store(0, .release);
    var src: Src = .{};
    var cap: Cap = .{};
    var c = clock(1, 400, 10, &cap, &src); // silence 400 ns, 10 ns a row
    c.beginRound(7);
    c.setRows(30); // +300 ns: the bound moves to 700, the report cadence with it
    try std.testing.expectEqual(@as(u64, base + 700), c.deadline().?);
    advance(175);
    try std.testing.expect(!c.tick());
    advance(600); // inside the widened bound: no exit, and only the first report so far
    try std.testing.expect(!c.tick());
    try std.testing.expectEqual(@as(usize, 1), cap.n);
    try std.testing.expect(!c.expired());
    advance(700);
    try std.testing.expect(c.tick());
    try std.testing.expectEqual(@as(u32, 1), exits.load(.acquire));
}

test "a round that ends inside its own report line is not expired" {
    exits.store(0, .release);
    var src: Src = .{};
    var cap: Cap = .{};
    var c = clock(1, 400, 0, &cap, &src);
    cap.close_clock = &c;
    c.beginRound(7);
    advance(400); // the report fires, the sink ends the round, and the clock re-checks before deciding
    try std.testing.expect(!c.tick());
    try std.testing.expectEqual(@as(usize, 1), cap.n);
    try std.testing.expect(!c.expired());
    try std.testing.expectEqual(@as(u32, 0), exits.load(.acquire));
}

test "a fault a witness saw outside the silence stands even with the round closed" {
    exits.store(0, .release);
    var cap: Cap = .{};
    var c = clock(1, 400, 0, &cap, null);
    c.beginRound(5);
    c.endRound(); // a silence expiry would stand down here; a witness's does not
    try std.testing.expect(c.fault(.{ .round_watch = .{ .rank = 1, .round = 5 } }));
    try std.testing.expectEqual(@as(u32, 1), exits.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), cap.n);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), "round watch") != null);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), "round 5") != null);
    try std.testing.expect(!c.fault(.{ .round_watch = .{ .rank = 1, .round = 5 } })); // one latch, as any other route
    try std.testing.expectEqual(@as(u32, 1), exits.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), cap.n);
}

test "one expiry prints one named line, the FAIL frame carries its reason, and a second expiry changes nothing" {
    exits.store(0, .release);
    var fates: [2]fate.Fate = undefined;
    try fatePair(&fates);
    defer for (&fates) |*f| f.close();
    var cap: Cap = .{};
    var c = clock(1, 400, 0, &cap, null);
    c.fate = &fates[1];
    c.beginRound(12);
    advance(10_000_000);
    try std.testing.expect(c.tick()); // the silence expired: fatal through rank 1's fate
    _ = c.expire(.{ .plan_recv = .{ .rank = 1, .round = 12, .allow_s = 0, .received = 0 } }); // idempotent
    try waitExits(2);
    try std.testing.expectEqual(@as(usize, 1), cap.n);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), "plan read") != null);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), peerwait.knob) != null);
    const r0 = fates[0].reason(); // rank 0's own reason: rank 1's FAIL frame, verbatim
    try std.testing.expect(std.mem.indexOf(u8, r0, "plan read") != null);
    try std.testing.expect(std.mem.indexOf(u8, r0, "round 12") != null);
    try std.testing.expect(std.mem.indexOf(u8, r0, "PeerStalled") != null);
}

test "each wait site's sentence names the site, the rank that waits, the knob its bound runs on, and the counters it watched" {
    const Site = struct { rank: u32, c: peerwait.Cause, want: []const []const u8 };
    // every variant the layer can expire on, as its wait site builds it: the comptime check in peerwait.zig only
    // proves the fields exist, so the sentences are pinned here, one row per site. A row states what its site states
    // (its counters, and its own bound where the bound is not the clock's) and leaves the rest to `fill`, exactly as
    // the clock's expiry path does. `rank` is the clock's rank, which is the site's rank: rank 0 is a rank and not
    // "unset", and it is rank 0 that sends the plan, so that row proves the name survives the fill.
    const sites = [_]Site{
        .{ .rank = 1, .c = .{ .plan_recv = .{ .received = 4 } }, .want = &.{ "plan read", "rank 1", "round 7", "for 300000 ms past", peerwait.knob, "frames read 4" } },
        .{ .rank = 0, .c = .{ .plan_send = .{ .sent = 9 } }, .want = &.{ "plan send", "rank 0", "round 7", "for 300000 ms past", peerwait.knob, "plans sent 9" } },
        .{ .rank = 1, .c = .{ .device_wait = .{ .what = "all-gather" } }, .want = &.{ "device wait", "rank 1", "round 7", "all-gather", "for 300000 ms past", peerwait.knob } },
        // the proxy's bound is the mailbox timeout: its site names that length itself, so the clock's allowance must
        // not overwrite it, and the line quotes that knob and not the peer wait
        .{ .rank = 0, .c = .{ .proxy_inflight = .{ .allow_s = 120, .why = "posted write" } }, .want = &.{ "proxy in flight", "rank 0", "round 7", "posted write", "for 120000 ms past", peerwait.mailbox_knob } },
        // the round boundary's poll: the error the transport recorded, by name
        .{ .rank = 1, .c = .{ .transport_fault = .{ .fault = @errorName(error.PeerFailed) } }, .want = &.{ "transport fault", "rank 1", "round 7", "PeerFailed", peerwait.knob } },
        .{ .rank = 1, .c = .{ .round_watch = .{} }, .want = &.{ "round watch", "rank 1", "round 7", "open 300000 ms past", peerwait.knob } },
    };
    for (sites) |s| {
        var c = s.c;
        c.fill(s.rank, 7, 300); // the clock fills the round and the bound it runs on
        var buf: [448]u8 = undefined;
        const line = c.format(&buf) catch |e| {
            std.debug.print("{s}: {t}\n", .{ @tagName(c), e });
            return error.TestUnexpectedResult;
        };
        for (s.want) |want| {
            if (std.mem.indexOf(u8, line, want) == null) {
                std.debug.print("{s}\n  missing '{s}'\n", .{ line, want });
                return error.TestUnexpectedResult;
            }
        }
    }
}
