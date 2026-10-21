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
/// before it left running, so a bound is measured from a known instant.
fn clock(rank: u32, silence_ns: u64, per_row_ns: u64, cap: *Cap, src: ?*Src) Clock {
    now_t.store(base, .monotonic);
    return .{
        .settings = .{ .rank = rank, .silence_ns = silence_ns, .prefill_ns_per_row = per_row_ns },
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
    c.expire(.{ .plan_recv = .{ .rank = 1, .round = 12, .allow_s = 0, .received = 0 } }); // idempotent
    try waitExits(2);
    try std.testing.expectEqual(@as(usize, 1), cap.n);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), "plan read") != null);
    try std.testing.expect(std.mem.indexOf(u8, cap.at(0), peerwait.knob) != null);
    const r0 = fates[0].reason(); // rank 0's own reason: rank 1's FAIL frame, verbatim
    try std.testing.expect(std.mem.indexOf(u8, r0, "plan read") != null);
    try std.testing.expect(std.mem.indexOf(u8, r0, "round 12") != null);
    try std.testing.expect(std.mem.indexOf(u8, r0, "PeerStalled") != null);
}
