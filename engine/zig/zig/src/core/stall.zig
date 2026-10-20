//! A round's clock on its own thread: a report when one runs past `limit_s`, and exit 70 past `exit_s`
//! (the vendor Python family's `stall.Watch`). The engine thread publishes a round's start and phase; the watch
//! thread reads them without a lock, so a wedged round holding a host mutex cannot stall the observer.
const std = @import("std");

pub const exit_code: u8 = 70;

/// The engine thread's round sections, in order.
pub const Phase = enum(u8) { idle, admit, plan, pieces, finals, decode, deliver, after };

pub const Settings = struct {
    rank: u32 = 0,
    limit_s: f64 = 300, // TF_DSV41_STALL_S: first report; 0 = no report
    exit_s: f64 = 0, // TF_DSV41_STALL_EXIT_S: exit 70; 0 = never
    poll_ns: u64 = std.time.ns_per_s,

    pub const off: Settings = .{ .limit_s = 0, .exit_s = 0 };

    pub fn on(s: Settings) bool {
        return s.limit_s > 0 or s.exit_s > 0;
    }
};

/// Where a report goes: one write a line, no allocation. The default writes fd 2; `ctx` is the watch it belongs to.
pub const Sink = struct {
    ctx: ?*anyopaque = null,
    line: *const fn (?*anyopaque, []const u8) void = writeStderr,
};

pub const Watch = struct {
    io: std.Io,
    settings: Settings,
    sink: Sink = .{},
    exit_fn: *const fn (u8) void = exitNow, // fate.zig's seam: std.c._exit(70)
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false), // closer writes
    exited: std.atomic.Value(bool) = .init(false), // watch thread writes
    reports: std.atomic.Value(u64) = .init(0), // watch thread writes
    armed: std.atomic.Value(bool) = .init(false), // engine thread writes
    round_no: std.atomic.Value(u64) = .init(0),
    t_round: std.atomic.Value(u64) = .init(0), // monotonic ns, at begin
    current: std.atomic.Value(u8) = .init(@backingInt(Phase.idle)),
    t_phase: std.atomic.Value(u64) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    waiting: std.atomic.Value(u32) = .init(0),
    pieces: std.atomic.Value(u32) = .init(0),
    finals: std.atomic.Value(u32) = .init(0),
    seen_round: u64 = std.math.maxInt(u64), // watch thread only
    last_report: u64 = 0, // watch thread only

    pub fn init(io: std.Io, settings: Settings) Watch {
        return .{ .io = io, .settings = settings };
    }

    pub fn on(w: *const Watch) bool {
        return w.settings.on();
    }

    /// Spawns the watch; a no-op when both knobs are off or it is already running.
    pub fn start(w: *Watch) !void {
        if (!w.on() or w.thread != null) return;
        if (w.sink.ctx == null and w.sink.line == writeStderr) w.sink.ctx = w;
        w.thread = try std.Thread.spawn(.{}, loop, .{w});
    }

    /// Stops and joins the watch; idempotent.
    pub fn close(w: *Watch) void {
        w.stop.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }

    /// A round starts: publish its number, both timestamps and the phase, then arm (release).
    pub fn begin(w: *Watch, round_no: u64, p: Phase) void {
        const now = nowNs(w.io);
        w.round_no.store(round_no, .release);
        w.t_round.store(now, .release);
        w.t_phase.store(now, .release);
        w.current.store(@backingInt(p), .release);
        w.armed.store(true, .release);
    }

    pub fn phase(w: *Watch, p: Phase) void {
        w.t_phase.store(nowNs(w.io), .release);
        w.current.store(@backingInt(p), .release);
    }

    pub fn end(w: *Watch) void {
        w.current.store(@backingInt(Phase.idle), .release);
        w.armed.store(false, .release);
    }

    /// The queue depths a report shows; diagnostics, not the arm.
    pub fn counts(w: *Watch, running: u32, waiting: u32) void {
        w.running.store(running, .release);
        w.waiting.store(waiting, .release);
    }

    /// What the planner filled this round; diagnostics, not the arm.
    pub fn plan(w: *Watch, pieces: u32, finals: u32) void {
        w.pieces.store(pieces, .release);
        w.finals.store(finals, .release);
    }

    /// One poll: reports on Python's cadence and exits past `exit_s`; true once it exited.
    pub fn tick(w: *Watch, now_ns: ?u64) bool {
        if (!w.on()) return false;
        const now = now_ns orelse nowNs(w.io);
        // acquire first: an armed round it sees has its number, start and phase published
        if (!w.armed.load(.acquire)) return false;
        const rnd = w.round_no.load(.acquire);
        if (rnd != w.seen_round) {
            w.seen_round = rnd;
            w.last_report = 0;
        }
        const t_round = w.t_round.load(.acquire);
        const late = now -| t_round;
        var reported = false;
        if (w.settings.limit_s > 0) {
            const limit = secondsToNs(w.settings.limit_s);
            const due = if (w.last_report == 0) late >= limit else now -| w.last_report >= limit *| 4;
            if (due) {
                w.report(now, rnd, t_round);
                w.last_report = now;
                reported = true;
            }
        }
        if (w.settings.exit_s <= 0) return false;
        if (late < secondsToNs(w.settings.exit_s)) return false;
        // the round may have ended (or another begun) during the report
        if (!w.armed.load(.acquire) or w.round_no.load(.acquire) != rnd) return false;
        if (!reported) {
            w.report(now, rnd, t_round);
            w.last_report = now;
            if (!w.armed.load(.acquire) or w.round_no.load(.acquire) != rnd) return false;
        }
        w.exitLine(now, rnd, t_round);
        w.exited.store(true, .release);
        w.exit_fn(exit_code);
        return true;
    }

    fn loop(w: *Watch) void {
        while (!w.stop.load(.acquire)) {
            if (w.tick(null)) return;
            std.Io.sleep(w.io, .fromNanoseconds(w.settings.poll_ns), .awake) catch {};
        }
    }

    fn report(w: *Watch, now: u64, rnd: u64, t_round: u64) void {
        const p: Phase = @fromBackingInt(@intCast(w.current.load(.acquire)));
        const t_phase = w.t_phase.load(.acquire);
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "[dsv41 stall] r{d}: round {d} has run {d:.1} s (phase '{s}' for {d:.1} s; TF_DSV41_STALL_S={d})\n  host: {d} running, {d} waiting; plan {d} pieces, {d} finals\n", .{
            w.settings.rank,
            rnd,
            nsToSeconds(now -| t_round),
            @tagName(p),
            nsToSeconds(now -| t_phase),
            w.settings.limit_s,
            w.running.load(.acquire),
            w.waiting.load(.acquire),
            w.pieces.load(.acquire),
            w.finals.load(.acquire),
        }) catch return;
        w.sink.line(w.sink.ctx, line);
        _ = w.reports.fetchAdd(1, .release);
    }

    fn exitLine(w: *Watch, now: u64, rnd: u64, t_round: u64) void {
        var buf: [256]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "[dsv41 stall] r{d}: round {d} ran {d:.0} s >= TF_DSV41_STALL_EXIT_S={d}: exiting ({d}) so the watchdog restarts both ranks\n", .{
            w.settings.rank,
            rnd,
            nsToSeconds(now -| t_round),
            w.settings.exit_s,
            exit_code,
        }) catch return;
        w.sink.line(w.sink.ctx, line);
    }
};

pub fn exitNow(code: u8) void {
    std.c._exit(code);
}

fn writeStderr(ctx: ?*anyopaque, bytes: []const u8) void {
    const w: *const Watch = @ptrCast(@alignCast(ctx.?));
    std.Io.File.stderr().writeStreamingAll(w.io, bytes) catch {};
}

fn nowNs(io: std.Io) u64 {
    return @intCast(@max(0, std.Io.Clock.awake.now(io).toNanoseconds()));
}

fn nsToSeconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
}

fn secondsToNs(s: f64) u64 {
    const ns = s * std.time.ns_per_s;
    if (!(ns > 0)) return 0;
    if (ns >= @as(f64, @floatFromInt(std.math.maxInt(u64)))) return std.math.maxInt(u64);
    return @intFromFloat(ns);
}

const Cap = struct {
    lines: u32 = 0,
    text: [1024]u8 = undefined,
    len: usize = 0,

    fn line(ctx: ?*anyopaque, bytes: []const u8) void {
        const c: *@This() = @ptrCast(@alignCast(ctx.?));
        c.lines += 1;
        const n = @min(bytes.len, c.text.len);
        @memcpy(c.text[0..n], bytes[0..n]);
        c.len = n;
    }
};

const Exit = struct {
    var code: u8 = 0;

    fn record(c: u8) void {
        code = c;
    }
};

test "stall: a report at the limit, then every four limits; an idle or ended round never reports" {
    const s: u64 = std.time.ns_per_s;
    var cap: Cap = .{};
    var w = Watch.init(std.testing.io, .{ .limit_s = 300, .exit_s = 0 });
    w.sink = .{ .ctx = &cap, .line = Cap.line };
    // an idle watch is never armed
    try std.testing.expect(!w.tick(10_000 * s));
    try std.testing.expectEqual(@as(u32, 0), cap.lines);
    w.begin(1, .admit);
    const t1 = w.t_round.load(.acquire);
    try std.testing.expect(!w.tick(t1 + 299 * s));
    try std.testing.expectEqual(@as(u32, 0), cap.lines);
    try std.testing.expect(!w.tick(t1 + 300 * s));
    try std.testing.expectEqual(@as(u32, 1), cap.lines);
    try std.testing.expect(std.mem.indexOf(u8, cap.text[0..cap.len], "round 1 has run 300.0 s (phase 'admit' for 300.0 s; TF_DSV41_STALL_S=300)") != null);
    try std.testing.expect(std.mem.indexOf(u8, cap.text[0..cap.len], "  host: 0 running, 0 waiting; plan 0 pieces, 0 finals") != null);
    // the next report is four limits after the first, not at 600 s
    try std.testing.expect(!w.tick(t1 + 1499 * s));
    try std.testing.expectEqual(@as(u32, 1), cap.lines);
    try std.testing.expect(!w.tick(t1 + 1500 * s));
    try std.testing.expectEqual(@as(u32, 2), cap.lines);
    // a round that ended is not armed, and a fresh round's age starts at its own begin
    w.end();
    try std.testing.expect(!w.tick(t1 + 2000 * s));
    w.begin(2, .decode);
    const t2 = w.t_round.load(.acquire);
    try std.testing.expect(!w.tick(t2 + 299 * s));
    try std.testing.expect(!w.tick(t2 + 300 * s));
    try std.testing.expect(std.mem.indexOf(u8, cap.text[0..cap.len], "round 2 has run 300.0 s (phase 'decode'") != null);
}

test "stall: exit 70 past exit_s, with one report on that tick when the cadence did not report" {
    const s: u64 = std.time.ns_per_s;
    Exit.code = 0;
    var cap: Cap = .{};
    // limit 0: one report immediately before the exit
    var w = Watch.init(std.testing.io, .{ .limit_s = 0, .exit_s = 10 });
    w.sink = .{ .ctx = &cap, .line = Cap.line };
    w.exit_fn = Exit.record;
    w.begin(7, .decode);
    w.counts(4, 2);
    w.plan(6, 2);
    const t7 = w.t_round.load(.acquire);
    try std.testing.expect(!w.tick(t7 + 9 * s));
    try std.testing.expectEqual(@as(u32, 0), cap.lines);
    try std.testing.expect(w.tick(t7 + 10 * s));
    try std.testing.expectEqual(@as(u8, 70), Exit.code);
    try std.testing.expectEqual(@as(u32, 2), cap.lines);
    try std.testing.expect(std.mem.indexOf(u8, cap.text[0..cap.len], "exiting (70) so the watchdog restarts both ranks") != null);
    try std.testing.expectEqual(@as(u64, 1), w.reports.load(.acquire));

    // exit_s under limit_s: the cadence is not due, so the exit tick forces its own report
    Exit.code = 0;
    var early: Cap = .{};
    var x = Watch.init(std.testing.io, .{ .limit_s = 300, .exit_s = 100 });
    x.sink = .{ .ctx = &early, .line = Cap.line };
    x.exit_fn = Exit.record;
    x.begin(8, .pieces);
    const t8 = x.t_round.load(.acquire);
    try std.testing.expect(x.tick(t8 + 100 * s));
    try std.testing.expectEqual(@as(u8, 70), Exit.code);
    try std.testing.expectEqual(@as(u32, 2), early.lines);
}

test "stall: a round that ends or changes during the report is not exited" {
    const s: u64 = std.time.ns_per_s;
    Exit.code = 0;
    var ended = Watch.init(std.testing.io, .{ .limit_s = 0, .exit_s = 10 });
    const Ender = struct {
        w: *Watch,
        fn line(ctx: ?*anyopaque, _: []const u8) void {
            const e: *@This() = @ptrCast(@alignCast(ctx.?));
            e.w.end();
        }
    };
    var ender = Ender{ .w = &ended };
    ended.sink = .{ .ctx = &ender, .line = Ender.line };
    ended.exit_fn = Exit.record;
    ended.begin(1, .decode);
    const te = ended.t_round.load(.acquire);
    try std.testing.expect(!ended.tick(te + 10 * s));
    try std.testing.expectEqual(@as(u8, 0), Exit.code);

    var changed = Watch.init(std.testing.io, .{ .limit_s = 0, .exit_s = 10 });
    const Changer = struct {
        w: *Watch,
        fn line(ctx: ?*anyopaque, _: []const u8) void {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            c.w.begin(2, .plan);
        }
    };
    var changer = Changer{ .w = &changed };
    changed.sink = .{ .ctx = &changer, .line = Changer.line };
    changed.exit_fn = Exit.record;
    changed.begin(1, .decode);
    const tc = changed.t_round.load(.acquire);
    try std.testing.expect(!changed.tick(tc + 10 * s));
    try std.testing.expectEqual(@as(u8, 0), Exit.code);
}
