//! A round wedged inside the planner: the stall watch reports and exits 70 without the host's mutex.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Request = api.Request;

const Exit = struct {
    var code: std.atomic.Value(u8) = .init(0);

    fn record(c: u8) void {
        code.store(c, .release);
    }
};

const Cap = struct {
    mutex: std.Io.Mutex = .init,
    text: [2048]u8 = undefined,
    len: usize = 0,

    fn line(ctx: ?*anyopaque, bytes: []const u8) void {
        const c: *@This() = @ptrCast(@alignCast(ctx.?));
        c.mutex.lockUncancelable(std.testing.io);
        defer c.mutex.unlock(std.testing.io);
        const n = @min(bytes.len, c.text.len);
        @memcpy(c.text[0..n], bytes[0..n]);
        c.len = n;
    }

    fn copy(c: *Cap, out: []u8) []const u8 {
        c.mutex.lockUncancelable(std.testing.io);
        defer c.mutex.unlock(std.testing.io);
        const n = @min(c.len, out.len);
        @memcpy(out[0..n], c.text[0..n]);
        return out[0..n];
    }
};

test "a round wedged in plan exits 70 past the watch's exit knob" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });

    const Gate = struct {
        open: std.atomic.Value(bool) = .init(false),
        entered: std.atomic.Value(bool) = .init(false),
    };
    var gate: Gate = .{};
    const Plan = struct {
        gate: *Gate,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(_: *anyopaque, _: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            const p = self(ctx);
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            p.gate.entered.store(true, .release);
            while (!p.gate.open.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        fn left(_: *anyopaque, _: usize) void {}
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl = Plan{ .gate = &gate };
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };

    Exit.code.store(0, .release);
    var cap: Cap = .{};
    try host.startStall(.{ .limit_s = 0.02, .exit_s = 0.05, .poll_ns = 5 * std.time.ns_per_ms }, Exit.record, .{ .ctx = &cap, .line = Cap.line });
    try host.start();
    defer host.stop();
    // a wedged planner must not hold the stop: the gate opens before the join
    defer gate.open.store(true, .release);

    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 1 << 30 };
    try host.engine().submit(1, &request, .{ .ctx = &cap, .event = Sink.event });

    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (Exit.code.load(.acquire) == 0 and std.Io.Clock.awake.now(std.testing.io).toNanoseconds() < deadline)
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
    try std.testing.expect(gate.entered.load(.acquire));
    try std.testing.expectEqual(@as(u8, 70), Exit.code.load(.acquire));
    var text: [2048]u8 = undefined;
    const line = cap.copy(&text);
    try std.testing.expect(std.mem.indexOf(u8, line, "exiting (70) so the watchdog restarts both ranks") != null);
}

const Sink = struct {
    fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
};
