//! The round bracket at the host seam: every round the engine stamps opens one and closes one, in pairs, and what it
//! hands the hooks is the round number the markers will carry and the rows that round admitted.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Request = api.Request;

const Rec = struct {
    begins: std.atomic.Value(u32) = .init(0),
    ends: std.atomic.Value(u32) = .init(0),
    rows: std.atomic.Value(u64) = .init(0), // summed across the rounds seen: an allowance needs the total
    last_round: std.atomic.Value(u64) = .init(0),

    fn begin(ctx: *anyopaque, round_no: u64, rows: u64) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.last_round.store(round_no, .release);
        _ = r.rows.fetchAdd(rows, .release);
        _ = r.begins.fetchAdd(1, .release);
    }

    fn end(ctx: *anyopaque) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        _ = r.ends.fetchAdd(1, .release);
    }
};

const Sink = struct {
    fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
};

test "every round the engine stamps opens and closes one bracket, carrying round and rows" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var rec: Rec = .{};
    host.on_round = .{ .ctx = &rec, .begin = Rec.begin, .end = Rec.end };

    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 1 << 30 };
    try host.engine().submit(1, &request, .{ .ctx = &rec, .event = Sink.event });
    try host.start();
    // the fake target decodes fast: a few hundred ms is many rounds
    const until = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 400 * std.time.ns_per_ms;
    while (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() < until)
        std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    host.stop();

    const begins = rec.begins.load(.acquire);
    try std.testing.expectEqual(begins, rec.ends.load(.acquire));
    try std.testing.expect(begins >= 2); // an admit round and at least one decode round
    try std.testing.expect(rec.rows.load(.acquire) >= prompt.len); // the admitted prompt counts at least once
    try std.testing.expectEqual(host.round_no, rec.last_round.load(.acquire));
}
