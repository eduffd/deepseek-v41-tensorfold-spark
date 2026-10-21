//! The plan link's two failure shapes kept apart: a peer that died stays `PeerClosed`, a peer that stopped moving is
//! `PeerStalled` named by the wait it expired on, and a link with no deadline source keeps blocking forever (gateB).
const std = @import("std");
const posix = std.posix;
const sock = @import("sock.zig");
const bootstrap = @import("bootstrap.zig");
const planlink = @import("planlink.zig");
const peerwait = @import("peerwait.zig");

const PlanLink = planlink.PlanLink;
const Cause = peerwait.Cause;
const ms = std.time.ns_per_ms;

/// Two connected ranks' links: rank 0's at index 1, rank 1's at index 0.
fn linked() !struct { l: sock.Fd, lead: PlanLink, follow: PlanLink } {
    const ls = try bootstrap.testListener();
    errdefer sock.close(ls.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", ls.port, deadline);
    const b = try sock.accept(ls.fd, deadline);
    var f0: [planlink.max_ranks]sock.Fd = @splat(-1);
    var f1: [planlink.max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    return .{ .l = ls.fd, .lead = PlanLink.init(0, 2, f0), .follow = PlanLink.init(1, 2, f1) };
}

/// The armed clock the link asks per wait: every read or send gets `now + after_ns`.
const Clocked = struct {
    after_ns: u64,
    fn deadline(ctx: *anyopaque) ?u64 {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return sock.nowNs() + s.after_ns;
    }
};

/// What the link reported expired. One writer (the blocked thread), read after it returns.
const CauseCap = struct {
    n: std.atomic.Value(u32) = .init(0),
    last: Cause = undefined,

    fn call(ctx: *anyopaque, c: Cause) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.last = c;
        _ = s.n.fetchAdd(1, .release);
    }

    fn wantName(c: *@This(), want: []const u8) !void {
        return std.testing.expectEqualStrings(want, @tagName(c.last));
    }
};

test "a plan read past its deadline is PeerStalled and names the wait; a hang-up past it is still PeerClosed" {
    var p = try linked();
    defer {
        p.follow.close();
        p.lead.close();
        sock.close(p.l);
    }
    var cl = Clocked{ .after_ns = 20 * ms };
    var cap: CauseCap = .{};
    p.follow.deadline_src = .{ .ctx = &cl, .call = Clocked.deadline };
    p.follow.on_deadline = .{ .ctx = &cap, .call = CauseCap.call };
    var buf: [8]i64 = undefined;
    try std.testing.expectError(error.PeerStalled, p.follow.recv(&buf));
    try std.testing.expectEqual(@as(u32, 1), cap.n.load(.acquire));
    try cap.wantName("plan_recv");
    switch (cap.last) {
        .plan_recv => |c| {
            try std.testing.expectEqual(@as(u32, 1), c.rank);
            try std.testing.expectEqual(@as(u64, 0), c.received);
        },
        else => {},
    }
    // the shape a departure keeps: a closed leader past the same armed deadline is never a stall
    p.lead.close();
    try std.testing.expectError(error.PeerClosed, p.follow.recv(&buf));
    try std.testing.expectEqual(@as(u32, 1), cap.n.load(.acquire));
}

test "a leader's plan a peer stopped reading is PeerStalled before the count moves" {
    var p = try linked();
    defer {
        p.follow.close();
        p.lead.close();
        sock.close(p.l);
    }
    // follow never reads: the window fills and the frame parks mid-send
    const small: c_int = 8192;
    _ = std.c.setsockopt(p.lead.fds[1], posix.SOL.SOCKET, posix.SO.SNDBUF, &small, @sizeOf(c_int));
    try p.lead.send(&.{ 1, 2, 3 }); // accepted into the buffer: this frame counted
    var cl = Clocked{ .after_ns = 60 * ms };
    var cap: CauseCap = .{};
    p.lead.deadline_src = .{ .ctx = &cl, .call = Clocked.deadline };
    p.lead.on_deadline = .{ .ctx = &cap, .call = CauseCap.call };
    const big = try std.testing.alloc(i64, 1 << 20); // 8 MiB, far past what the pair's buffers hold
    defer std.testing.allocator.free(big);
    @memset(big, 7);
    try std.testing.expectError(error.PeerStalled, p.lead.send(big));
    try std.testing.expectEqual(@as(u64, 1), p.lead.sent.load(.acquire)); // mid-frame: the frame never landed
    try std.testing.expectEqual(@as(u32, 1), cap.n.load(.acquire));
    try cap.wantName("plan_send");
    switch (cap.last) {
        .plan_send => |c| {
            try std.testing.expectEqual(@as(u32, 0), c.rank);
            try std.testing.expectEqual(@as(u64, 1), c.sent); // the cause reports the accepted count, not this frame
        },
        else => {},
    }
}

test "a follower with no deadline source waits past any bound and wakes on the plan" {
    var p = try linked();
    defer {
        p.follow.close();
        p.lead.close();
        sock.close(p.l);
    }
    const Wait = struct {
        link: *PlanLink,
        buf: std.ArrayList(i64) = .empty,
        first: std.atomic.Value(i64) = .init(0),
        done: std.atomic.Value(bool) = .init(false),

        fn run(w: *@This()) void {
            defer w.buf.deinit(std.testing.allocator);
            const ints = w.link.recvGrow(std.testing.allocator, &w.buf) catch return;
            if (ints.len > 0) w.first.store(ints[0], .release);
            w.done.store(true, .release);
        }
    };
    var w = Wait{ .link = &p.follow };
    const t = try std.Thread.spawn(.{}, Wait.run, .{&w});
    sock.sleepNs(300 * ms); // the bound this test would use is shorter than the wait: an unarmed read ignores it
    try std.testing.expect(!w.done.load(.acquire));
    try p.lead.send(&.{7});
    t.join();
    try std.testing.expect(w.done.load(.acquire));
    try std.testing.expectEqual(@as(i64, 7), w.first.load(.acquire));
}

test "round markers are plan frames that count, on op words 66 and 67" {
    var p = try linked();
    defer {
        p.follow.close();
        p.lead.close();
        sock.close(p.l);
    }
    try std.testing.expectEqual(@as(i64, 66), planlink.op_round_open);
    try std.testing.expectEqual(@as(i64, 67), planlink.op_round_close);
    try p.lead.sendRoundOpen(9);
    try p.lead.sendRoundClose();
    var buf: [8]i64 = undefined;
    try std.testing.expectEqualSlices(i64, &.{ 66, 9 }, try p.follow.recv(&buf));
    try std.testing.expectEqualSlices(i64, &.{67}, try p.follow.recv(&buf));
    try std.testing.expectEqual(@as(u64, 2), p.lead.sent.load(.acquire));
    try std.testing.expectEqual(@as(u64, 2), p.follow.received.load(.acquire));
}
