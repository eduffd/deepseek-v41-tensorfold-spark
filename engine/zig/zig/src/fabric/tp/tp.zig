//! Tensor parallelism across processes (two DGX Sparks, one GPU each): bootstrap, collectives, fail-fast; `Session.open` brings a rank up.
const std = @import("std");
const cuda = @import("cuda");

pub const collective = @import("collective.zig");
pub const reference = @import("reference.zig");
pub const config = @import("config.zig");
pub const sock = @import("sock.zig");
pub const bootstrap = @import("bootstrap.zig");
pub const fate = @import("fate.zig");
pub const peerwait = @import("peerwait.zig");
pub const host = @import("host.zig");
pub const nccl = @import("nccl.zig");
pub const mailbox = @import("mailbox.zig");
pub const hybrid = @import("hybrid.zig");
pub const roce = @import("roce.zig");
pub const roce_verbs = @import("roce_verbs.zig");
pub const planlink = @import("planlink.zig");
pub const Verbs = @import("verbs").Verbs;

pub const Collective = collective.Collective;
pub const DType = collective.DType;
pub const Op = collective.Op;
pub const Config = config.Config;
pub const Bootstrap = bootstrap.Bootstrap;
pub const Fate = fate.Fate;

pub const Session = struct {
    cfg: Config,
    lib: cuda.nccl.Library,
    nccl: nccl.Nccl,
    mail: ?mailbox.Mailbox = null,
    roce: ?*roce.Roce = null,
    route: hybrid.Hybrid = undefined,
    /// The round plan's link (rank 0 sends, the others receive), on its own TCP connection.
    plan: planlink.PlanLink = undefined,
    fate: ?Fate = null,
    /// The peer-wait clock (peerwait.zig): armed inside a round, disarmed at its boundary, off with the knob at 0.
    peer: peerwait.Clock = .{},
    /// The fate thread's frozen watch: the last sample's counters, and when they last moved. ONE writer: the
    /// fate thread calls `progress` once per report period; the frozen bookkeeping mutates state, so no other
    /// thread calls it.
    last_sample: ?fate.Progress = null,
    moved_at: u64 = 0,

    /// Brings this rank up (in place: the session's address must not change afterwards). `kernel_image` is
    /// mailbox.cu's fatbin (the mailbox and RoCE kernels), empty when the build has none: those backends then use
    /// NCCL.
    pub fn open(s: *Session, d: *const cuda.Driver, cfg: Config, kernel_image: []const u8) !void {
        var b = try Bootstrap.open(cfg);
        defer b.close();
        try b.agree(&cfg.digest(), "settings (TF_TP_WORLD, TF_COMM_BACKEND, TF_TP_MAILBOX_MAX_KB, TF_DSV41_FAILFAST, TF_TP_ROCE_WIRE / _FALLBACK)");
        var lib = try cuda.nccl.Library.openPath(cfg.nccl_lib);
        errdefer lib.close();
        const v = try lib.version();
        try b.agree(std.mem.asBytes(&v), "NCCL version");
        var id: cuda.nccl.UniqueId = undefined;
        if (cfg.rank == 0) try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try b.broadcast(&id.internal, &id.internal);
        s.* = .{ .cfg = cfg, .lib = lib, .nccl = undefined };
        s.nccl = try nccl.Nccl.init(&s.lib, d, id, cfg.rank, cfg.world);
        errdefer s.nccl.deinit();
        switch (cfg.backend) {
            .nccl => {},
            .mailbox => if (try mailboxPossible(&b, kernel_image)) {
                s.mail = try mailbox.Mailbox.init(d, &b, kernel_image, cfg.mailbox_max_bytes, cfg.mailbox_timeout_ns);
                s.route = .{ .small = .{ .mailbox = &s.mail.? }, .nccl = &s.nccl, .max_bytes = cfg.mailbox_max_bytes };
            } else if (cfg.rank == 0) {
                std.log.warn("[tensorfold] tp: TF_COMM_BACKEND=mailbox needs two ranks on one host and the kernel; every rank uses NCCL", .{});
            },
            .roce => try s.openRoce(d, &b, kernel_image),
        }
        try s.agreeVar(&b);
        s.plan = planlink.PlanLink.init(cfg.rank, cfg.world, b.releasePlan());
        s.plan.spin_ns = cfg.plan_spin_ns;
        s.plan.pin = cfg.plan_pin;
        s.plan.avoid = cfg.roce_cpu;
        if (cfg.failfast) {
            s.fate = Fate.init(cfg, b.release());
            s.fate.?.onFail(.{ .ctx = s, .call = abortHook });
            // armed before the thread spawns: a field a running thread never races. The session's address is
            // fixed for its whole life (`open` runs in place; `close` joins the fate thread before teardown)
            s.fate.?.setProgress(.{ .ctx = s, .read = progressFrame });
            if (cfg.rank == 0) s.fate.?.setLeaderProgress(.{ .ctx = s, .read = leaderSent });
            try s.fate.?.start();
        }
        // TF_DSV41_PEER_WAIT_S: the clock beside the fate thread; at 0 it spawns no thread and the plan link's
        // deadline source hands out null forever, which is the path this tree had before
        s.peer = peerwait.Clock{
            .settings = .{
                .rank = cfg.rank,
                .silence_ns = cfg.peer_wait_ns,
                .prefill_ns_per_row = if (cfg.prefill_tps > 0) @intFromFloat(@min(1e9 / cfg.prefill_tps, 1e15)) else 0,
            },
            .progress = fate.ProgressSource{ .ctx = s, .read = peerProgress },
        };
        if (s.fate) |*f| s.peer.fate = f;
        s.plan.deadline_src = .{ .ctx = s, .call = peerDeadline };
        s.plan.on_deadline = .{ .ctx = s, .call = peerDeadlineFired };
        try s.peer.start();
        std.log.info("[tensorfold] tp: rank {d} of {d} up ({t}, NCCL {d}, fail-fast {s}, roce fast {s}: {d} poller(s), stagger {d} ns)", .{ cfg.rank, cfg.world, s.comm().kind(), v, if (cfg.failfast) "on" else "off", if (cfg.roce_fast & 1 != 0) "on" else "off", (cfg.roce_fast >> 8) & 0xff, cfg.roce_fast >> 16 });
    }

    /// RoCE for the small exchanges: a collective setup, then a load-time probe against NCCL's bytes; a failure on
    /// any rank puts every rank on NCCL (TF_TP_ROCE_FALLBACK=nccl) or stops them all (=error).
    fn openRoce(s: *Session, d: *const cuda.Driver, b: *Bootstrap, image: []const u8) !void {
        const cfg = s.cfg;
        var specs: [roce_verbs.max_hcas]roce_verbs.HcaSpec = undefined;
        var n_specs: usize = 0;
        if (cfg.roce_wire == .verbs) {
            if (cfg.roce_hca) |text| {
                n_specs = try roce_verbs.parseSpecs(text, &specs);
            } else if (Verbs.open(null)) |vb| {
                var v = vb;
                defer v.close();
                n_specs = roce_verbs.autoSpecs(&v, &specs, cfg.roce_hcas);
            } else |_| {}
        }
        const opts: roce.Options = .{ .wire = if (cfg.roce_wire == .shm) .shm else .verbs, .hcas = specs[0..n_specs], .traffic_class = cfg.roce_tc, .cpu = cfg.roce_cpu, .max_bytes = cfg.roceSlotBytes(), .timeout_ns = cfg.mailbox_timeout_ns, .fast = cfg.roce_fast, .blocks = cfg.roce_blocks };
        const r = roce.Roce.init(d, b, image, opts) catch |e| return s.roceFallback(e);
        s.roce = r;
        if (cfg.roce_chunked_bytes > 0 and !r.chunk_abi) {
            std.log.warn("TF_TP_ROCE_CHUNKED_KB={d}: the tp_roce kernel image has no chunked ABI (tp_roce_chunk_abi); refusing to start", .{cfg.roce_chunked_bytes / 1024});
            r.deinit();
            return error.ChunkAbiMissing;
        }
        s.route = .{ .small = .{ .roce = r }, .nccl = &s.nccl, .max_bytes = cfg.mailbox_max_bytes, .var_max_bytes = cfg.roce_var_bytes, .chunked_bytes = cfg.roce_chunked_bytes };
        const good = probe(s, d) catch false;
        var blobs: [bootstrap.max_ranks][]const u8 = undefined;
        var buf: [bootstrap.max_ranks]u8 = undefined;
        const all = try b.allGather(&.{@intFromBool(good)}, &buf, &blobs);
        for (all) |x| if (x[0] != 1) {
            r.deinit();
            s.roce = null;
            return s.roceFallback(error.ProbeMismatch);
        };
    }

    /// `Collective.varAgreed`: whether every rank's transport moves device lengths (`varFits(16)` on all), one all-gather
    /// over the bootstrap a session, kept on the route (NCCL alone never does).
    fn agreeVar(s: *Session, b: *Bootstrap) !void {
        var blobs: [bootstrap.max_ranks][]const u8 = undefined;
        var buf: [bootstrap.max_ranks]u8 = undefined;
        const all = try b.allGather(&.{@intFromBool(s.comm().varFits(16))}, &buf, &blobs);
        var ok = true;
        for (all) |x| ok = ok and x[0] == 1;
        if (s.mail != null or s.roce != null) s.route.var_agreed = ok;
    }

    fn roceFallback(s: *Session, e: anyerror) !void {
        if (s.cfg.roce_fallback == .@"error") return e;
        if (s.cfg.rank == 0) std.log.warn("[tensorfold] tp: RoCE unavailable ({t}); every rank uses NCCL (TF_TP_ROCE_FALLBACK=error refuses instead)", .{e});
    }

    /// One 4 KiB all-gather and one bf16 sum through RoCE and through NCCL: the bytes must be equal.
    fn probe(s: *Session, d: *const cuda.Driver) !bool {
        const n = 4096;
        // input [0, n), gathers [n, 3n) and [3n, 5n), sums [5n, 6n) and [6n, 7n)
        var a = try cuda.DeviceBuffer.alloc(d, 7 * n);
        defer a.free();
        var stream = try cuda.Stream.init(d, true);
        defer stream.deinit();
        var bytes: [n]u8 = undefined;
        reference.fill(&bytes, .bf16, s.cfg.rank, 0x7072_6f62);
        try a.uploadAsync(0, &bytes, stream.handle);
        const fast = s.route.iface();
        const slow = s.nccl.iface();
        try fast.allGather(a.ptr, a.ptr + n, n, .u8, stream.handle);
        try slow.allGather(a.ptr, a.ptr + 3 * n, n, .u8, stream.handle);
        try fast.allReduce(a.ptr, a.ptr + 5 * n, n / 2, .bf16, .sum, stream.handle);
        try slow.allReduce(a.ptr, a.ptr + 6 * n, n / 2, .bf16, .sum, stream.handle);
        try stream.synchronize();
        try fast.check();
        var got: [6 * n]u8 = undefined;
        try a.download(n, &got);
        const gathers = std.mem.eql(u8, got[0 .. 2 * n], got[2 * n .. 4 * n]);
        const sums = std.mem.eql(u8, got[4 * n .. 5 * n], got[5 * n .. 6 * n]);
        if (!gathers or !sums) std.log.warn("[tensorfold] tp roce probe: gather {s}, sum {s} against NCCL", .{ if (gathers) "equal" else "DIFFERS", if (sums) "equal" else "DIFFERS" });
        return gathers and sums;
    }

    fn mailboxPossible(b: *Bootstrap, image: []const u8) !bool {
        if (b.world != 2) return false;
        var id_buf: [64]u8 = undefined;
        const id = sock.hostId(&id_buf);
        var mine: [65]u8 = @splat(0);
        mine[0] = @intFromBool(image.len > 0);
        @memcpy(mine[1..][0..id.len], id);
        var blobs: [bootstrap.max_ranks][]const u8 = undefined;
        var buf: [2 * 65]u8 = undefined;
        const all = try b.allGather(&mine, &buf, &blobs);
        return all[0][0] == 1 and std.mem.eql(u8, all[0], all[1]);
    }

    fn abortHook(ctx: *anyopaque, _: []const u8) void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        s.comm().abort();
    }

    /// The communicator the model uses: NCCL, or the hybrid route over the mailbox / RoCE.
    pub fn comm(s: *Session) Collective {
        if (s.mail != null or s.roce != null) return s.route.iface();
        return s.nccl.iface();
    }

    /// This rank's transport progress: every word is an atomic host load, so it touches no GPU call, no lock
    /// and no allocation (the sampler's contract). `at`: this rank's monotonic clock at the send. Counters are
    /// cumulative since this rank's boot. One writer — the fate thread, once per report period.
    /// This rank's counters, pure: every word an atomic host load, no GPU call, no lock, no allocation. Two threads
    /// read it (the fate reporter on its report cadence, the peer-wait clock on its poll); nothing here mutates.
    fn counters(s: *Session) fate.Progress {
        var p: fate.Progress = .{
            .transport = s.comm().kind(),
            .plans = s.plan.received.load(.acquire),
        };
        if (s.roce) |r| {
            const x = r.progress();
            p.doorbell = x.doorbell;
            p.completed = x.completed;
            p.proxy_seq = x.proxy_seq;
            p.faulted = x.faulted;
            p.fault_seq = x.fault_seq;
            p.fault_why = x.fault_why;
            p.waited_us = x.waited_us;
            p.proxy_failed = x.proxy_failed;
            p.aborted = x.aborted;
            p.link_idle = x.link_idle;
            p.peer_flags = x.flags;
        }
        if (s.mail) |*m| {
            const x = m.progress(s.cfg.rank);
            p.faulted = p.faulted or x.failed;
            p.aborted = p.aborted or x.aborted;
            if (x.failed) {
                p.fault_seq = x.seq;
                p.fault_why = x.why;
                p.waited_us = x.waited_us;
            }
        }
        return p;
    }

    pub fn progress(s: *Session, at: u64) fate.Progress {
        var p = s.counters();
        if (s.last_sample) |l| {
            p.report_seq = l.report_seq + 1;
            const moved = p.plans != l.plans or p.doorbell != l.doorbell or p.completed != l.completed or
                p.proxy_seq != l.proxy_seq or p.fault_seq != l.fault_seq or p.faulted != l.faulted;
            if (moved) s.moved_at = at;
            p.frozen_ns = at -| s.moved_at;
        } else {
            p.report_seq = 1;
            s.moved_at = at;
        }
        s.last_sample = p;
        return p;
    }

    fn progressFrame(ctx: *anyopaque, out: *fate.Progress) void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        out.* = s.progress(sock.nowNs());
    }

    /// Rank 0's plans sent, read at each progress frame's arrival: plans the leader wrote that the follower has
    /// not read is a wedge's size.
    fn leaderSent(ctx: *anyopaque) u64 {
        const s: *const Session = @ptrCast(@alignCast(ctx));
        return s.plan.sent.load(.acquire);
    }

    /// The peer-wait clock's sample of the same counters. `plans` is the role's own: a leader's progress is its
    /// sends, so its clock never counts the silence of a read it is not making.
    fn peerProgress(ctx: *anyopaque, out: *fate.Progress) void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        out.* = s.counters();
        if (s.cfg.rank == 0) out.plans = s.plan.sent.load(.acquire);
    }

    /// The plan link asks for a deadline once per read and once per send; outside a round, or off the knob, this is
    /// null: the wait is forever, as it was before this layer.
    fn peerDeadline(ctx: *anyopaque) ?u64 {
        const s: *Session = @ptrCast(@alignCast(ctx));
        return s.peer.deadline();
    }

    /// A plan wait expired: the link named it, and the clock takes the pair down once (idempotent).
    fn peerDeadlineFired(ctx: *anyopaque, cause: peerwait.Cause) void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        _ = s.peer.expire(cause);
    }

    pub fn peerOn(s: *Session) bool {
        return s.cfg.peer_wait_ns > 0;
    }

    /// The round bracket, on the host's stamps (rank 0) or the leader's markers (the followers): the clock counts
    /// silence only inside a round, so an idle pair never ages toward an expiry.
    pub fn peerRoundBegin(s: *Session, round_no: u64) void {
        s.peer.beginRound(round_no);
    }

    pub fn peerRoundEnd(s: *Session) void {
        s.peer.endRound();
    }

    /// What the round admitted: an admitted prompt is legitimate in-round work, so it widens the bound.
    pub fn peerRows(s: *Session, rows: u64) void {
        s.peer.setRows(rows);
    }

    pub fn peerPhase(s: *Session, p: peerwait.Phase) void {
        s.peer.setPhase(p);
    }

    /// A follower's round watch giving up on a round the leader opened and never closed: this rank's own witness,
    /// named with the round its markers carry. The silence clock's `armed` gate does not apply here: the watch's
    /// evidence stands by itself.
    pub fn peerRoundWatchExit(s: *Session) void {
        _ = s.peer.fault(.{ .round_watch = .{ .rank = s.cfg.rank, .round = s.peer.round_no.load(.acquire) } });
    }

    /// This rank failed inside a round: every rank goes down (exit 70). Without fail-fast it only aborts.
    pub fn fatal(s: *Session, where: []const u8, err: anyerror) void {
        if (s.fate) |*f| return f.fatal(where, err);
        s.comm().abort();
    }

    /// A clean stop of every rank, announced by rank 0 (Forward.stop) before it closes: the followers end with no
    /// fail-fast exit; elsewhere this rank's own hang-ups are expected. Without fail-fast: nothing.
    pub fn announceStop(s: *Session) void {
        if (s.fate) |*f| f.announceStop();
    }

    /// This rank failed after `open`, before serving (a model's load, its assets, its buffers): the other ranks learn
    /// why and go down (fail-fast), and nothing of the session runs on once its owner frees it (the fate thread read a
    /// freed session: rank 1's segfault in fate.report, Spark 2026-10-09).
    pub fn bootFailed(s: *Session, err: anyerror) void {
        if (s.fate) |*f| f.failBoot("boot", err);
        s.close();
    }

    /// A normal stop: every rank calls it (rank 0 first marks the stop as expected with `fate.expectStop`).
    pub fn close(s: *Session) void {
        // a failed transport: every rank goes down (the peers stop posting) before this one lets go of anything
        if (s.roce) |r| if (r.failed()) if (s.fate) |*f| f.fatal("transport", error.BackendFailed);
        s.peer.close(); // the clock thread reads the plan counters: joined before the link and the fate channel go
        if (s.fate) |*f| f.close();
        s.plan.close();
        if (s.roce) |r| r.deinit();
        if (s.mail) |*m| m.deinit();
        s.nccl.deinit();
        s.lib.close();
    }
};

test {
    _ = collective;
    _ = reference;
    _ = config;
    _ = sock;
    _ = bootstrap;
    _ = mailbox;
    _ = roce_verbs;
    _ = planlink;
    _ = @import("roce_test.zig");
    _ = @import("roce_inflight_test.zig");
    _ = @import("host_test.zig");
    _ = @import("fate_test.zig");
    _ = @import("peerwait_test.zig");
    _ = @import("planlink_stall_test.zig");
}
