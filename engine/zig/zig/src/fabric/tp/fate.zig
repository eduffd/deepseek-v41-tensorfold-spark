//! Fail fast on every rank (the Python engine's G16 design): a failure or a vanished rank ends all ranks with exit 70 about a second later.
const std = @import("std");
const sock = @import("sock.zig");
const Config = @import("config.zig").Config;
const posix = std.posix;
/// Linux POLLRDHUP: the peer shut its side down (std has no name for it).
const pollrdhup: i16 = 0x2000;

pub const exit_code: u8 = 70;
const magic: u32 = 0x46463431;
const kind_mem: u32 = 1;
const kind_fail: u32 = 2;
/// rank 0's clean stop: the hang-up that follows is no failure (sent before rank 0 closes its sockets)
const kind_stop: u32 = 3;
/// a rank's transport progress, its counters and ages: telemetry, never a control frame
const kind_progress: u32 = 4;
const Frame = extern struct { magic: u32, kind: u32, value: i64 };

pub const max_ranks = 64;
pub const max_hooks = 8;

pub const Kind = @import("collective.zig").Kind;

/// How a failed wait ended. The wire carries its int code; `decode` is the only reader of that code, so a
/// reader's `@enumFrom` never sees an unvalidated int. `unknown`: the wire names a fault no shipped reader explains.
pub const FaultWhy = enum(u64) { none = 0, abort = 1, timeout = 2, unknown = 3 };

/// A rank's transport progress as the channel carries it. Producer: `tp.Session` (the reporting rank).
/// Carrier: Fate. Rank-0 store: Fate. Reader: the family's read hook. Counters are cumulative since the
/// reporting rank's boot; a watchdog derives deltas from consecutive polls. Fate decides nothing: it frames,
/// stores, ages, and reports.
pub const Progress = struct {
    transport: Kind,
    /// plan frames this rank received: the universal progress signal, on every backend
    plans: u64 = 0,
    doorbell: u64 = 0,
    completed: u64 = 0,
    /// the proxy's last posted doorbell seq
    proxy_seq: u64 = 0,
    /// ns THIS rank's counters have not moved, measured at send time on this rank's clock. An age, not a
    /// stamp: a late report ages by itself, so a dropped frame lengthens a freeze and can never hide one.
    frozen_ns: u64 = 0,
    /// this rank's own frame counter, +1 each report; rank 0 reads the jumps as drops
    report_seq: u64 = 0,
    faulted: bool = false,
    fault_seq: u64 = 0,
    fault_why: FaultWhy = .none,
    waited_us: u64 = 0,
    proxy_failed: bool = false,
    aborted: bool = false,
    /// every post of this rank reaped
    link_idle: bool = false,
    /// this rank's view of the leader's doorbell (a half-wedge seen from the other side)
    peer_flags: [2]u64 = .{ 0, 0 },
};

/// The wire encoding, private to this module. `transport` is the code of `collective.Kind`, `fault_why` the
/// code of `Progress.fault_why`; the four flags ride `bits`. The reader validates both codes in `readFrame`
/// and stores only the decoded ones, so no raw wire int ever reaches a reader's enum (`@enumFrom` of a
/// runtime int is a panic in `.debug` and UB in `.fast`).
const ProgressWire = extern struct {
    plans: u64,
    doorbell: u64,
    completed: u64,
    proxy_seq: u64,
    frozen_ns: u64,
    report_seq: u64,
    fault_seq: u64,
    waited_us: u64,
    peer_flag0: u64,
    peer_flag1: u64,
    transport: u32,
    fault_why: u32,
    /// 1 faulted, 2 proxy_failed, 4 aborted, 8 link_idle
    bits: u32,
};

pub const wire_bytes: usize = @sizeOf(ProgressWire);

fn named(names: []const []const u8, name: []const u8) bool {
    inline for (names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

comptime {
    const pf: []const []const u8 = @typeInfo(Progress).@"struct".field_names;
    const wf: []const []const u8 = @typeInfo(ProgressWire).@"struct".field_names;
    for (pf) |name|
        if (!named(&.{ "faulted", "proxy_failed", "aborted", "link_idle", "peer_flags" }, name) and !named(wf, name))
            @compileError("ProgressWire drops the Progress field " ++ name);
    for (wf) |name|
        if (!named(&.{ "bits", "peer_flag0", "peer_flag1" }, name) and !named(pf, name))
            @compileError("ProgressWire carries a field Progress lacks: " ++ name);
    std.debug.assert(@sizeOf(ProgressWire) == 96);
    std.debug.assert(@sizeOf(Frame) + @sizeOf(ProgressWire) == 112);
}

fn encode(p: Progress) ProgressWire {
    return .{
        .plans = p.plans,
        .doorbell = p.doorbell,
        .completed = p.completed,
        .proxy_seq = p.proxy_seq,
        .frozen_ns = p.frozen_ns,
        .report_seq = p.report_seq,
        .fault_seq = p.fault_seq,
        .waited_us = p.waited_us,
        .peer_flag0 = p.peer_flags[0],
        .peer_flag1 = p.peer_flags[1],
        .transport = @intCast(@backingInt(p.transport)),
        .fault_why = @intCast(@backingInt(p.fault_why)),
        .bits = @intCast(@as(u32, @intFromBool(p.faulted)) | (@as(u32, @intFromBool(p.proxy_failed)) << 1) |
            (@as(u32, @intFromBool(p.aborted)) << 2) | (@as(u32, @intFromBool(p.link_idle)) << 3)),
    };
}

/// The boundary parse, the only decode of a wire code. A transport code outside the five kinds refuses the
/// frame; a fault whose `why` no shipped reader explains reads as a timeout (`mailbox.check` and
/// `roce.check` both read it that way), and a code above the known ones is a broken kernel made visible.
fn decode(w: ProgressWire) ?Progress {
    const transport = switch (w.transport) {
        0 => Kind.nccl,
        1 => .mailbox,
        2 => .hybrid,
        3 => .roce,
        4 => .host,
        else => return null,
    };
    const faulted = w.bits & 1 != 0;
    return .{
        .transport = transport,
        .plans = w.plans,
        .doorbell = w.doorbell,
        .completed = w.completed,
        .proxy_seq = w.proxy_seq,
        .frozen_ns = w.frozen_ns,
        .report_seq = w.report_seq,
        .faulted = faulted,
        .fault_seq = w.fault_seq,
        .fault_why = if (!faulted) .none else switch (w.fault_why) {
            1 => .abort,
            0, 2 => .timeout,
            else => .unknown,
        },
        .waited_us = w.waited_us,
        .proxy_failed = w.bits & 2 != 0,
        .aborted = w.bits & 4 != 0,
        .link_idle = w.bits & 8 != 0,
        .peer_flags = .{ w.peer_flag0, w.peer_flag1 },
    };
}

// the codes `decode` accepted, read back out of the slot: total by construction, so each `unreachable` is
// the compiler's guarantee and not a runtime path
fn kindOf(code: u32) Kind {
    return switch (code) {
        0 => .nccl,
        1 => .mailbox,
        2 => .hybrid,
        3 => .roce,
        4 => .host,
        else => unreachable,
    };
}

fn whyOf(code: u32) FaultWhy {
    return switch (code) {
        0 => .none,
        1 => .abort,
        2 => .timeout,
        3 => .unknown,
        else => unreachable,
    };
}

/// A rank's own progress sampler. `read` must touch no lock, no GPU call and no allocation: it loads
/// host-mapped atomic words and the atomic plan counter.
pub const ProgressSource = struct { ctx: *anyopaque, read: *const fn (ctx: *anyopaque, out: *Progress) void };
/// The leader's own plan-sent counter, read at each progress frame's arrival: it makes a wedge's size visible
/// (plans the leader sent that the follower has not read). Null on ranks that are not the leader.
pub const LeaderSource = struct { ctx: *anyopaque, read: *const fn (ctx: *anyopaque) u64 };

/// One progress frame written as the sender would not write it, for a test of the boundary parse: `declared`
/// is the header's length, the payload is a contract-shaped wire carrying `raw`. One send, so a frame the
/// reader accepts leaves the stream aligned.
pub const RawWire = extern struct { plans: u64 = 0, transport: u32 = 0, fault_why: u32 = 0, bits: u32 = 0 };
pub fn writeProgressRaw(fd: sock.Fd, declared: usize, raw: RawWire) bool {
    const w: ProgressWire = .{
        .plans = raw.plans,
        .doorbell = 0,
        .completed = 0,
        .proxy_seq = 0,
        .frozen_ns = 0,
        .report_seq = 0,
        .fault_seq = 0,
        .waited_us = 0,
        .peer_flag0 = 0,
        .peer_flag1 = 0,
        .transport = raw.transport,
        .fault_why = raw.fault_why,
        .bits = raw.bits,
    };
    const head: Frame = .{ .magic = magic, .kind = kind_progress, .value = @intCast(declared) };
    var buf: [16 + 96]u8 = undefined;
    @memcpy(buf[0..16], std.mem.asBytes(&head));
    @memcpy(buf[16..], std.mem.asBytes(&w));
    return sock.trySend(fd, buf[0 .. 16 + @min(declared, 96)]);
}

/// Reads and discards `n` bytes so a frame's payload ends where the next header starts. `readFrame` clamps
/// the declared length, so this loop is bounded by that clamp.
fn drain(fd: sock.Fd, n: usize, deadline: u64) void {
    var buf: [256]u8 = undefined;
    var left: usize = n;
    while (left > 0) {
        const take = @min(left, buf.len);
        sock.recvExact(fd, buf[0..take], deadline) catch return;
        left -= take;
    }
}

/// A callback with its context (an abort of a collective, rank 0's request failing); `reason` lives until exit.
pub const Hook = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, reason: []const u8) void };

pub const Fate = struct {
    rank: u32,
    world: u32,
    fds: [max_ranks]sock.Fd,
    grace_ns: u64,
    report_ns: u64,
    /// This rank's usable memory in bytes for the MEM report (default: MemAvailable).
    usable: *const fn () u64 = memAvailable,
    /// The process exit; tests replace it (it may return there).
    exit_fn: *const fn (u8) void = exitNow,
    hooks: [max_hooks]Hook = undefined,
    n_hooks: std.atomic.Value(u32) = .init(0),
    dead: std.atomic.Value(bool) = .init(false),
    closing: std.atomic.Value(bool) = .init(false),
    /// Set once the exit was asked for (tests read it; the process is gone in serving).
    exited: std.atomic.Value(bool) = .init(false),
    reason_buf: [512]u8 = undefined,
    reason_len: usize = 0,
    send_lock: std.atomic.Value(bool) = .init(false),
    /// Rank 0: each rank's last report (bytes) and its monotonic time.
    peer_mem: [max_ranks]std.atomic.Value(u64) = @splat(.init(0)),
    peer_mem_at: [max_ranks]std.atomic.Value(u64) = @splat(.init(0)),
    /// The progress sampler, armed on every rank before `start` (a field set before the thread spawns: no
    /// race). Only a follower's `watchK` reports, so on rank 0 it is armed and unused.
    progress_src: ?ProgressSource = null,
    /// Rank 0 only: its own plan-sent source, armed the same way. Null on the followers.
    leader_src: ?LeaderSource = null,
    wake: [2]sock.Fd = .{ -1, -1 },
    thread: ?std.Thread = null,
    prog: [max_ranks]Slot = @splat(.{}),

    /// Rank 0's per-peer progress snapshot. Writer: the fate thread alone. Reader: any thread. A lock-free
    /// seqlock — the stall snapshot's per-field atomics, upgraded to consistent snapshots: seq odd means a
    /// write is in flight and a reader retries until a read comes out undivided. The writer never blocks and
    /// takes no lock; a reader's retry is a bounded spin (the write window is a handful of atomic stores with
    /// no syscall and no allocation), and its end hands out the frozen values with the read counted
    /// torn-suspect in `bad`.
    const Slot = struct {
        seq: std.atomic.Value(u32) = .init(0),
        plans: std.atomic.Value(u64) = .init(0),
        doorbell: std.atomic.Value(u64) = .init(0),
        completed: std.atomic.Value(u64) = .init(0),
        proxy_seq: std.atomic.Value(u64) = .init(0),
        frozen_ns: std.atomic.Value(u64) = .init(0),
        report_seq: std.atomic.Value(u64) = .init(0),
        /// frames the full socket buffer dropped (summed report_seq jumps)
        gaps: std.atomic.Value(u64) = .init(0),
        /// frames the boundary parse refused; `peerProgress` adds one for a torn-suspect read
        bad: std.atomic.Value(u64) = .init(0),
        frames_stored: std.atomic.Value(u64) = .init(0),
        /// the leader's plan-sent count at the last arrival
        leader_sent: std.atomic.Value(u64) = .init(0),
        /// last arrival where the reports kept pace with the leader's sends
        lockstep_at: std.atomic.Value(u64) = .init(0),
        fault_seq: std.atomic.Value(u64) = .init(0),
        waited_us: std.atomic.Value(u64) = .init(0),
        peer_flag0: std.atomic.Value(u64) = .init(0),
        peer_flag1: std.atomic.Value(u64) = .init(0),
        /// validated kind code | validated why code << 8 | flags << 16: written only by `readFrame`'s decode,
        /// which refuses out-of-range values, so every stored code is one of the enums' cases
        small: std.atomic.Value(u32) = .init(0),
        /// arrival stamp (monotonic ns); 0: never reported
        at: std.atomic.Value(u64) = .init(0),
    };

    /// The channel over the bootstrap's sockets (`Bootstrap.release`); `start` runs its thread.
    pub fn init(cfg: Config, fds: [max_ranks]sock.Fd) Fate {
        return .{ .rank = cfg.rank, .world = cfg.world, .fds = fds, .grace_ns = cfg.grace_ns, .report_ns = cfg.report_ns };
    }

    pub fn start(f: *Fate) !void {
        var p: [2]c_int = undefined;
        if (std.c.pipe2(&p, .{ .CLOEXEC = true }) != 0) return error.SocketFailed;
        f.wake = p;
        f.thread = if (f.rank == 0) try std.Thread.spawn(.{}, watch0, .{f}) else try std.Thread.spawn(.{}, watchK, .{f});
    }

    /// Runs `hook` when the ranks go down; collectives register their abort here.
    pub fn onFail(f: *Fate, hook: Hook) void {
        const i = f.n_hooks.load(.acquire);
        std.debug.assert(i < max_hooks);
        f.hooks[i] = hook;
        f.n_hooks.store(i + 1, .release);
    }

    /// This rank failed (`where`: what it was doing): every rank goes down. Returns only in tests.
    pub fn fatal(f: *Fate, where: []const u8, err: anyerror) void {
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "rank {d} failed in {s}: {t}", .{ f.rank, where, err }) catch "rank failed";
        f.die(msg, true);
    }

    /// This rank failed while booting (after the session came up, before serving): every rank goes down with its
    /// reason (`fatal`), and the channel's thread is stopped and joined before the caller frees what holds this Fate.
    /// In serving `fatal` exits the process; where it returns (tests), the thread may not outlive the memory.
    pub fn failBoot(f: *Fate, where: []const u8, err: anyerror) void {
        f.fatal(where, err);
        f.close();
    }

    /// A normal stop is coming (rank 0 before it tells the others to stop; the others when told): hang-ups that
    /// follow are no failure.
    pub fn expectStop(f: *Fate) void {
        f.closing.store(true, .release);
    }

    /// Rank 0's clean stop of every rank (the leader's `Forward.stop`): a STOP frame to each rank ahead of the hang-up
    /// its close brings, so a follower ends quietly whatever it is still doing; elsewhere `expectStop`. The channel is
    /// ordered: a follower reads the STOP before it can see rank 0's socket close.
    pub fn announceStop(f: *Fate) void {
        if (f.closing.swap(true, .acq_rel)) return;
        if (f.rank != 0) return;
        const h: Frame = .{ .magic = magic, .kind = kind_stop, .value = 0 };
        f.lock();
        defer f.unlock();
        for (f.fds[1..f.world]) |fd| if (fd >= 0) {
            _ = sock.trySend(fd, std.mem.asBytes(&h));
        };
    }

    /// A normal stop: the thread ends and the sockets close.
    pub fn close(f: *Fate) void {
        f.closing.store(true, .release);
        if (f.wake[1] >= 0) _ = std.c.write(f.wake[1], "x", 1);
        if (f.thread) |t| t.join();
        f.thread = null;
        for (&f.fds) |*fd| if (fd.* >= 0) {
            sock.close(fd.*);
            fd.* = -1;
        };
        for (f.wake) |fd| if (fd >= 0) sock.close(fd);
        f.wake = .{ -1, -1 };
    }

    /// Rank 0: rank `r`'s last reported usable bytes and the age of the report, or null before its first.
    pub fn peerMem(f: *const Fate, r: u32) ?struct { bytes: u64, age_ns: u64 } {
        const at = f.peer_mem_at[r].load(.acquire);
        if (at == 0) return null;
        return .{ .bytes = f.peer_mem[r].load(.acquire), .age_ns = sock.nowNs() -| at };
    }

    /// Rank 1 (or a test rank): arm the progress sampler. Not thread-safe; call before `start`.
    pub fn setProgress(f: *Fate, src: ProgressSource) void {
        f.progress_src = src;
    }

    /// Rank 0: arm its own plan-sent source. Not thread-safe; call before `start`.
    pub fn setLeaderProgress(f: *Fate, src: LeaderSource) void {
        f.leader_src = src;
    }

    /// Rank 0's read of peer `r`'s newest report: the counters, the ages, and the channel's own honesty
    /// counters. Null before its first frame — no evidence, so a reader shows no peer at all.
    pub const Sample = struct {
        prog: Progress,
        /// seconds since this frame's arrival on the reader's clock: the channel's age (a dropped frame ages
        /// the peer out: staleness is a fact, never a stall)
        age_ns: u64,
        /// the peer's counters have not moved this long, its age included
        frozen_s: f64,
        gaps: u64,
        bad: u64,
        /// reports the peer wrote
        frames_sent: u64,
        /// reports this rank accepted
        frames_received: u64,
        /// plans the leader sent that the peer has not read: a wedge's size, 0 in lockstep
        outstanding: u64,
        /// seconds since the pair last kept pace; null: never reported in lockstep
        since_lockstep_s: ?f64,
    };

    pub fn peerProgress(f: *const Fate, r: u32) ?Sample {
        const s: *Slot = @constCast(&f.prog[r]);
        var spins: u32 = 0;
        while (true) : (spins += 1) {
            const s1 = s.seq.load(.acquire);
            const small = s.small.load(.monotonic);
            const p: Progress = .{
                .plans = s.plans.load(.monotonic),
                .doorbell = s.doorbell.load(.monotonic),
                .completed = s.completed.load(.monotonic),
                .proxy_seq = s.proxy_seq.load(.monotonic),
                .frozen_ns = s.frozen_ns.load(.monotonic),
                .report_seq = s.report_seq.load(.monotonic),
                .fault_seq = s.fault_seq.load(.monotonic),
                .waited_us = s.waited_us.load(.monotonic),
                .peer_flags = .{ s.peer_flag0.load(.monotonic), s.peer_flag1.load(.monotonic) },
                .transport = kindOf(small & 0xff),
                .fault_why = whyOf((small >> 8) & 0xff),
                .faulted = small & (1 << 16) != 0,
                .proxy_failed = small & (1 << 17) != 0,
                .aborted = small & (1 << 18) != 0,
                .link_idle = small & (1 << 19) != 0,
            };
            const at = s.at.load(.monotonic);
            if (at == 0) return null;
            // every field this snapshot reports, including the channel's own counters, loads inside the
            // window the marker validates — the CAS below proves none of them moved
            const got = s.frames_stored.load(.monotonic);
            const lead = s.leader_sent.load(.monotonic);
            const lockstep = s.lockstep_at.load(.monotonic);
            const gaps = s.gaps.load(.monotonic);
            const bad = s.bad.load(.monotonic);
            // The second check is an Add-0 seq_cst RMW: a real RMW is a full barrier, and it fences
            // the field loads above. Zig 0.17 lowers a no-op CAS to a plain load, and a plain load
            // only orders the accesses after it, so torn field reads complete after its seq read and
            // pass validation as undivided. The marker is a counter, so a completed epoch reads back
            // s1 + 2 and fails the comparison.
            const s2 = s.seq.rmw(.Add, @as(u32, 0), .seq_cst);
            const age_ns = sock.nowNs() -| at;
            var snap: Sample = .{
                .prog = p,
                .age_ns = age_ns,
                .frozen_s = @as(f64, @floatFromInt(p.frozen_ns +| age_ns)) / 1e9,
                .gaps = gaps,
                .bad = bad,
                .frames_sent = p.report_seq,
                .frames_received = got,
                .outstanding = if (lead > p.plans) lead - p.plans else 0,
                .since_lockstep_s = if (lockstep != 0 and p.plans >= lead) @as(f64, @floatFromInt(age_ns)) / 1e9 else if (lockstep == 0) null else @as(f64, @floatFromInt(sock.nowNs() -| lockstep)) / 1e9,
            };
            if (s1 & 1 == 0 and s2 == s1) return snap;
            if (spins < (1 << 20)) {
                std.atomic.spinLoopHint();
                continue;
            }
            // The writer never closed this window. Today's writer is the fate thread, and it exits at
            // its loop's end, never mid-window; this arm covers a writer that does. Hand out the
            // frozen values with the read counted torn-suspect: the age and the freeze grow on the
            // reader's clock, so a wedged writer reads as staleness, never as a heal, and no reader
            // blocks past the spin.
            snap.bad += 1;
            return snap;
        }
    }

    /// The writer, the fate thread only. Zig 0.17 has no standalone fence, so the marker pair carries them:
    /// the marker is a COUNTER and each step is a `fetchAdd(1)` — a seq_cst RMW is a full barrier, so this
    /// rank's field stores cannot land before the odd marker. A bare `.release`/`.acquire` pair on a 0/1
    /// flag would leave the stores free to land before the marker, and a completed write invisible to a
    /// reader that checked the flag twice. On GB10's weak ordering, in the serving build's `.fast`, that
    /// accepts a torn snapshot as undivided and nothing traps.
    fn storeProgress(f: *Fate, peer: u32, p: Progress, at: u64) void {
        const s = &f.prog[peer];
        _ = s.frames_stored.fetchAdd(1, .monotonic);
        const prev = s.report_seq.load(.monotonic);
        if (prev != 0 and p.report_seq > prev + 1) _ = s.gaps.fetchAdd(p.report_seq - prev - 1, .monotonic);
        const sent = if (f.leader_src) |src| src.read(src.ctx) else s.leader_sent.load(.monotonic);
        _ = s.seq.fetchAdd(1, .seq_cst); // opens the window: s0 + 1, odd
        s.plans.store(p.plans, .monotonic);
        s.doorbell.store(p.doorbell, .monotonic);
        s.completed.store(p.completed, .monotonic);
        s.proxy_seq.store(p.proxy_seq, .monotonic);
        s.frozen_ns.store(p.frozen_ns, .monotonic);
        s.report_seq.store(p.report_seq, .monotonic);
        s.fault_seq.store(p.fault_seq, .monotonic);
        s.waited_us.store(p.waited_us, .monotonic);
        s.peer_flag0.store(p.peer_flags[0], .monotonic);
        s.peer_flag1.store(p.peer_flags[1], .monotonic);
        s.small.store(
            @intCast(@as(u32, @backingInt(p.transport)) | (@as(u32, @truncate(@backingInt(p.fault_why))) << 8) |
                (@as(u32, @intFromBool(p.faulted)) << 16) | (@as(u32, @intFromBool(p.proxy_failed)) << 17) |
                (@as(u32, @intFromBool(p.aborted)) << 18) | (@as(u32, @intFromBool(p.link_idle)) << 19)),
            .monotonic,
        );
        s.leader_sent.store(sent, .monotonic);
        s.at.store(at, .monotonic);
        _ = s.seq.fetchAdd(1, .release); // closes the window: s0 + 2, even — the counter never returns to
        // a value a reader that started earlier has already validated, so a completed write always divides it
        if (p.plans >= sent) s.lockstep_at.store(at, .monotonic); // plans against plans, the same unit
    }

    /// Test hook: opens peer `peer`'s window and never closes it, so a reader faces a writer killed
    /// mid-window. The reader's answer is a counted torn-suspect read, never a blocked caller.
    pub fn wedgeProgressSlot(f: *Fate, peer: u32) void {
        _ = f.prog[peer].seq.fetchAdd(1, .seq_cst);
    }

    pub fn reason(f: *const Fate) []const u8 {
        return f.reason_buf[0..f.reason_len];
    }

    fn lock(f: *Fate) void {
        while (f.send_lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn unlock(f: *Fate) void {
        f.send_lock.store(false, .release);
    }

    fn die(f: *Fate, why: []const u8, own: bool) void {
        if (f.dead.swap(true, .acq_rel)) {
            // the first caller exits the process; a second (the fate thread or a round thread) waits for it
            while (!f.exited.load(.acquire)) sock.sleepNs(10 * std.time.ns_per_ms);
            return;
        }
        const n = @min(why.len, f.reason_buf.len);
        @memcpy(f.reason_buf[0..n], why[0..n]);
        f.reason_len = n;
        const r = f.reason();
        std.log.warn("[tensorfold] tp fail-fast: rank {d}: {s}; exiting ({d}) in {d} ms so every rank restarts", .{ f.rank, r, exit_code, f.grace_ns / std.time.ns_per_ms });
        // aborts first: kernels waiting on the dead peer end now; a hook that blocks cannot hold the exit
        if (std.Thread.spawn(.{}, runHooks, .{ f, r })) |t| t.detach() else |_| runHooks(f, r);
        if (!f.closing.load(.acquire)) {
            const head: Frame = .{ .magic = magic, .kind = kind_fail, .value = @intCast(r.len) };
            f.lock();
            for (f.fds[0..f.world]) |fd| if (fd >= 0) {
                if (own or f.rank == 0) {
                    _ = sock.trySend(fd, std.mem.asBytes(&head));
                    _ = sock.trySend(fd, r);
                }
                sock.hangup(fd);
            };
            f.unlock();
        }
        sock.sleepNs(f.grace_ns);
        f.exited.store(true, .release);
        f.exit_fn(exit_code);
    }

    fn runHooks(f: *Fate, r: []const u8) void {
        const n = f.n_hooks.load(.acquire);
        for (f.hooks[0..n]) |h| h.call(h.ctx, r);
    }

    /// Reads one frame from `fd`: a MEM report, a FAIL (the ranks go down), or the end (the peer is gone).
    fn readFrame(f: *Fate, peer: u32, fd: sock.Fd) bool {
        var h: Frame = undefined;
        const deadline = sock.nowNs() + std.time.ns_per_s;
        sock.recvExact(fd, std.mem.asBytes(&h), deadline) catch {
            if (f.closing.load(.acquire)) return false;
            var buf: [128]u8 = undefined;
            f.die(std.fmt.bufPrint(&buf, "rank {d} closed the fate channel (it failed, exited or was killed)", .{peer}) catch "peer gone", false);
            return false;
        };
        if (h.magic != magic) {
            f.die("a frame that is not the fate channel's", false);
            return false;
        }
        if (h.kind == kind_stop) {
            // rank 0 stops every rank: its hang-up is next, and no failure
            f.closing.store(true, .release);
            return false;
        }
        if (h.kind == kind_mem) {
            f.peer_mem[peer].store(@intCast(@max(h.value, 0)), .release);
            f.peer_mem_at[peer].store(sock.nowNs(), .release);
            return true;
        }
        // Telemetry, and the only branch that keeps reading: a progress frame that will not parse is counted
        // and dropped, never fatal (a broken report has no business killing a serving pair, while control
        // frames keep dying it). It lands before the fail fallthrough below, which would consume the payload
        // as a reason and end the ranks. The payload is drained whatever its declared size, so a refused frame
        // leaves the stream aligned: an unaligned stream hits the magic check, which stays fatal, so a
        // telemetry bug could heal a healthy pair — the exact false heal this channel exists to close.
        if (h.kind == kind_progress) {
            const s = &f.prog[peer];
            const declared: usize = @intCast(std.math.clamp(h.value, 0, 4096));
            var raw: [wire_bytes]u8 align(@alignOf(ProgressWire)) = @splat(0); // bytesToValue reads the
            // wire's own fields out of this buffer: the alignment is part of that contract
            sock.recvExact(fd, raw[0..@min(declared, wire_bytes)], deadline) catch {
                _ = s.bad.fetchAdd(1, .monotonic);
                return true;
            };
            drain(fd, declared -| @min(declared, wire_bytes), deadline);
            if (declared != wire_bytes) {
                _ = s.bad.fetchAdd(1, .monotonic);
                return true;
            }
            const p = decode(std.mem.bytesToValue(ProgressWire, raw[0..wire_bytes])) orelse {
                _ = s.bad.fetchAdd(1, .monotonic);
                return true;
            };
            f.storeProgress(peer, p, sock.nowNs());
            return true;
        }
        var text: [512]u8 = undefined;
        const len: usize = @intCast(std.math.clamp(h.value, 0, text.len));
        sock.recvExact(fd, text[0..len], deadline) catch {};
        f.die(if (len > 0) text[0..len] else "a peer failed", false);
        return false;
    }

    fn watch0(f: *Fate) void {
        var p: [max_ranks + 1]std.c.pollfd = undefined;
        var who: [max_ranks + 1]u32 = undefined;
        var n: usize = 0;
        for (f.fds[1..f.world], 1..) |fd, r| if (fd >= 0) {
            p[n] = .{ .fd = fd, .events = posix.POLL.IN | pollrdhup, .revents = 0 };
            who[n] = @intCast(r);
            n += 1;
        };
        p[n] = .{ .fd = f.wake[0], .events = posix.POLL.IN, .revents = 0 };
        while (!f.dead.load(.acquire) and !f.closing.load(.acquire)) {
            if (std.c.poll(&p, @intCast(n + 1), -1) <= 0) continue;
            for (p[0..n], who[0..n]) |*q, r| if (q.revents != 0) {
                if (!f.readFrame(r, q.fd)) return;
            };
        }
    }

    fn watchK(f: *Fate) void {
        const fd = f.fds[0];
        var p = [_]std.c.pollfd{
            .{ .fd = fd, .events = posix.POLL.IN | pollrdhup, .revents = 0 },
            .{ .fd = f.wake[0], .events = posix.POLL.IN, .revents = 0 },
        };
        const ms: c_int = @intCast(@max(f.report_ns / std.time.ns_per_ms, 1));
        f.report(fd);
        while (!f.dead.load(.acquire) and !f.closing.load(.acquire)) {
            const ready = std.c.poll(&p, 2, ms);
            if (ready == 0) {
                f.report(fd);
                continue;
            }
            if (ready > 0 and p[0].revents != 0 and !f.readFrame(0, fd)) return;
        }
    }

    /// One MEM frame, never blocking (a full buffer drops it: the next comes one period later), then the
    /// rank's progress as ONE send of the whole 112-byte frame: a split send can land a header and drop its
    /// payload, and the reader would then parse the payload bytes as a header. The lock spans both sends —
    /// `die` and `announceStop` send their control frames on this same socket under it, and a frame
    /// interleaved with a FAIL header lands on the reader's magic check, which is fatal.
    fn report(f: *Fate, fd: sock.Fd) void {
        const h: Frame = .{ .magic = magic, .kind = kind_mem, .value = @intCast(f.usable()) };
        var buf: [@sizeOf(Frame) + wire_bytes]u8 = undefined;
        const src = f.progress_src;
        if (src) |s| {
            var p: Progress = undefined;
            s.read(s.ctx, &p);
            const head: Frame = .{ .magic = magic, .kind = kind_progress, .value = wire_bytes };
            @memcpy(buf[0..@sizeOf(Frame)], std.mem.asBytes(&head));
            @memcpy(buf[@sizeOf(Frame)..], std.mem.asBytes(&encode(p)));
        }
        f.lock();
        defer f.unlock();
        _ = sock.trySend(fd, std.mem.asBytes(&h));
        if (src != null) _ = sock.trySend(fd, &buf);
    }
};

fn exitNow(code: u8) void {
    std.c._exit(code);
}

test "the progress slot's snapshots stay consistent while its writer runs" {
    var f = Fate.init(.{ .world = 2, .rank = 0, .grace_ns = 0, .report_ns = std.time.ns_per_s }, @splat(-1));
    const Write = struct {
        fn run(x: *Fate) void {
            var i: u64 = 1;
            while (i < 200_000) : (i += 1) {
                // completed lands before doorbell, both at i: any torn mix reads completed > doorbell
                x.storeProgress(1, .{ .transport = .roce, .plans = i, .doorbell = i, .completed = i, .report_seq = i + 1 }, i);
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Write.run, .{&f});
    var reads: u64 = 0;
    while (reads < 20_000) : (reads += 1) {
        const s = f.peerProgress(1) orelse continue;
        std.debug.assert(s.prog.completed <= s.prog.doorbell);
        std.debug.assert(s.prog.report_seq == s.prog.plans + 1);
        std.debug.assert(s.prog.transport == .roce);
    }
    t.join();
}

test "the ledger stamps lockstep, names the outstanding gap, and counts a jump as drops" {
    var f = Fate.init(.{ .world = 2, .rank = 0, .grace_ns = 0, .report_ns = std.time.ns_per_s }, @splat(-1));
    var sent: u64 = 0;
    const Lead = struct {
        fn read(ctx: *anyopaque) u64 {
            return @as(*u64, @ptrCast(@alignCast(ctx))).*;
        }
    };
    f.leader_src = .{ .ctx = &sent, .read = Lead.read };
    const first = sock.nowNs();
    f.storeProgress(1, .{ .transport = .roce, .plans = 7, .report_seq = 1 }, first);
    const a = f.peerProgress(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 0), a.outstanding);
    try std.testing.expect(a.since_lockstep_s != null);
    sent = 10;
    f.storeProgress(1, .{ .transport = .roce, .plans = 7, .report_seq = 2 }, sock.nowNs());
    const b = f.peerProgress(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 3), b.outstanding);
    try std.testing.expect(b.since_lockstep_s != null and b.since_lockstep_s.? >= a.since_lockstep_s.?);
    sent = 40;
    f.storeProgress(1, .{ .transport = .roce, .plans = 7, .report_seq = 3 }, sock.nowNs());
    const c = f.peerProgress(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 33), c.outstanding);
    try std.testing.expectEqual(@as(u64, 0), c.gaps);
    // reports outrun the leader's plans on a healthy busy pair (two reports a second, plans a few a round):
    // lockstep is plans against plans, so a growing report count must never stamp the pair back into lockstep
    sent = 12;
    var tick: u64 = 4;
    while (tick <= 13) : (tick += 1) f.storeProgress(1, .{ .transport = .roce, .plans = 7, .report_seq = tick }, sock.nowNs());
    const g = f.peerProgress(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 5), g.outstanding);
    try std.testing.expect(g.since_lockstep_s != null and g.since_lockstep_s.? >= c.since_lockstep_s.?);
    f.storeProgress(1, .{ .transport = .roce, .plans = 8, .report_seq = 17 }, sock.nowNs());
    const d = f.peerProgress(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 3), d.gaps);
    try std.testing.expectEqual(@as(u64, 8), d.prog.plans);
}

test "a writer killed mid-window ages out as staleness, and the read is counted torn-suspect" {
    var f = Fate.init(.{ .world = 2, .rank = 0, .grace_ns = 0, .report_ns = std.time.ns_per_s }, @splat(-1));
    f.storeProgress(1, .{ .transport = .roce, .plans = 7, .report_seq = 1 }, sock.nowNs());
    const before = f.peerProgress(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 0), before.bad);
    f.wedgeProgressSlot(1);
    const after = f.peerProgress(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 1), after.bad);
    try std.testing.expectEqual(before.prog.plans, after.prog.plans);
    try std.testing.expect(after.age_ns >= before.age_ns);
}

/// MemAvailable from /proc/meminfo in bytes (0 when unreadable).
pub fn memAvailable() u64 {
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var buf: [4096]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return 0;
    const text = buf[0..@intCast(n)];
    const key = "MemAvailable:";
    const at = std.mem.indexOf(u8, text, key) orelse return 0;
    var it = std.mem.tokenizeAny(u8, text[at + key.len ..], " \n");
    const kb = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    return kb * 1024;
}
