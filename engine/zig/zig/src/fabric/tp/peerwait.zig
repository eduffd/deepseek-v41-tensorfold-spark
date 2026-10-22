//! One clock per rank that bounds how long it will wait inside a round for its peer: silence of this rank's own
//! progress counters plus a prompt allowance, disarmed at the round boundary. An expiry names the wait it expired on
//! and takes the pair down through Fate, the exit 70 every other fault route already uses.
const std = @import("std");
const sock = @import("sock.zig");
const fate = @import("fate.zig");

pub const knob = "TF_DSV41_PEER_WAIT_S";
pub const alias_knob = "GLM53_TF_PEER_WAIT_S";
/// The knob the proxy's in-flight bound takes its length from (roce.zig): a report that names the peer wait for a
/// transport's own timeout sends the operator to the wrong setting, so the wait site names the one it runs on.
pub const mailbox_knob = "TF_TP_MAILBOX_TIMEOUT_S";

/// Why a rank gave up waiting. One variant per wait site; the sentence a variant renders is the named line the
/// detecting rank prints and the reason the FAIL frame carries, verbatim.
pub const Cause = union(enum) {
    plan_recv: struct { rank: u32 = 0, round: u64 = 0, allow_s: f64 = 0, received: u64 = 0 },
    plan_send: struct { rank: u32 = 0, round: u64 = 0, allow_s: f64 = 0, sent: u64 = 0 },
    device_wait: struct { rank: u32 = 0, round: u64 = 0, allow_s: f64 = 0, what: []const u8 = "collective" },
    proxy_inflight: struct { rank: u32 = 0, round: u64 = 0, allow_s: f64 = 0, why: []const u8 = "doorbell" },
    transport_fault: struct { rank: u32 = 0, round: u64 = 0, allow_s: f64 = 0, fault: []const u8 = "" },
    round_watch: struct { rank: u32 = 0, round: u64 = 0, allow_s: f64 = 0 },

    /// The named sentence, comptime-fixed per variant: a fault is a name, never a spin.
    pub fn format(c: Cause, buf: []u8) error{NoSpaceLeft}![]const u8 {
        return switch (c) {
            .plan_recv => |p| std.fmt.bufPrint(buf, "plan read: rank {d} saw no progress on round {d} for {d} ms past {s} (frames read {d})", .{ p.rank, p.round, ms(p.allow_s), knob, p.received }),
            .plan_send => |p| std.fmt.bufPrint(buf, "plan send: rank {d} had a peer stop reading round {d}'s plan for {d} ms past {s} (plans sent {d})", .{ p.rank, p.round, ms(p.allow_s), knob, p.sent }),
            .device_wait => |p| std.fmt.bufPrint(buf, "device wait: rank {d} round {d} stayed in {s} for {d} ms past {s}", .{ p.rank, p.round, p.what, ms(p.allow_s), knob }),
            .proxy_inflight => |p| std.fmt.bufPrint(buf, "proxy in flight: rank {d} round {d}'s {s} never completed for {d} ms past {s}", .{ p.rank, p.round, p.why, ms(p.allow_s), mailbox_knob }),
            .transport_fault => |p| std.fmt.bufPrint(buf, "transport fault: rank {d} round {d}: {s} past {s}", .{ p.rank, p.round, p.fault, knob }),
            .round_watch => |p| std.fmt.bufPrint(buf, "round watch: rank {d} round {d} open {d} ms past {s}", .{ p.rank, p.round, ms(p.allow_s), knob }),
        };
    }

    /// A wait site knows its rank and its counters; the round and the allowance belong to the clock, and a zero
    /// means "fill me from yours". A site that runs on a bound of its own (the proxy's, off the mailbox knob) names
    /// its allowance here, and the clock leaves it alone. Rank is filled only when the site left it unset, and the
    /// clock's rank is its own rank, so rank 0 keeps its name: the leader is the rank that sends the plan.
    pub fn fill(c: *Cause, rank: u32, round: u64, allow_s: f64) void {
        switch (c.*) {
            inline else => |*p| {
                if (p.rank == 0) p.rank = rank;
                if (p.round == 0) p.round = round;
                if (p.allow_s == 0) p.allow_s = allow_s;
            },
        }
    }
};

fn ms(s: f64) u64 {
    if (!(s > 0.0)) return 0;
    return @intFromFloat(@min(s, 1e15) * 1000.0);
}

fn fieldType(comptime T: type, comptime name: []const u8) ?type {
    const s = @typeInfo(T).@"struct";
    for (s.field_names, s.field_types) |f, ty| if (std.mem.eql(u8, f, name)) return ty;
    return null;
}

// "a named fault, not a spin" as a type constraint: a variant without its rank and allowance fails the build.
comptime {
    const u = @typeInfo(Cause).@"union";
    for (u.field_types) |ty| {
        const rank = fieldType(ty, "rank") orelse @compileError("a Cause variant must carry rank:u32: " ++ @typeName(ty));
        if (rank != u32) @compileError("a Cause variant's rank must be u32: " ++ @typeName(ty));
        const allow = fieldType(ty, "allow_s") orelse @compileError("a Cause variant must carry allow_s:f64: " ++ @typeName(ty));
        if (allow != f64) @compileError("a Cause variant's allow_s must be f64: " ++ @typeName(ty));
    }
}

/// Where in the round this rank waits: the phase decides which expiry a silence becomes.
pub const Phase = enum(u8) { plan_recv, plan_send, device, proxy };

pub const Settings = struct {
    rank: u32 = 0,
    /// TF_DSV41_PEER_WAIT_S in ns: the silence bound. Zero turns the layer off entirely: no deadline, no thread.
    silence_ns: u64 = 0,
    /// rows / GLM53_TF_STALL_PREFILL_TPS widens the bound: an admitted prompt is legitimate in-round work.
    prefill_ns_per_row: u64 = 0,
    poll_ns: u64 = std.time.ns_per_s,
};

/// The report line goes through the sink: the session wires the log, tests record what was said.
pub const Sink = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, line: []const u8) void };

fn exitNow(code: u8) void {
    std.c._exit(code);
}

/// The snapshot the clock watches: the export's own counters, no new field. Report metadata (ages, report seqs)
/// is not progress, so it never resets the silence.
fn countersMoved(a: fate.Progress, b: fate.Progress) bool {
    return a.plans != b.plans or a.doorbell != b.doorbell or a.completed != b.completed or
        a.proxy_seq != b.proxy_seq or a.fault_seq != b.fault_seq or a.faulted != b.faulted;
}

pub const Clock = struct {
    settings: Settings = .{},
    now: *const fn () u64 = sock.nowNs,
    sink: ?Sink = null,
    progress: ?fate.ProgressSource = null,
    exit_fn: *const fn (code: u8) void = exitNow,
    fate: ?*fate.Fate = null,

    armed: std.atomic.Value(bool) = .init(false),
    expired_flag: std.atomic.Value(bool) = .init(false),
    round_no: std.atomic.Value(u64) = .init(0),
    rows: std.atomic.Value(u64) = .init(0),
    phase: std.atomic.Value(Phase) = .init(.plan_recv),
    moved_at: std.atomic.Value(u64) = .init(0),
    next_due: std.atomic.Value(u64) = .init(0),
    resample: std.atomic.Value(bool) = .init(false),
    stopping: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    // the counter snapshot the tick compares: the clock thread alone touches it (beginRound only flags a re-sample)
    last: fate.Progress = .{ .transport = .host },
    have_last: bool = false,

    pub fn boundNs(c: *const Clock) u64 {
        return c.settings.silence_ns + c.rows.load(.monotonic) *| c.settings.prefill_ns_per_row;
    }

    /// The next wait's absolute deadline: null off, outside a round, or already expired. A null is HEAD's path.
    pub fn deadline(c: *Clock) ?u64 {
        if (c.settings.silence_ns == 0 or !c.armed.load(.acquire) or c.expired()) return null;
        return c.moved_at.load(.monotonic) + c.boundNs();
    }

    pub fn beginRound(c: *Clock, round_no: u64) void {
        const t = c.now();
        c.round_no.store(round_no, .monotonic);
        c.rows.store(0, .monotonic);
        c.phase.store(.plan_recv, .monotonic);
        c.moved_at.store(t, .monotonic);
        c.next_due.store(t + c.boundNs() / 4, .monotonic);
        c.resample.store(true, .release);
        c.armed.store(true, .release);
    }

    pub fn endRound(c: *Clock) void {
        c.armed.store(false, .release);
    }

    pub fn setRows(c: *Clock, rows: u64) void {
        c.rows.store(rows, .release);
    }

    pub fn setPhase(c: *Clock, p: Phase) void {
        c.phase.store(p, .release);
    }

    pub fn expired(c: *Clock) bool {
        return c.expired_flag.load(.acquire);
    }

    pub fn start(c: *Clock) !void {
        if (c.settings.silence_ns == 0) return;
        c.thread = try std.Thread.spawn(.{}, loop, .{c});
    }

    pub fn close(c: *Clock) void {
        if (c.thread) |t| {
            c.stopping.store(true, .release);
            t.join();
            c.thread = null;
        }
    }

    fn loop(c: *Clock) void {
        while (!c.stopping.load(.acquire)) {
            _ = c.tick();
            sock.sleepNs(c.settings.poll_ns);
        }
    }

    /// One clock-thread pass: counter movement restarts the silence, a quarter of the bound reports, every bound
    /// after reports again, and silence past the bound expires. True when this call was the expiry.
    pub fn tick(c: *Clock) bool {
        if (c.settings.silence_ns == 0 or !c.armed.load(.acquire) or c.expired()) return false;
        const t = c.now();
        if (c.progress) |src| {
            var p: fate.Progress = .{ .transport = .host };
            src.read(src.ctx, &p);
            const fresh = c.resample.cmpxchgStrong(true, false, .acq_rel, .acquire) == null or !c.have_last;
            if (fresh or countersMoved(c.last, p)) {
                if (!fresh) {
                    // the counters moved: this rank is not the one that stopped, so the silence restarts here
                    c.moved_at.store(t, .monotonic);
                    c.next_due.store(t + c.boundNs() / 4, .monotonic);
                }
                c.last = p;
                c.have_last = true;
            }
        }
        if (c.expired()) return false;
        const bound = c.boundNs();
        const ma = c.moved_at.load(.monotonic);
        const late = if (t > ma) t - ma else 0;
        if (late >= bound) return c.expire(c.cause());
        if (late >= bound / 4 and t >= c.next_due.load(.monotonic)) {
            var sbuf: [448]u8 = undefined;
            c.report(c.cause().format(&sbuf) catch "peer wait expired");
            c.next_due.store(t + bound, .release);
        }
        return false;
    }

    fn cause(c: *Clock) Cause {
        const r = c.round_no.load(.monotonic);
        return switch (c.phase.load(.monotonic)) {
            .plan_recv => .{ .plan_recv = .{ .rank = c.settings.rank, .round = r, .received = c.last.plans } },
            .plan_send => .{ .plan_send = .{ .rank = c.settings.rank, .round = r, .sent = c.last.plans } },
            .device => .{ .device_wait = .{ .rank = c.settings.rank, .round = r } },
            .proxy => .{ .proxy_inflight = .{ .rank = c.settings.rank, .round = r } },
        };
    }

    /// The one expiry: the named line first, then the pair comes down. A round that closed inside its own report
    /// line is not expired: the clock re-checks `armed` after the line, the way stall.zig's watch does.
    pub fn expire(c: *Clock, cause0: Cause) bool {
        if (c.settings.silence_ns == 0) return false;
        if (c.expired_flag.swap(true, .acq_rel)) return false;
        var sbuf: [448]u8 = undefined;
        const sentence = c.speak(cause0, &sbuf);
        if (!c.armed.load(.acquire)) {
            c.expired_flag.store(false, .release);
            return false;
        }
        c.down(sentence);
        return true;
    }

    /// An expiry a witness saw from outside the silence (a follower's round watch, a proxy's in-flight bound): the
    /// same one latch and the same named line, no `armed` gate: the witness stands on its own evidence.
    pub fn fault(c: *Clock, cause0: Cause) bool {
        if (c.settings.silence_ns == 0) return false;
        if (c.expired_flag.swap(true, .acq_rel)) return false;
        var sbuf: [448]u8 = undefined;
        c.down(c.speak(cause0, &sbuf));
        return true;
    }

    /// Fill what the wait site could not know, render the sentence, print the named line.
    fn speak(c: *Clock, cause0: Cause, buf: []u8) []const u8 {
        var filled = cause0;
        filled.fill(c.settings.rank, c.round_no.load(.monotonic), @as(f64, @floatFromInt(c.boundNs())) / 1e9);
        const sentence = filled.format(buf) catch "peer wait expired";
        c.report(sentence);
        return sentence;
    }

    /// The exit this rank's fault route takes: Fate with the named reason (the FAIL frame carries it to the peer),
    /// or the bare 70 where fail-fast is off.
    fn down(c: *Clock, sentence: []const u8) void {
        if (c.fate) |f| f.fatal(sentence, error.PeerStalled) else c.exit_fn(fate.exit_code);
    }

    fn report(c: *Clock, sentence: []const u8) void {
        var lbuf: [512]u8 = undefined;
        const text = std.fmt.bufPrint(&lbuf, "[dsv41 peer-wait] {s}", .{sentence}) catch sentence;
        if (c.sink) |sk| sk.call(sk.ctx, text) else std.log.warn("{s}", .{text});
    }
};
