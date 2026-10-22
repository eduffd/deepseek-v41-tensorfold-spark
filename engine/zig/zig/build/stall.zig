//! The pair's one allowance (zig/src/core/stall.zig) as a module of its own, so `tp`'s peer-wait clock and the engine
//! seam's `/health` gap read the same `Allowance`.
//!
//! `stall.zig` belongs to exactly one module inside one binary, so a recipe builds it once per target and hands the same
//! module to every module that reaches it by name (`zig/build/tp.zig` and every `engine_api`). A second `createModule`
//! for the same file in one artifact is what makes the shipped exes refuse to link:
//! `error: file exists in modules 'engine_api' and 'stall'`. Only `zig build install` links both, and it is part of every
//! commit's check, so a green suite implies the pair's binaries link.

const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stall.zig"), .target = target, .optimize = optimize });
}

/// `test-stall`: the watch's cadence, its exit, and the one allowance. A module's tests run only where that module is a
/// root, so once stall.zig stopped being a file `engine_api` reached by path, its five tests needed this root: without
/// the step they compile and never run, in a suite that stays green.
pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/stall.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true,
    }) }));
    b.step("test-stall", "The pair's stall watch and allowance: the report cadence, exit 70 at the exit knob, the hook that names the round, and the nanoseconds a clock arms on").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
