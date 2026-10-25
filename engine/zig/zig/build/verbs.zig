//! The verbs library's own invariants get a test root of their own, so the suite counts what it only imports.

const std = @import("std");

// a module's tests run only where it is a root; `verbs.zig` reaches `tp` by name alone, so its blocks ran in no recipe.
pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/fabric/verbs.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true,
    }) }));
    b.step("test-verbs", "fabric/verbs.zig's own invariants: bond device lists, the GID scan's link-local preference, and the TF_FABRIC_VERBS=1 device-list smoke").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
