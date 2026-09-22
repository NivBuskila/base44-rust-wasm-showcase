import { describe, expect, it } from "vitest";

import { GpuRenderer } from "./gpu/renderer";
import { Renderer } from "./render/renderer";

/**
 * The contract both backends owe the app.
 *
 * `SceneRenderer` is a compile-time interface, and that is weaker than it looks
 * here: each backend is constructed behind a capability gate (`gpu/index.ts`), so
 * on any given machine only one of the two ever runs. A method the WebGL2 path
 * quietly lost, or a WebGPU-only addition the app starts depending on, is
 * therefore something a user discovers on hardware nobody tested — the typecheck
 * is satisfied by whichever object the caller holds, and there is no run of the
 * other one to contradict it.
 *
 * Neither class can be *instantiated* here: constructing one wants a real canvas
 * with a live GL or WebGPU context, which jsdom has neither of. What is checkable
 * without a device is the shape — the surface the frame loop, the performance
 * governor and the diagnostics call into — so that is what this pins, on both
 * prototypes at once. Anything about pixels stays a browser job (see AGENTS.md).
 *
 * It cannot go further and forbid *extra* public methods: TypeScript's `private`
 * is erased, so every internal helper is an own property of the prototype and
 * such a check would only count them.
 */

/** Every member of `SceneRenderer` that is a method, and whether it is optional. */
const CONTRACT: ReadonlyArray<{ name: string; optional?: boolean }> = [
  { name: "render" },
  { name: "setVideo" },
  { name: "setQualityScale" },
  { name: "sampleLuminance" },
  { name: "dispose" },
  // Only the WebGPU backend owns the pool, so the WebGL2 one may omit this.
  { name: "particlesAlive", optional: true },
];

const BACKENDS: ReadonlyArray<{
  label: string;
  ctor: new (...a: never[]) => unknown;
}> = [
  { label: "webgl2", ctor: Renderer as unknown as new () => unknown },
  { label: "webgpu", ctor: GpuRenderer as unknown as new () => unknown },
];

describe.each(BACKENDS)("$label backend", ({ ctor }) => {
  const proto = ctor.prototype as Record<string, unknown>;

  it.each(CONTRACT.filter((m) => !m.optional))("implements %s", ({ name }) => {
    expect(typeof proto[name]).toBe("function");
  });

  it("keeps optional members callable when present", () => {
    for (const { name, optional } of CONTRACT) {
      if (!optional) continue;
      if (name in proto) expect(typeof proto[name]).toBe("function");
    }
  });

  it("accepts the governor's quality range without throwing on the shape", () => {
    // The governor calls `setQualityScale` on whichever backend is live, so both
    // must take exactly one argument — an arity drift here is a silently ignored
    // quality tier on one backend only.
    expect((proto.setQualityScale as (s: number) => void).length).toBe(1);
    expect((proto.render as (f: unknown) => void).length).toBe(1);
  });
});

describe("the WebGPU backend", () => {
  it("is the one that owns the particle pool", () => {
    expect("particlesAlive" in GpuRenderer.prototype).toBe(true);
  });
});
