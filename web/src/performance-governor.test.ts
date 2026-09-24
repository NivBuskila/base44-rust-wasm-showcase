import { describe, expect, it, vi } from 'vitest';

import { PerformanceGovernor, type QualityKnobs } from './performance-governor';

// Mirrors of the governor's private ladder constants. Pinned here on purpose:
// a change to how fast the ladder moves should fail a test, not slip through.
const SETTLE_MS = 750;
const SETTLE_FRAMES = 20;
const DOWN_MS = 500;
const UP_MS = 4000;

const BASE_PRESSURE = 28;
const BASE_PARTICLES = 120_000;

/** Frames needed to cover `ms` at a steady `fps` (the loop feeds real deltas). */
const framesFor = (ms: number, fps: number) => Math.ceil((ms * fps) / 1000);
/** Frames the settle window lasts at `fps`: time and frame count must both pass. */
const settleAt = (fps: number) => Math.max(SETTLE_FRAMES, framesFor(SETTLE_MS, fps));

function rig(basePressure = BASE_PRESSURE, baseParticles = BASE_PARTICLES) {
  const knobs: QualityKnobs = {
    setPressureIters: vi.fn(),
    setParticleCount: vi.fn(),
    setRenderScale: vi.fn(),
  };
  const governor = new PerformanceGovernor(knobs, basePressure, baseParticles);
  const feed = (fps: number, frames: number) => {
    for (let i = 0; i < frames; i++) governor.update(fps, 1000 / fps);
  };
  return { knobs, governor, feed };
}

/** Skips the boot settle window, during which fps is deliberately ignored. */
function settled() {
  const r = rig();
  r.feed(50, settleAt(50));
  return r;
}

describe('PerformanceGovernor', () => {
  it('starts at tier 0 and touches nothing while it settles', () => {
    const { knobs, governor, feed } = rig();
    feed(20, settleAt(20));
    expect(governor.level).toBe(0);
    expect(knobs.setRenderScale).not.toHaveBeenCalled();
    expect(knobs.setPressureIters).not.toHaveBeenCalled();
    expect(knobs.setParticleCount).not.toHaveBeenCalled();
  });

  it('one long first frame does not end the settle window', () => {
    const { governor } = rig();
    // A shader-compile stall, then frames that would drop a tier if judged.
    governor.update(1, 5000);
    // Judged on time alone the stall would have ended it and this would drop.
    for (let i = 0; i < SETTLE_FRAMES + framesFor(DOWN_MS, 40) - 1; i++) governor.update(40, 25);
    expect(governor.level).toBe(0);
    governor.update(40, 25);
    expect(governor.level).toBe(1);
  });

  it('drops one tier after sustained missed frames, not on a single hitch', () => {
    const { knobs, governor, feed } = settled();
    feed(40, framesFor(DOWN_MS, 40) - 1);
    expect(governor.level).toBe(0);
    feed(40, 1);
    expect(governor.level).toBe(1);
    // Tier 1: resolution first and hardest, pressure follows, pool untouched.
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(0.85);
    expect(knobs.setPressureIters).toHaveBeenLastCalledWith(Math.round(BASE_PRESSURE * 0.75));
    expect(knobs.setParticleCount).toHaveBeenLastCalledWith(BASE_PARTICLES);
  });

  it('a single stalled frame counts for a bounded time only', () => {
    const { governor } = settled();
    governor.update(45, 10_000);
    expect(governor.level).toBe(0);
  });

  it('skips a rung when frames are badly missed', () => {
    const { knobs, governor, feed } = settled();
    feed(20, framesFor(DOWN_MS, 20));
    expect(governor.level).toBe(2);
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(0.72);
  });

  it('a good frame resets the miss counter', () => {
    const { governor, feed } = settled();
    feed(40, framesFor(DOWN_MS, 40) - 1);
    feed(55, 1);
    feed(40, framesFor(DOWN_MS, 40) - 1);
    expect(governor.level).toBe(0);
  });

  it('settles after every change before judging again', () => {
    const { governor, feed } = settled();
    feed(40, framesFor(DOWN_MS, 40));
    expect(governor.level).toBe(1);
    // Still bad, but the fps average has not caught up with the new tier.
    feed(40, settleAt(40) + framesFor(DOWN_MS, 40) - 1);
    expect(governor.level).toBe(1);
    feed(40, 1);
    expect(governor.level).toBe(2);
  });

  it('reaches the last rung within a few seconds on a slow device', () => {
    const { governor, feed } = rig();
    // Boot settle, a two-rung drop, settle again, the last rung.
    const frames = 2 * settleAt(15) + 2 * framesFor(DOWN_MS, 15);
    feed(15, frames);
    expect(governor.level).toBe(3);
    expect((frames / 15) * 1000).toBeLessThan(5000);
  });

  it('never drops below the last rung and keeps the absolute floors', () => {
    // Bases so small that every fraction would go under the floors.
    const { knobs, governor, feed } = rig(12, 3000);
    feed(10, 20 * settleAt(10));
    expect(governor.level).toBe(3);
    expect(knobs.setPressureIters).toHaveBeenLastCalledWith(10);
    expect(knobs.setParticleCount).toHaveBeenLastCalledWith(2000);
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(0.6);
  });

  it('climbs back slowly and only from sustained headroom', () => {
    const { knobs, governor, feed } = settled();
    feed(40, framesFor(DOWN_MS, 40));
    feed(40, settleAt(40));
    expect(governor.level).toBe(1);

    // 55 fps is "holding", not "headroom": it must neither drop nor climb.
    feed(55, framesFor(UP_MS, 55) * 2);
    expect(governor.level).toBe(1);

    // 62.5 fps is an exact 16 ms frame, so the window sums without drift.
    feed(62.5, framesFor(UP_MS, 62.5) - 1);
    expect(governor.level).toBe(1);
    feed(62.5, 1);
    expect(governor.level).toBe(0);
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(1);
    expect(knobs.setPressureIters).toHaveBeenLastCalledWith(BASE_PRESSURE);
    expect(knobs.setParticleCount).toHaveBeenLastCalledWith(BASE_PARTICLES);
  });

  it('ignores garbage frame rates', () => {
    const { governor } = settled();
    for (const fps of [NaN, Infinity, -Infinity, 0, -5]) {
      for (let i = 0; i < 100; i++) governor.update(fps, 16);
    }
    expect(governor.level).toBe(0);
  });

  it('pausing returns to the user settings and freezes adaptation', () => {
    const { knobs, governor, feed } = settled();
    feed(40, framesFor(DOWN_MS, 40));
    expect(governor.level).toBe(1);

    governor.setPaused(true);
    expect(governor.level).toBe(0);
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(1);
    feed(10, 200);
    expect(governor.level).toBe(0);

    // Unpausing starts a fresh settle window with clean counters.
    governor.setPaused(false);
    feed(40, settleAt(40) + framesFor(DOWN_MS, 40) - 1);
    expect(governor.level).toBe(0);
    feed(40, 1);
    expect(governor.level).toBe(1);
  });

  it('a warm-up hold keeps the tier through slow frames, then settles before judging', () => {
    const { knobs, governor, feed } = settled();
    governor.setHold(true);
    // ~10 s of a cold MediaPipe start at 20 fps: would otherwise hit the bottom rung.
    feed(20, framesFor(10_000, 20));
    expect(governor.level).toBe(0);
    expect(knobs.setRenderScale).not.toHaveBeenCalled();

    governor.setHold(false);
    feed(40, settleAt(40) + framesFor(DOWN_MS, 40) - 1);
    expect(governor.level).toBe(0);
    feed(40, 1);
    expect(governor.level).toBe(1);
  });

  it('a warm-up hold is bounded, so a device that never finishes still adapts', () => {
    const { governor, feed } = settled();
    governor.setHold(true);
    feed(20, framesFor(30_000, 20));
    expect(governor.level).toBe(0);
    feed(20, settleAt(20) + framesFor(DOWN_MS, 20));
    expect(governor.level).toBe(2);
  });

  it('rebasing re-applies the current tier on top of the new 100%', () => {
    const { knobs, governor, feed } = settled();
    feed(20, framesFor(DOWN_MS, 20));
    expect(governor.level).toBe(2);

    governor.rebase(40, 500_000);
    expect(governor.level).toBe(2);
    expect(knobs.setPressureIters).toHaveBeenLastCalledWith(Math.round(40 * 0.6));
    expect(knobs.setParticleCount).toHaveBeenLastCalledWith(Math.round(500_000 * 0.7));

    // At tier 0 there is nothing to re-apply: the caller's values stand.
    const fresh = settled();
    fresh.governor.rebase(40, 500_000);
    expect(fresh.knobs.setParticleCount).not.toHaveBeenCalled();
  });
});
