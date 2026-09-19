import { describe, expect, it, vi } from 'vitest';

import { PerformanceGovernor, type QualityKnobs } from './performance-governor';

// Mirrors of the governor's private ladder constants. Pinned here on purpose:
// a change to how fast the ladder moves should fail a test, not slip through.
const SETTLE_FRAMES = 45;
const DOWN_FRAMES = 30;
const UP_FRAMES = 240;

const BASE_PRESSURE = 28;
const BASE_PARTICLES = 120_000;

function rig() {
  const knobs: QualityKnobs = {
    setPressureIters: vi.fn(),
    setParticleCount: vi.fn(),
    setRenderScale: vi.fn(),
  };
  const governor = new PerformanceGovernor(knobs, BASE_PRESSURE, BASE_PARTICLES);
  const feed = (fps: number, frames: number) => {
    for (let i = 0; i < frames; i++) governor.update(fps);
  };
  return { knobs, governor, feed };
}

/** Skips the boot settle window, during which fps is deliberately ignored. */
function settled() {
  const r = rig();
  r.feed(60, SETTLE_FRAMES);
  return r;
}

describe('PerformanceGovernor', () => {
  it('starts at tier 0 and touches nothing while it settles', () => {
    const { knobs, governor, feed } = rig();
    feed(20, SETTLE_FRAMES);
    expect(governor.level).toBe(0);
    expect(knobs.setRenderScale).not.toHaveBeenCalled();
    expect(knobs.setPressureIters).not.toHaveBeenCalled();
    expect(knobs.setParticleCount).not.toHaveBeenCalled();
  });

  it('drops one tier after sustained missed frames, not on a single hitch', () => {
    const { knobs, governor, feed } = settled();
    feed(40, DOWN_FRAMES - 1);
    expect(governor.level).toBe(0);
    feed(40, 1);
    expect(governor.level).toBe(1);
    // Tier 1: resolution first and hardest, pressure follows, pool untouched.
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(0.85);
    expect(knobs.setPressureIters).toHaveBeenLastCalledWith(Math.round(BASE_PRESSURE * 0.75));
    expect(knobs.setParticleCount).toHaveBeenLastCalledWith(BASE_PARTICLES);
  });

  it('a good frame resets the miss counter', () => {
    const { governor, feed } = settled();
    feed(40, DOWN_FRAMES - 1);
    feed(55, 1);
    feed(40, DOWN_FRAMES - 1);
    expect(governor.level).toBe(0);
  });

  it('settles after every change before judging again', () => {
    const { governor, feed } = settled();
    feed(20, DOWN_FRAMES);
    expect(governor.level).toBe(1);
    // Still terrible, but the fps average has not caught up with the new tier.
    feed(20, SETTLE_FRAMES + DOWN_FRAMES - 1);
    expect(governor.level).toBe(1);
    feed(20, 1);
    expect(governor.level).toBe(2);
  });

  it('never drops below the last rung and keeps the absolute floors', () => {
    const knobs: QualityKnobs = {
      setPressureIters: vi.fn(),
      setParticleCount: vi.fn(),
      setRenderScale: vi.fn(),
    };
    // Bases so small that every fraction would go under the floors.
    const governor = new PerformanceGovernor(knobs, 12, 3000);
    for (let i = 0; i < 10 * (SETTLE_FRAMES + DOWN_FRAMES); i++) governor.update(10);
    expect(governor.level).toBe(3);
    expect(knobs.setPressureIters).toHaveBeenLastCalledWith(10);
    expect(knobs.setParticleCount).toHaveBeenLastCalledWith(2000);
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(0.6);
  });

  it('climbs back slowly and only from sustained headroom', () => {
    const { knobs, governor, feed } = settled();
    feed(20, DOWN_FRAMES);
    feed(20, SETTLE_FRAMES);
    expect(governor.level).toBe(1);

    // 55 fps is "holding", not "headroom": it must neither drop nor climb.
    feed(55, UP_FRAMES * 2);
    expect(governor.level).toBe(1);

    feed(60, UP_FRAMES - 1);
    expect(governor.level).toBe(1);
    feed(60, 1);
    expect(governor.level).toBe(0);
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(1);
    expect(knobs.setPressureIters).toHaveBeenLastCalledWith(BASE_PRESSURE);
    expect(knobs.setParticleCount).toHaveBeenLastCalledWith(BASE_PARTICLES);
  });

  it('ignores garbage frame rates', () => {
    const { governor, feed } = settled();
    for (const fps of [NaN, Infinity, -Infinity, 0, -5]) feed(fps, DOWN_FRAMES * 2);
    expect(governor.level).toBe(0);
  });

  it('pausing returns to the user settings and freezes adaptation', () => {
    const { knobs, governor, feed } = settled();
    feed(20, DOWN_FRAMES);
    expect(governor.level).toBe(1);

    governor.setPaused(true);
    expect(governor.level).toBe(0);
    expect(knobs.setRenderScale).toHaveBeenLastCalledWith(1);
    feed(10, 10 * DOWN_FRAMES);
    expect(governor.level).toBe(0);

    // Unpausing starts a fresh settle window with clean counters.
    governor.setPaused(false);
    feed(10, SETTLE_FRAMES + DOWN_FRAMES - 1);
    expect(governor.level).toBe(0);
    feed(10, 1);
    expect(governor.level).toBe(1);
  });

  it('rebasing re-applies the current tier on top of the new 100%', () => {
    const { knobs, governor, feed } = settled();
    feed(20, DOWN_FRAMES + SETTLE_FRAMES + DOWN_FRAMES);
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
