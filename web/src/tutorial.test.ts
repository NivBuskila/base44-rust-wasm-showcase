import { describe, expect, it } from 'vitest';
import { CLEAR_MS, HOLD_MS, STEPS, TutorialProgress } from './tutorial';

describe('TutorialProgress', () => {
  it('covers every single-hand spell and leaves warp to the reference', () => {
    expect(STEPS.map((s) => s.spell)).toEqual([
      'attract',
      'vortex',
      'ignite',
      'shatter',
      'freeze',
      'repel',
      'release',
    ]);
  });

  it('never puts two combo-opening spells back to back', () => {
    const pairs = [
      ['attract', 'repel'],
      ['repel', 'vortex'],
      ['attract', 'freeze'],
    ];
    const order = STEPS.map((s) => s.spell);
    for (let i = 0; i + 1 < order.length; i++) {
      expect(pairs).not.toContainEqual([order[i], order[i + 1]]);
    }
  });

  it('waits for the hand to rest before arming the next step', () => {
    const p = new TutorialProgress();
    p.feed(['attract', 'idle'], 0);
    expect(p.feed(['attract', 'idle'], HOLD_MS)).toBe(1);
    expect(p.resting).toBe(true);
    // Straight into the next pose: ignored while the gate is shut.
    expect(p.feed(['vortex', 'idle'], HOLD_MS + 10)).toBe(0);
    expect(p.feed(['vortex', 'idle'], HOLD_MS + 10 + CLEAR_MS + HOLD_MS)).toBe(0);
    // Resting clears the gate, and only then does the step charge.
    p.feed(['idle', 'idle'], 5000);
    expect(p.feed(['idle', 'idle'], 5000 + CLEAR_MS)).toBe(0);
    expect(p.resting).toBe(false);
    p.feed(['vortex', 'idle'], 6000);
    expect(p.feed(['vortex', 'idle'], 6000 + HOLD_MS)).toBe(1);
  });

  it('charges while the spell is held and passes at HOLD_MS', () => {
    const p = new TutorialProgress();
    expect(p.feed(['attract', 'idle'], 1000)).toBe(0);
    expect(p.feed(['attract', 'idle'], 1000 + HOLD_MS / 2)).toBeCloseTo(0.5);
    expect(p.index).toBe(0);
    expect(p.feed(['attract', 'idle'], 1000 + HOLD_MS)).toBe(1);
    expect(p.index).toBe(1);
    expect(p.step?.spell).toBe('vortex');
  });

  it('resets the charge when the pose breaks', () => {
    const p = new TutorialProgress();
    p.feed(['attract', 'idle'], 0);
    p.feed(['attract', 'idle'], HOLD_MS / 2);
    expect(p.feed(['idle', 'idle'], HOLD_MS / 2 + 16)).toBe(0);
    // Re-acquired: the timer starts over rather than resuming.
    expect(p.feed(['attract', 'idle'], HOLD_MS)).toBe(0);
    expect(p.feed(['attract', 'idle'], HOLD_MS + HOLD_MS / 2)).toBeCloseTo(0.5);
  });

  it('accepts the spell on either hand', () => {
    const p = new TutorialProgress();
    p.feed(['idle', 'attract'], 0);
    expect(p.feed(['idle', 'attract'], HOLD_MS)).toBe(1);
  });

  it('passes release on sight, since it is never latched', () => {
    const p = new TutorialProgress();
    while (p.step?.spell !== 'release') p.skip();
    expect(p.feed(['release', 'idle'], 0)).toBe(1);
    expect(p.done).toBe(true);
  });
});
