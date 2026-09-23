import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { watchHand } from './landing-hand';

describe('watchHand', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('opens once after a hand is held up long enough', () => {
    const onFull = vi.fn();
    watchHand(() => 1, () => {}, onFull);
    vi.advanceTimersByTime(1100);
    expect(onFull).not.toHaveBeenCalled();
    vi.advanceTimersByTime(200);
    expect(onFull).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(2000);
    expect(onFull).toHaveBeenCalledTimes(1);
  });

  it('drains when the hand leaves, so a passing hand never opens it', () => {
    let hands = 1;
    const onFull = vi.fn();
    let last = 0;
    watchHand(() => hands, (c) => (last = c), onFull);
    vi.advanceTimersByTime(600);
    hands = 0;
    vi.advanceTimersByTime(400);
    expect(last).toBe(0);
    hands = 1;
    vi.advanceTimersByTime(1100);
    expect(onFull).not.toHaveBeenCalled();
  });

  it('stops polling on stop()', () => {
    const hands = vi.fn(() => 1);
    const watch = watchHand(hands, () => {}, () => {});
    vi.advanceTimersByTime(100);
    watch.stop();
    const calls = hands.mock.calls.length;
    vi.advanceTimersByTime(1000);
    expect(hands.mock.calls.length).toBe(calls);
  });
});
