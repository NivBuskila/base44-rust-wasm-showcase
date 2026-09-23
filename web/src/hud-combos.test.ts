// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { ComboBook } from './hud-combos';

describe('spellbook without projectiles', () => {
  it('shows sequences and duets without advertising throw', () => {
    const parent = document.createElement('div');
    const book = new ComboBook(parent);
    book.setBook('nova:attract,repel');
    book.setDuetBook('tide:attract+repel');
    expect(parent.querySelector('[data-combo="nova"]')).not.toBeNull();
    expect(parent.querySelector('[data-combo="tide"]')).not.toBeNull();
    expect(parent.querySelector('[data-combo="throw"]')).toBeNull();
    expect(parent.textContent).not.toContain('shots');
  });
});
