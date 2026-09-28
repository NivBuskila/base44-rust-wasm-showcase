/**
 * What kind of device and browser the page landed in.
 *
 * Only the cheap signals a page can read without asking the user anything. The
 * matchers take their inputs as arguments so the whole matrix is unit-tested;
 * the zero-argument wrappers read the live browser and are safe to call in a
 * test runner, where `navigator` and `matchMedia` may be missing.
 */

/** In-app webviews: the camera there is often refused or never prompted for. */
const IN_APP = /LinkedInApp|FBAN|FBAV|Instagram|Twitter/i;

/**
 * iPhone, iPod or iPad. iPadOS reports a desktop "Macintosh" user agent, so a
 * Mac that has a touchscreen is an iPad; real Macs report no touch points.
 */
export function isAppleMobile(userAgent: string, maxTouchPoints: number): boolean {
  if (/iPhone|iPad|iPod/.test(userAgent)) return true;
  return /Macintosh/.test(userAgent) && maxTouchPoints > 1;
}

/** LinkedIn, X, Facebook or Instagram's built-in browser. */
export function isInAppBrowser(userAgent: string): boolean {
  return IN_APP.test(userAgent);
}

/** The primary pointer is a finger: a phone or a tablet. */
export function coarsePointer(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
}

/** {@link isInAppBrowser} for this browser. */
export function inAppBrowser(): boolean {
  return typeof navigator !== 'undefined' && isInAppBrowser(navigator.userAgent ?? '');
}

/**
 * A phone or tablet: a coarse pointer, or Apple mobile WebKit whatever pointer
 * it reports (an iPad with a trackpad has a fine one).
 */
export function touchDevice(): boolean {
  if (coarsePointer()) return true;
  if (typeof navigator === 'undefined') return false;
  return isAppleMobile(navigator.userAgent ?? '', navigator.maxTouchPoints ?? 0);
}
