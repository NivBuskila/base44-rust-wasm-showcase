import { describe, expect, it } from 'vitest';
import { isAppleMobile, isInAppBrowser } from './device-hints';

const IPHONE_SAFARI =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1';
const IPHONE_LINKEDIN =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 [LinkedInApp]/9.31.1';
const IPADOS =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15';
const ANDROID_CHROME =
  'Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36';
const ANDROID_FACEBOOK = `${ANDROID_CHROME} [FBAN/EMA;FBLC/en_US;FBAV/500.0.0.0.0;]`;
const DESKTOP_CHROME =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36';

describe('isAppleMobile', () => {
  it('matches an iPhone, including an in-app webview on one', () => {
    expect(isAppleMobile(IPHONE_SAFARI, 5)).toBe(true);
    expect(isAppleMobile(IPHONE_LINKEDIN, 5)).toBe(true);
  });

  it('tells an iPad (desktop user agent, touch points) from a Mac', () => {
    expect(isAppleMobile(IPADOS, 5)).toBe(true);
    expect(isAppleMobile(IPADOS, 0)).toBe(false);
  });

  it('leaves Android and desktop alone', () => {
    expect(isAppleMobile(ANDROID_CHROME, 5)).toBe(false);
    expect(isAppleMobile(DESKTOP_CHROME, 0)).toBe(false);
  });
});

describe('isInAppBrowser', () => {
  it('matches the social apps’ webviews', () => {
    expect(isInAppBrowser(IPHONE_LINKEDIN)).toBe(true);
    expect(isInAppBrowser(ANDROID_FACEBOOK)).toBe(true);
  });

  it('leaves real browsers alone', () => {
    expect(isInAppBrowser(IPHONE_SAFARI)).toBe(false);
    expect(isInAppBrowser(ANDROID_CHROME)).toBe(false);
    expect(isInAppBrowser(DESKTOP_CHROME)).toBe(false);
  });
});
