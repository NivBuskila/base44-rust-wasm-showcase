/**
 * Picks the WebGPU device the GPU backend runs on, or explains why not.
 *
 * The decision is deliberately conservative. WebGPU on a *software* adapter
 * (SwiftShader, llvmpipe, the browser's fallback adapter) is slower than the
 * WebGL2 path it would replace and — worse — makes the headless test suite
 * measure a rasteriser the user never sees, so those are refused and the app
 * falls back to the Rust/WASM pool plus WebGL2. `?gpu=off` forces that path on
 * any machine, `?gpu=on` accepts whatever adapter the browser offers.
 */

export interface GpuGate {
  ok: boolean;
  reason: string;
}

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** Adapter description for the HUD and the console, best effort. */
  label: string;
}

const SOFTWARE_HINTS = ['swiftshader', 'llvmpipe', 'software', 'lavapipe', 'warp'];

function requested(): 'on' | 'off' | 'auto' {
  if (typeof location === 'undefined') return 'auto';
  const v = new URLSearchParams(location.search).get('gpu');
  return v === 'on' || v === 'off' ? v : 'auto';
}

function describe(adapter: GPUAdapter): string {
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  if (!info) return 'webgpu';
  const parts = [info.description, info.architecture, info.vendor].filter(
    (s): s is string => typeof s === 'string' && s.length > 0,
  );
  return parts.length > 0 ? parts[0] : 'webgpu';
}

function looksSoftware(adapter: GPUAdapter): boolean {
  const a = adapter as GPUAdapter & { info?: GPUAdapterInfo; isFallbackAdapter?: boolean };
  if (a.isFallbackAdapter || a.info?.isFallbackAdapter) return true;
  const text = [a.info?.description, a.info?.architecture, a.info?.vendor, a.info?.device]
    .filter((s): s is string => typeof s === 'string')
    .join(' ')
    .toLowerCase();
  return SOFTWARE_HINTS.some((hint) => text.includes(hint));
}

/**
 * Requests a device. Never throws: a `null` context plus a reason is the
 * normal outcome on a browser without WebGPU.
 */
export async function acquireGpu(): Promise<{ ctx: GpuContext | null; gate: GpuGate }> {
  const want = requested();
  if (want === 'off') return { ctx: null, gate: { ok: false, reason: 'disabled by ?gpu=off' } };

  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return { ctx: null, gate: { ok: false, reason: 'WebGPU unavailable in this browser' } };

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (err) {
    return { ctx: null, gate: { ok: false, reason: `requestAdapter failed: ${String(err)}` } };
  }
  if (!adapter) return { ctx: null, gate: { ok: false, reason: 'no WebGPU adapter' } };

  if (want !== 'on' && looksSoftware(adapter)) {
    return {
      ctx: null,
      gate: { ok: false, reason: `software WebGPU adapter (${describe(adapter)}); WebGL2 is faster` },
    };
  }

  // A 1M-particle pool is 32 MB of storage; ask for it explicitly so a device
  // with tight defaults says no here rather than at the first bind.
  const limits: Record<string, number> = {};
  const wantStorage = 64 * 1024 * 1024;
  if (adapter.limits.maxStorageBufferBindingSize >= wantStorage) {
    limits.maxStorageBufferBindingSize = wantStorage;
  }
  if (adapter.limits.maxBufferSize >= wantStorage) limits.maxBufferSize = wantStorage;

  try {
    const device = await adapter.requestDevice({ requiredLimits: limits });
    return {
      ctx: { adapter, device, label: describe(adapter) },
      gate: { ok: true, reason: describe(adapter) },
    };
  } catch (err) {
    return { ctx: null, gate: { ok: false, reason: `requestDevice failed: ${String(err)}` } };
  }
}
