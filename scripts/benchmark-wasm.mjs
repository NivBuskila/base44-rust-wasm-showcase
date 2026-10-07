// Times the per-frame engine step of the baseline WASM build in Node.
// Writes the metrics to the path in argv[2] (default wasm-benchmark.json)
// and exits 1 when the median step exceeds BENCH_LIMIT_MS (default 50).
import { readFileSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { initSync, AetherEngine } from '../web/src/wasm/aether.js';

const LIMIT_MS = Number(process.env.BENCH_LIMIT_MS ?? 50);
const WARMUP = 60;
const FRAMES = 600;
const DT = 1 / 60;
const out = process.argv[2] ?? 'wasm-benchmark.json';

initSync({ module: readFileSync(new URL('../web/src/wasm/aether_bg.wasm', import.meta.url)) });
const engine = new AetherEngine(1);
for (let i = 0; i < WARMUP; i++) engine.step(DT);

const times = [];
for (let i = 0; i < FRAMES; i++) {
  const t0 = performance.now();
  engine.step(DT);
  times.push(performance.now() - t0);
}
times.sort((a, b) => a - b);
const pct = (p) => times[Math.min(times.length - 1, Math.floor(p * times.length))];
const round = (x) => Math.round(x * 1000) / 1000;

const metrics = {
  frames: FRAMES,
  limitMs: LIMIT_MS,
  medianMs: round(pct(0.5)),
  p95Ms: round(pct(0.95)),
  maxMs: round(times[times.length - 1]),
  meanMs: round(times.reduce((a, b) => a + b, 0) / FRAMES),
};
metrics.passed = metrics.medianMs <= LIMIT_MS;

writeFileSync(out, JSON.stringify(metrics, null, 2));
console.log(JSON.stringify(metrics));
process.exit(metrics.passed ? 0 : 1);
