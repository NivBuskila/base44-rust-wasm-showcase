/**
 * Bounded log of WebGPU errors this document hit.
 *
 * A WebGPU error is not an exception: an invalid command makes the driver drop
 * the *whole* submitted command buffer, so the visible symptom is a frame that
 * simply never appears — and an intermittent one reads as flicker, with nothing
 * thrown anywhere. The only report is `device.onuncapturederror`, which lands in
 * the console of the tab that hit it.
 *
 * That console is unreachable from tooling, so the messages are kept here and
 * folded into the QA session record, which does leave the tab (see
 * `qa-recorder.ts`). Bounded and de-duplicated: a per-frame error would
 * otherwise produce thousands of identical lines.
 */

/** Distinct messages kept; past this the log stops growing. */
const MAX_MESSAGES = 8;

/** Characters kept per message — driver messages carry long backtraces. */
const MAX_LENGTH = 400;

const seen = new Map<string, number>();

/** Records one error, counting repeats of a message already seen. */
export function recordGpuError(message: string): void {
  const key = message.slice(0, MAX_LENGTH);
  const count = seen.get(key);
  if (count !== undefined) {
    seen.set(key, count + 1);
    return;
  }
  if (seen.size >= MAX_MESSAGES) return;
  seen.set(key, 1);
}

/** The log, newest last, each line prefixed with how often it occurred. */
export function gpuErrorLog(): string[] {
  return Array.from(seen, ([message, count]) => `x${count} ${message}`);
}
