/**
 * Loading the two MediaPipe graphs, split out of `perception.ts`.
 *
 * Everything delegate-shaped lives here: the GPU delegate fails on machines with
 * no usable WebGL2, and the failure surfaces as a rejected task construction
 * rather than a flag to query — so trying it and falling back *is* the
 * capability check. Both halves are built together because one can succeed while
 * the other fails, and that half owns a WASM graph plus a GL context which would
 * poison the retry if it were leaked.
 *
 * The glue is imported dynamically: the app only loads graphs when a camera
 * exists, so an ambient-mode session never pays for the chunk.
 */

import type { GestureRecognizer, PoseLandmarker } from "@mediapipe/tasks-vision";

import { HANDS } from "../constants";
import {
  HAND_MODEL,
  POSE_MODEL,
  describe,
  muteInfoLogs,
  resolveModel,
  resolveWasmPath,
} from "./assets";

export interface LoadedGraphs {
  recognizer: GestureRecognizer;
  landmarker: PoseLandmarker;
  delegate: "GPU" | "CPU";
  handModel: string;
  poseModel: string;
}

/** Thrown when neither delegate produced a usable pair of tasks. */
export class GraphLoadError extends Error {}

/**
 * Resolves the runtime and both models, then builds the tasks on the first
 * delegate that accepts them. Rejects with a `GraphLoadError` whose message is
 * the reason the status line shows.
 */
export async function loadGraphs(): Promise<LoadedGraphs> {
  const unmute = muteInfoLogs();
  try {
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(
      await resolveWasmPath(),
    );
    const [handModel, poseModel] = await Promise.all([
      resolveModel(HAND_MODEL),
      resolveModel(POSE_MODEL),
    ]);

    const failures: string[] = [];
    for (const delegate of ["GPU", "CPU"] as const) {
      const built = await Promise.allSettled([
        vision.GestureRecognizer.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: handModel, delegate },
          runningMode: "VIDEO",
          numHands: HANDS,
        }),
        vision.PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: poseModel, delegate },
          runningMode: "VIDEO",
          numPoses: 1,
          outputSegmentationMasks: true,
        }),
      ]);
      const [recognizer, landmarker] = built;
      if (
        recognizer.status === "fulfilled" &&
        landmarker.status === "fulfilled"
      ) {
        return {
          recognizer: recognizer.value,
          landmarker: landmarker.value,
          delegate,
          handModel,
          poseModel,
        };
      }
      const why: string[] = [];
      for (const task of built) {
        if (task.status === "fulfilled") task.value.close();
        else why.push(describe(task.reason));
      }
      failures.push(`${delegate}: ${why.join(" / ")}`);
      console.warn(
        `[aether] ${delegate} delegate unavailable: ${why.join(" / ")}`,
      );
    }
    throw new GraphLoadError(
      `no usable delegate (${failures.join("; ")})`,
    );
  } catch (err) {
    throw err instanceof GraphLoadError
      ? err
      : new GraphLoadError(describe(err));
  } finally {
    unmute();
  }
}

/** Closes both tasks, never throwing: teardown runs on the unload path. */
export function closeGraphs(
  ...tasks: ({ close(): void } | null)[]
): void {
  for (const task of tasks) {
    try {
      task?.close();
    } catch (err) {
      console.warn("[aether] closing a vision task failed", err);
    }
  }
}
