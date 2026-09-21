/**
 * The WebGL2 scene draw and the debug blit, mirroring `gpu/scene-pass.ts`.
 *
 * Stateless over the GL objects — the textures belong to `SceneSources` and the
 * programs to `ScenePrograms`, both of which die with the context — so this is
 * the per-frame uniform block and the two uploads, nothing more. Keeping it
 * beside the WebGPU pass is the point: the camera-gating rule and the dye/video
 * texel arithmetic are written once per backend and read side by side.
 */

import { FLUID_H, FLUID_W } from "../constants";
import type { RenderFrame } from "../types";
import type { RenderTarget } from "./gl";
import { aspectOf, cameraFit, usesCamera } from "./look";
import type { ScenePrograms } from "./programs";
import type { SceneSources } from "./sources";
import type { ModeStyle } from "./styles";

/**
 * Uploads the dye (and, when the mode shows it, the camera) and draws the
 * treated background into `scene`.
 *
 * The camera mode is a stronger statement than the camera toggle: the user
 * asked to look at the feed, so `showCamera` only gates the aether view. And
 * `usesCamera` is not a style choice, it is the identity: `particles` tints both
 * the body and the rim to black, so running the layer there costs a full-frame
 * `texSubImage2D` plus five fetches per pixel to add exactly zero. On a
 * software rasteriser that was a quarter of the frame.
 */
export function drawScenePass(
  gl: WebGL2RenderingContext,
  programs: ScenePrograms,
  sources: SceneSources,
  scene: RenderTarget,
  frame: RenderFrame,
  style: ModeStyle,
  intensity: number,
  time: number,
  canvasW: number,
  canvasH: number,
  video: HTMLVideoElement | null,
): void {
  const p = programs.scene;
  sources.uploadGrid(sources.dyeTex, frame.dye);

  const wantCamera =
    usesCamera(style) &&
    (frame.mode === "camera" || frame.mode === "blend" || frame.showCamera);
  const hasVideo = wantCamera && sources.uploadVideo(frame.video ?? video);

  scene.bind();
  p.use();
  p.tex("u_dye", 0, sources.dyeTex);
  p.tex("u_video", 1, sources.videoTex);
  p.tex("u_noise", 2, sources.noiseTex);
  p.f2("u_dyeSize", FLUID_W, FLUID_H);
  p.f2("u_dyeTexel", 1 / FLUID_W, 1 / FLUID_H);
  p.f1("u_dyeAmount", style.dye);
  p.f1("u_bgAmount", style.bg);
  p.f1("u_intensity", intensity);
  p.f1("u_time", time);
  p.f1("u_hasVideo", hasVideo ? 1 : 0);
  p.f3("u_camTint", style.camTint[0], style.camTint[1], style.camTint[2]);
  p.f3("u_camEdge", style.camEdge[0], style.camEdge[1], style.camEdge[2]);
  p.f1("u_camToe", style.camToe);
  p.f1("u_camRaw", style.camRaw);
  p.f2(
    "u_videoTexel",
    1 / Math.max(1, sources.videoTexW),
    1 / Math.max(1, sources.videoTexH),
  );

  const [camScaleX, camScaleY] = cameraFit(
    aspectOf(canvasW, canvasH, 1),
    sources.videoAspect,
  );
  p.f2("u_camScale", camScaleX, camScaleY);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

/**
 * Raw obstacle/flow texture straight to the default framebuffer, ungraded.
 *
 * `frame.debug` is null whenever the engine has nothing to show. Say so with a
 * flat field rather than leaving the last frame on screen.
 */
export function drawDebugPass(
  gl: WebGL2RenderingContext,
  programs: ScenePrograms,
  sources: SceneSources,
  frame: RenderFrame,
  canvasW: number,
  canvasH: number,
): void {
  const source =
    frame.debug && sources.uploadGrid(sources.debugTex, frame.debug)
      ? sources.debugTex
      : null;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvasW, canvasH);
  if (!source) {
    gl.clearColor(0.03, 0.035, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return;
  }
  programs.blit.use();
  programs.blit.tex("u_src", 0, source);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
