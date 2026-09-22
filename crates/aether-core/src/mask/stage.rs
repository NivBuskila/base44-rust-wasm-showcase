//! Ingest: copying an incoming mask into staging (mirroring, clamping) and
//! the pipeline that turns it into the binary obstacle field.

use super::*;

impl BodyMask {
    /// Lazily allocated staging grid matching the incoming mask resolution.
    pub(super) fn staging(&mut self, sw: usize, sh: usize) -> &mut Grid {
        let (sw, sh) = (sw.max(2), sh.max(2));
        if self
            .incoming
            .as_ref()
            .is_some_and(|g| g.w != sw || g.h != sh)
        {
            self.incoming = None;
        }
        self.incoming.get_or_insert_with(|| Grid::new(sw, sh))
    }

    /// Copies an incoming mask into the staging grid, mirroring and clamping.
    /// Returns false if the dimensions do not describe the slice, in which case
    /// nothing has been touched.
    ///
    /// Generic over the sample type so the 8-bit path does not have to allocate
    /// a float copy of the mask every frame.
    fn stage<T: Copy>(
        &mut self,
        src: &[T],
        sw: usize,
        sh: usize,
        mirror: bool,
        load: fn(T) -> f32,
    ) -> bool {
        let Some(n) = sw.checked_mul(sh) else {
            return false;
        };
        if sw < 2 || sh < 2 || src.len() < n {
            return false;
        }
        let staging = self.staging(sw, sh);
        for y in 0..sh {
            let row = y * sw;
            for x in 0..sw {
                let sx = if mirror { sw - 1 - x } else { x };
                // `max` then `min` maps NaN to 0 branchlessly, the trick
                // `Grid::sample` documents — and the reason this is not
                // `clamp`, which propagates NaN. A NaN confidence otherwise
                // survives the resample and poisons the obstacle field, and
                // from there the velocity field, permanently.
                #[allow(clippy::manual_clamp)]
                let v = load(src[row + sx]).max(0.0).min(1.0);
                staging.data[row + x] = v;
            }
        }
        true
    }

    /// Ingests a float confidence mask at an arbitrary resolution.
    ///
    /// `mirror` flips horizontally, matching the mirrored preview the user sees.
    /// The mask covers the whole camera frame, so it is stretched across the
    /// whole grid; the aspect ratios differ and correcting for that would crop
    /// the body instead.
    pub fn update_f32(&mut self, src: &[f32], sw: usize, sh: usize, mirror: bool, dt: f32) {
        if !self.stage(src, sw, sh, mirror, |v| v) {
            return;
        }
        self.process(sane_dt(dt));
    }

    /// Ingests an 8-bit mask at an arbitrary resolution.
    pub fn update_u8(&mut self, src: &[u8], sw: usize, sh: usize, mirror: bool, dt: f32) {
        if !self.stage(src, sw, sh, mirror, |v| v as f32 * (1.0 / 255.0)) {
            return;
        }
        self.process(sane_dt(dt));
    }

    /// The pipeline proper, from the filled staging grid onwards.
    pub(super) fn process(&mut self, dt: f32) {
        // Moved out and back so `raw` can be borrowed mutably while the staging
        // grid is read.
        if let Some(staged) = self.incoming.take() {
            self.raw.resample_from(&staged);
            self.incoming = Some(staged);
        }

        // A NaN threshold makes every comparison false, which silently disables
        // body collision; a garbage HUD value should degrade, not disappear.
        let level = if self.cfg.threshold.is_finite() {
            self.cfg.threshold.clamp(0.0, 1.0)
        } else {
            FALLBACK_THRESHOLD
        };

        let cells = self.raw.len().max(1);
        let raw_solid = self.raw.data.iter().filter(|&&c| c >= level).count() as f32 / cells as f32;
        if raw_solid > MAX_COVERAGE {
            // Checked before anything is committed, so a failed segmentation
            // cannot even pollute the temporal history it would take several
            // frames to walk back out of.
            return;
        }

        // A body that has just appeared has to engage on the first frame. The
        // EMA exists to stop an *established* silhouette from boiling; ramping
        // in from zero instead means the mask cannot cross the threshold at all
        // until a half-life has passed, which reads as input lag. Snapping
        // `prev` as well keeps the arrival silent: a 0 -> 1 step is a full
        // amplitude delta, and pushing on it would slam the fluid every time
        // tracking re-acquires.
        let cold = !self.present && raw_solid >= MIN_COVERAGE;
        if cold {
            self.smooth.data.copy_from_slice(&self.raw.data);
            self.prev.data.copy_from_slice(&self.raw.data);
        } else {
            let keep = ema_keep(self.cfg.half_life, dt);
            for (s, &r) in self.smooth.data.iter_mut().zip(&self.raw.data) {
                *s = lerp(r, *s, keep);
            }
        }

        for (o, &c) in self.obstacle.data.iter_mut().zip(&self.smooth.data) {
            *o = if c >= level { 1.0 } else { 0.0 };
        }
        // Blur-then-rethreshold is a cheap morphological close. A bare
        // threshold leaves single-cell speckle all over the silhouette, and
        // speckle in the obstacle field makes the pressure solve ring.
        let passes = self.cfg.blur_passes.min(MAX_BLUR_PASSES);
        if passes > 0 {
            self.obstacle.blur(&mut self.scratch, passes);
            for o in &mut self.obstacle.data {
                *o = if *o >= CLOSE_LEVEL { 1.0 } else { 0.0 };
            }
        }

        let solid = self.obstacle.data.iter().filter(|&&o| o >= 0.5).count();
        self.coverage = solid as f32 / cells as f32;
        self.present = self.coverage >= MIN_COVERAGE;

        self.rebuild_edge(dt);
        self.prev.data.copy_from_slice(&self.smooth.data);

        if !self.present {
            // Not just "no body": the field is cleared outright. See the
            // invariant in the module docs — solid cells surviving with
            // `present == false` get frozen into the solver.
            self.obstacle.zero();
            self.edge.zero();
            self.coverage = 0.0;
        }
        self.since_update = 0.0;
        self.fade = if self.present { 1.0 } else { 0.0 };
    }
}
