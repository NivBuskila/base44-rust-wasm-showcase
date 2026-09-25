use super::support::*;

/// A disc in the middle, a bar pinned to the left border and scattered cells,
/// with soft values either side of the 0.5 threshold.
fn body_mask(w: usize, h: usize, seed: u64) -> Grid {
    let mut m = Grid::new(w, h);
    let mut rng = Rng::new(seed);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - w as f32 * 0.6, y as f32 - h as f32 * 0.5);
            let disc = (dx * dx + dy * dy).sqrt() < h as f32 * 0.25;
            let bar = x < 3 && y > h / 3;
            let speck = rng.next_f32() < 0.004;
            m.data[y * w + x] = if disc || bar || speck {
                rng.range(0.5, 1.0)
            } else {
                rng.range(0.0, 0.49)
            };
        }
    }
    m
}

fn brute_reach(mask: &Grid, r: usize) -> Vec<bool> {
    let (w, h) = (mask.w, mask.h);
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), x.saturating_add(r).min(w - 1));
            let (y0, y1) = (y.saturating_sub(r), y.saturating_add(r).min(h - 1));
            out[y * w + x] = (y0..=y1).any(|yy| (x0..=x1).any(|xx| mask.data[yy * w + xx] >= 0.5));
        }
    }
    out
}

#[test]
fn reach_matches_a_brute_force_dilation() {
    let (w, h) = (61, 37);
    let mask = body_mask(w, h, 7);
    let mut reach = Reach::new(w, h);
    for r in [0, 1, 2, 5, 17, 40, 100, usize::MAX] {
        reach.build(&mask.data, w, h, r);
        let want = brute_reach(&mask, r);
        for (i, &near) in want.iter().enumerate() {
            assert_eq!(reach.near(i), near, "radius {r}, cell {i}");
        }
    }
}

/// A stirred field around `body_mask`, with enough speed that traces span
/// several cells and many end inside or beside the body.
fn stirred_around_body(dt: f32) -> Fluid {
    let (w, h) = (96, 54);
    let mut f = Fluid::new(w, h);
    f.set_obstacle(&body_mask(w, h, 3));
    let mut rng = Rng::new(11);
    for _ in 0..12 {
        let (x, y) = (rng.range(4.0, 91.0), rng.range(4.0, 49.0));
        f.add_force(x, y, rng.signed() * 4000.0, rng.signed() * 4000.0, 6.0);
        f.add_dye(x, y, [1.0, 0.6, 0.2], 6.0);
        f.step(dt, &Params::default());
    }
    f
}

/// The trace maps as the old path built them: every endpoint tested against
/// the body, then resolved into a stencil.
fn reference_traces(f: &Fluid, dt: f32) -> (Vec<Stencil>, Vec<Stencil>) {
    let (w, h) = (f.w, f.h);
    let (mut back, mut fwd) = (Vec::new(), Vec::new());
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (fx, fy) = (x as f32, y as f32);
            let (u0, v0) = (f.vel.u.data[i], f.vel.v.data[i]);
            let (bx, by) = f.integrate(fx, fy, -dt, u0, v0);
            let (bx, by) = f.trace_to_fluid(fx, fy, bx, by);
            back.push(Stencil::at(bx, by, w, h));
            let (gx, gy) = f.integrate(fx, fy, dt, u0, v0);
            let (gx, gy) = f.trace_to_fluid(fx, fy, gx, gy);
            fwd.push(Stencil::at(gx, gy, w, h));
        }
    }
    (back, fwd)
}

#[test]
fn skipping_cells_out_of_reach_changes_no_trace() {
    for dt in [1.0 / 144.0, 1.0 / 60.0, 1.0 / 20.0] {
        let mut f = stirred_around_body(dt);
        assert!(f.has_obstacle);
        let (want_back, want_fwd) = reference_traces(&f, dt);
        f.build_traces(dt);
        let (back, fwd) = f.trace_stencils();
        assert_eq!(back, &want_back[..], "backward traces at dt {dt}");
        assert_eq!(fwd, &want_fwd[..], "forward traces at dt {dt}");
    }
}

#[test]
fn a_huge_finite_velocity_reaches_the_whole_grid() {
    let dt = 1.0 / 60.0;
    let mut f = stirred_around_body(dt);
    f.vel.v.data[f.w * 30 + 10] = 1e30;
    assert!(f.trace_reach(dt).expect("finite") >= f.w.max(f.h));
    let (want_back, want_fwd) = reference_traces(&f, dt);
    f.build_traces(dt);
    let (back, fwd) = f.trace_stencils();
    assert_eq!(back, &want_back[..]);
    assert_eq!(fwd, &want_fwd[..]);
}

#[test]
fn a_non_finite_velocity_falls_back_to_testing_every_cell() {
    let dt = 1.0 / 60.0;
    let mut f = stirred_around_body(dt);
    f.vel.u.data[f.w * 20 + 30] = f32::INFINITY;
    assert_eq!(f.trace_reach(dt), None);
    let (want_back, want_fwd) = reference_traces(&f, dt);
    f.build_traces(dt);
    let (back, fwd) = f.trace_stencils();
    assert_eq!(back, &want_back[..]);
    assert_eq!(fwd, &want_fwd[..]);
}

#[test]
fn the_body_fade_matches_the_per_cell_band_scan() {
    let dt = 1.0 / 60.0;
    let mut f = stirred_around_body(dt);
    let (w, h) = (f.w, f.h);
    let band = BODY_CONTACT_BAND;
    let fade = crate::math::decay(BODY_CONTACT_FADE, dt);

    // The band scan as it was: any wall within `band` along the row, then any
    // such row hit within `band` down the column.
    let mut want = f.dye.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if f.obstacle.data[i] >= 0.5 {
                for c in &mut want {
                    c.data[i] = 0.0;
                }
                continue;
            }
            let near = (y.saturating_sub(band)..=(y + band).min(h - 1)).any(|yy| {
                (x.saturating_sub(band)..=(x + band).min(w - 1))
                    .any(|xx| f.obstacle.data[yy * w + xx] >= 0.5)
            });
            if near {
                for c in &mut want {
                    c.data[i] *= fade;
                }
            }
        }
    }

    f.fade_dye_at_body(dt);
    for (k, (got, want)) in f.dye.iter().zip(&want).enumerate() {
        assert!(
            got.data
                .iter()
                .zip(&want.data)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "dye channel {k} differs"
        );
    }
}

/// The farthest, in cells on either axis, that any stencil corner of any trace
/// this step lands from the cell it started at.
fn widest_trace(f: &Fluid, dt: f32) -> usize {
    let (w, h) = (f.w, f.h);
    let mut widest = 0;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (u0, v0) = (f.vel.u.data[i], f.vel.v.data[i]);
            for sign in [-1.0, 1.0] {
                let (tx, ty) = f.integrate(x as f32, y as f32, sign * dt, u0, v0);
                let (ix, _) = corner_of(tx, w);
                let (iy, _) = corner_of(ty, h);
                let reach = [ix.abs_diff(x), (ix + 1).abs_diff(x)]
                    .into_iter()
                    .chain([iy.abs_diff(y), (iy + 1).abs_diff(y)])
                    .max()
                    .unwrap_or(0);
                widest = widest.max(reach);
            }
        }
    }
    widest
}

#[test]
fn every_trace_stencil_lies_inside_the_reach_radius() {
    // The last cell of the radius is rounding margin; the bound itself must
    // hold without it.
    for dt in [1.0 / 144.0, 1.0 / 60.0, 1.0 / 20.0] {
        let f = stirred_around_body(dt);
        let bound = f.trace_reach(dt).expect("finite field") - 1;
        let widest = widest_trace(&f, dt);
        assert!(
            widest <= bound,
            "stirred: reached {widest} cells, bound {bound}"
        );
    }

    // A uniform flow traces exactly `dt * speed`; a whole number of cells is
    // the case where the `+1` stencil corner lands right on the bound.
    for (dt, speed) in [(1.0 / 20.0, 60.0), (1.0 / 60.0, 120.0), (1.0 / 30.0, 97.0)] {
        let mut f = Fluid::new(64, 40);
        f.vel.u.data.fill(speed);
        f.vel.v.data.fill(-speed);
        let bound = f.trace_reach(dt).expect("finite field") - 1;
        let widest = widest_trace(&f, dt);
        assert!(
            widest <= bound,
            "uniform {speed}: reached {widest}, bound {bound}"
        );
        assert!(
            widest + 1 >= bound,
            "uniform {speed}: bound {bound} is loose ({widest})"
        );
    }
}
