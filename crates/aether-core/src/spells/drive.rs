//! The model-free drive: raw optical flow and the body silhouette's boundary
//! velocity pushed into the fluid every step.

use super::*;

/// The two whole-field drives — optical flow and the body silhouette's boundary
/// motion — fused into one pass.
///
/// Fused rather than sequential because each pass is a read-modify-write over
/// the whole velocity field; doing them separately doubles the traffic on the
/// largest array in the engine to no benefit.
pub(super) fn drive_fields(
    flow: &VecField,
    body_edge: &VecField,
    fluid: &mut Fluid,
    params: &Params,
    dt: f32,
    report: &mut SpellReport,
) {
    // Both are injected every frame, so they are *rates*: the field settles at
    // roughly rate / dissipation. The `dt * 60` factor expresses them as "per
    // frame at 60 fps" while staying frame-rate independent.
    let flow_gain = if params.flow_force > 0.0 {
        params.flow_force * dt * 60.0
    } else {
        0.0
    };
    // The engine hands us a zero field whenever no body is tracked, which is
    // the common case, so a linear scan is worth it to skip a full
    // sample-and-write pass over the velocity field.
    let body_gain = if params.body_push > 0.0 && !is_zero(body_edge) {
        params.body_push * dt * 60.0
    } else {
        0.0
    };
    if flow_gain == 0.0 && body_gain == 0.0 {
        return;
    }

    let (gw, gh) = (fluid.width(), fluid.height());
    let (fsx, fsy) = resample_scale(flow, gw, gh);
    let (bsx, bsy) = resample_scale(body_edge, gw, gh);
    let mut injected = 0.0;
    let vel = fluid.velocity_mut();
    for y in 0..gh {
        let fy = y as f32 * fsy;
        let by = y as f32 * bsy;
        for x in 0..gw {
            let mut du = 0.0;
            let mut dv = 0.0;
            if flow_gain != 0.0 {
                let (u, v) = flow.sample(x as f32 * fsx, fy);
                du += u * flow_gain;
                dv += v * flow_gain;
            }
            if body_gain != 0.0 {
                let (u, v) = body_edge.sample(x as f32 * bsx, by);
                du += u * body_gain;
                dv += v * body_gain;
            }
            let du = clamp_impulse(du);
            let dv = clamp_impulse(dv);
            vel.add(x, y, du, dv);
            injected += du * du + dv * dv;
        }
    }
    report.injected += injected;
}
