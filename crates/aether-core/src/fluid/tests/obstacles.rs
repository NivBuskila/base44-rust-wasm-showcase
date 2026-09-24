//! Solid cells: exactly at rest after a step, and no leak through a wall.

use super::support::*;

#[test]
fn obstacle_cells_are_exactly_at_rest_after_a_step() {
    let mut f = Fluid::new(64, 48);
    f.set_obstacle(&bar_mask(64, 48, 28, 34));
    let params = Params::default();
    for _ in 0..20 {
        // Aim straight at the bar.
        f.add_force(20.0, 24.0, 300.0, 0.0, 9.0);
        f.step(DT, &params);
    }
    for y in 0..48 {
        for x in 0..64 {
            let i = y * 64 + x;
            if f.solid.data[i] >= 0.5 {
                assert_eq!(f.vel.u.data[i], 0.0, "u nonzero in wall at {x},{y}");
                assert_eq!(f.vel.v.data[i], 0.0, "v nonzero in wall at {x},{y}");
            }
        }
    }
}

#[test]
fn replacing_an_obstacle_updates_the_cached_walls_before_the_next_step() {
    let (w, h) = (32, 24);
    let mut f = Fluid::new(w, h);
    let left = 12 * w + 8;
    let right = 12 * w + 22;
    let mut mask = Grid::new(w, h);
    mask.data[left] = 1.0;
    f.set_obstacle(&mask);
    assert!(f.has_obstacle);
    assert_eq!(f.solid.data[left], 1.0);

    mask.data[left] = 0.0;
    mask.data[right] = 1.0;
    f.set_obstacle(&mask);
    assert_eq!(f.solid.data[left], 0.0);
    assert_eq!(f.solid.data[right], 1.0);
    f.vel.u.data[right] = 10.0;
    f.step(DT, &inert_params());
    assert_eq!(f.vel.u.data[right], 0.0);

    f.set_obstacle(&Grid::new(w, h));
    assert!(!f.has_obstacle);
    assert_eq!(f.solid.data[right], 0.0);
    assert_eq!(f.solid.data[0], 1.0); // The domain border stays solid.
}

#[test]
fn fluid_does_not_leak_through_a_solid_wall() {
    let (w, h) = (64, 48);
    let mut f = Fluid::new(w, h);
    f.set_obstacle(&bar_mask(w, h, 30, 34));
    let params = Params::default();
    for _ in 0..300 {
        f.add_force(12.0, 24.0, 320.0, 0.0, 8.0);
        f.add_dye(12.0, 24.0, [0.4, 0.4, 0.4], 6.0);
        f.step(DT, &params);
    }
    assert!(f.max_speed() > 10.0, "the push never took");

    let mut far_speed = 0.0f32;
    let mut far_dye = 0.0f32;
    for y in 0..h {
        for x in 38..w {
            let i = y * w + x;
            far_speed = far_speed.max(
                (f.vel.u.data[i] * f.vel.u.data[i] + f.vel.v.data[i] * f.vel.v.data[i]).sqrt(),
            );
            far_dye += f.dye[0].data[i];
        }
    }
    assert!(
        far_speed < 1e-4,
        "velocity leaked past the wall: {far_speed} (near side {})",
        f.max_speed()
    );
    assert!(far_dye < 1e-6, "dye leaked past the wall: {far_dye}");
}
