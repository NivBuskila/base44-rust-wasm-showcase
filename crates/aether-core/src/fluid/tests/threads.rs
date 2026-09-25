use super::support::*;

/// Stirs a field around a bar obstacle on a pool of `threads` workers.
///
/// The grid is 96x54 so that most thread counts split its rows into bands of
/// unequal length, which is where a band's row offset can go wrong.
fn stirred_on(threads: usize) -> Fluid {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(|| {
        let (w, h) = (96, 54);
        let mut fluid = Fluid::new(w, h);
        fluid.set_obstacle(&bar_mask(w, h, 60, 64));
        let params = Params::default();
        for step in 0..60 {
            let angle = step as f32 * DT * 3.0;
            let (sx, sy) = (40.0 + 14.0 * angle.cos(), 27.0 + 14.0 * angle.sin());
            fluid.add_force(sx, sy, -angle.sin() * 900.0, angle.cos() * 900.0, 6.0);
            fluid.add_dye(sx, sy, [1.4, 0.7, 0.25], 5.0);
            fluid.step(DT, &params);
        }
        fluid
    })
}

/// Every stage writes each cell from reads that do not depend on how the rows
/// were banded, so the field must come out bit-identical on any pool size.
#[test]
fn every_thread_count_computes_the_same_field() {
    let reference = stirred_on(1);
    for threads in [2, 3, 5, 7] {
        let f = stirred_on(threads);
        let grids = [
            ("u", &reference.velocity().u, &f.velocity().u),
            ("v", &reference.velocity().v, &f.velocity().v),
            ("dye", &reference.dye()[0], &f.dye()[0]),
        ];
        for (name, want, got) in grids {
            let differing = want
                .data
                .iter()
                .zip(&got.data)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "{name}: {differing} cells differ on {threads} threads"
            );
        }
    }
}
