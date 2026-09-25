//! Data-parallel primitives with a serial fallback.
//!
//! Every hot loop in the engine is written as "split the work into
//! independent chunks, run a pure function on each". This module is the one
//! place that decides whether those chunks run on one core or all of them:
//! with the `parallel` feature they go through rayon (natively a thread pool,
//! in the browser a pool of Web Workers over shared WASM memory), without it
//! they run in order on the calling thread.
//!
//! The algorithms above this layer are identical either way — the serial path
//! is literally the same closures called in a `for` loop — so `cargo test` on
//! the host verifies the same code the browser runs, and a run with
//! `--features parallel` verifies that the chunking itself is sound.

/// Parallel tasks per thread for work that does not split evenly.
const TASKS_PER_THREAD: usize = 4;

/// Number of worker threads the pool will spread work over. `1` when the
/// serial fallback is compiled in.
#[inline]
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Chunk length that hands every thread roughly the same amount of work while
/// staying at or above `min` elements, so a small job is not shredded into
/// chunks whose scheduling costs more than their contents.
#[inline]
pub fn chunk_len(total: usize, min: usize) -> usize {
    let per_thread = total.div_ceil(threads().max(1));
    per_thread.max(min).max(1)
}

/// Like [`chunk_len`], but cut into several chunks per thread, for work whose
/// cost per element is uneven (a dead particle is cheap, a respawn is not) or
/// whose threads do not all start at once (a worker woken from sleep starts
/// late). A thread that finishes early then steals a chunk instead of idling
/// until the slowest one is done. The serial fallback still gets one chunk.
#[inline]
pub fn balanced_len(total: usize, min: usize) -> usize {
    let tasks = match threads() {
        1 => 1,
        t => t * TASKS_PER_THREAD,
    };
    total.div_ceil(tasks).max(min).max(1)
}

/// Runs `f` on a pool thread and waits for it, so every parallel region `f`
/// opens forks from a worker rather than from the calling thread.
///
/// A region opened from outside the pool is injected into it while the caller
/// blocks, idle, until the region drains, and that handoff is paid per region.
/// A region opened from inside the pool is a work-stealing join that the
/// opening thread works on too. The engine step opens a few dozen regions a
/// frame, so it enters the pool once and opens them all from inside. The
/// serial fallback just calls `f`.
#[inline]
pub fn install<R, F>(f: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    #[cfg(feature = "parallel")]
    {
        rayon::scope(|_| f())
    }
    #[cfg(not(feature = "parallel"))]
    {
        f()
    }
}

/// Runs `f(chunk_index, chunk)` over consecutive `size`-long chunks of `data`.
#[inline]
pub fn chunks_mut<T, F>(data: &mut [T], size: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    let size = size.max(1);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (i, c) in data.chunks_mut(size).enumerate() {
            f(i, c);
        }
    }
}

/// Runs `f` over every item of `items`, in parallel when possible.
#[inline]
pub fn for_each_mut<T, F>(items: &mut [T], f: F)
where
    T: Send,
    F: Fn(&mut T) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.par_iter_mut().for_each(f);
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter_mut().for_each(f);
    }
}

/// Maps `f` over every item and sums the results.
#[inline]
pub fn sum_mut<T, F>(items: &mut [T], f: F) -> usize
where
    T: Send,
    F: Fn(&mut T) -> usize + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.par_iter_mut().map(f).sum()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter_mut().map(f).sum()
    }
}

/// Runs `f(row_index, row)` over every `w`-long row of `data`, grouping rows
/// into per-thread bands so a small grid does not pay one task per row.
#[inline]
pub fn rows_mut<T, F>(data: &mut [T], w: usize, h: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    let w = w.max(1);
    let rows_per_band = chunk_len(h, 4);
    chunks_mut(data, rows_per_band * w, |band, slab| {
        for (k, row) in slab.chunks_mut(w).enumerate() {
            f(band * rows_per_band + k, row);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_visit_every_element_once_in_order() {
        let mut v = vec![0u32; 1003];
        chunks_mut(&mut v, 17, |i, c| {
            for (k, x) in c.iter_mut().enumerate() {
                *x = (i * 17 + k) as u32 + 1;
            }
        });
        for (k, x) in v.iter().enumerate() {
            assert_eq!(*x, k as u32 + 1);
        }
    }

    #[test]
    fn rows_are_indexed_by_their_position() {
        let (w, h) = (7, 23);
        let mut v = vec![0usize; w * h];
        rows_mut(&mut v, w, h, |y, row| row.fill(y));
        for y in 0..h {
            assert!(v[y * w..(y + 1) * w].iter().all(|&r| r == y));
        }
    }

    #[test]
    fn sum_counts_across_items() {
        let mut items: Vec<usize> = (0..100).collect();
        assert_eq!(sum_mut(&mut items, |x| *x % 2), 50);
    }

    #[test]
    fn balanced_len_is_one_chunk_serially_and_several_per_thread_otherwise() {
        let total = 1_000_000usize;
        let chunks = total.div_ceil(balanced_len(total, 16));
        if threads() == 1 {
            assert_eq!(chunks, 1);
        } else {
            assert!(chunks > threads());
        }
        assert!(balanced_len(10, 64) >= 64);
        assert!(balanced_len(0, 0) >= 1);
    }

    #[test]
    fn regions_opened_inside_install_still_cover_every_element() {
        let v = install(|| {
            let mut v = vec![0usize; 1003];
            chunks_mut(&mut v, 17, |i, c| c.fill(i + 1));
            v
        });
        for (k, x) in v.iter().enumerate() {
            assert_eq!(*x, k / 17 + 1);
        }
    }

    #[test]
    fn chunk_len_never_zero() {
        assert!(chunk_len(0, 0) >= 1);
        assert!(chunk_len(5, 64) >= 64);
    }
}
