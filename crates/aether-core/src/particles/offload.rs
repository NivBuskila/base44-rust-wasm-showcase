//! The particle op log: what the pool records instead of doing when the
//! particles live on the GPU.
//!
//! Every mutation a spell can make to the pool — a burst, a ring, an impulse,
//! a damping field, a grip — is a small fixed-size record here, in call order.
//! The browser drains the log once per frame and replays it inside the
//! compute shader before advancing the particles, so the GPU pool sees exactly
//! the sequence the CPU pool would have executed.
//!
//! Records are flat `f32`s so JS can read them as one typed-array view over
//! WASM memory; `OP_STRIDE` and `MAX_OPS` are mirrored in
//! `web/src/constants.ts` and asserted at boot.

/// Floats per op record.
pub const OP_STRIDE: usize = 12;

/// Ops kept per frame. Spells fire a handful; a supernova plus a rush plus two
/// grips is under ten. Past this the excess is dropped, never reallocated.
pub const MAX_OPS: usize = 64;

/// Record kinds, in slot 0 of each record.
pub const OP_BURST: f32 = 1.0;
pub const OP_RING: f32 = 2.0;
pub const OP_IMPULSE: f32 = 3.0;
pub const OP_DAMP: f32 = 4.0;
pub const OP_GRIP: f32 = 5.0;

/// A per-frame op log with a fixed backing store.
pub struct OpLog {
    data: Vec<f32>,
    len: usize,
}

impl OpLog {
    pub fn new() -> Self {
        Self {
            data: vec![0.0; OP_STRIDE * MAX_OPS],
            len: 0,
        }
    }

    /// Appends one record; `fields` fills slots `1..` after the kind.
    /// Silently drops the record when the log is full.
    pub fn push(&mut self, kind: f32, fields: &[f32]) {
        if self.len >= MAX_OPS {
            return;
        }
        let o = self.len * OP_STRIDE;
        let rec = &mut self.data[o..o + OP_STRIDE];
        rec.fill(0.0);
        rec[0] = kind;
        for (dst, src) in rec[1..].iter_mut().zip(fields) {
            *dst = *src;
        }
        self.len += 1;
    }

    /// Records this frame so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forgets the frame's records without touching the backing store.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// The live records, `len() * OP_STRIDE` floats.
    #[inline]
    pub fn records(&self) -> &[f32] {
        &self.data[..self.len * OP_STRIDE]
    }

    /// Start of the backing store, for the zero-copy JS view.
    #[inline]
    pub fn ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }
}

impl Default for OpLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_appended_in_order_with_kind_first() {
        let mut log = OpLog::new();
        log.push(OP_BURST, &[1.0, 2.0, 3.0]);
        log.push(OP_GRIP, &[9.0]);
        assert_eq!(log.len(), 2);
        let r = log.records();
        assert_eq!(&r[..4], &[OP_BURST, 1.0, 2.0, 3.0]);
        assert_eq!(r[OP_STRIDE], OP_GRIP);
        assert_eq!(r[OP_STRIDE + 1], 9.0);
        // Fields beyond the ones given are zeroed, not stale.
        assert_eq!(r[OP_STRIDE + 2], 0.0);
    }

    #[test]
    fn a_full_log_drops_instead_of_growing() {
        let mut log = OpLog::new();
        let ptr = log.ptr();
        for i in 0..(MAX_OPS + 10) {
            log.push(OP_IMPULSE, &[i as f32]);
        }
        assert_eq!(log.len(), MAX_OPS);
        assert_eq!(log.ptr(), ptr, "the backing store must never move");
        log.clear();
        assert!(log.is_empty());
    }
}
