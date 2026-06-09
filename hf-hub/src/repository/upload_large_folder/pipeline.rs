//! Pipeline primitives for `upload_large_folder`: adaptive commit batching,
//! per-file stage seeding, shared-session xet upload batches, and the
//! single-flight committer.

const COMMIT_SIZE_SCALE: [usize; 10] = [20, 50, 75, 100, 125, 200, 250, 400, 600, 1000];

/// Adaptive commit-batch sizer mirroring Python's `COMMIT_SIZE_SCALE` logic.
/// Starts at index 1 (= 50). Grows one step on a fast full commit, shrinks one
/// step on failure, clamped to the scale bounds.
pub(crate) struct CommitChunkSizer {
    idx: usize,
}

impl CommitChunkSizer {
    pub(crate) fn new() -> Self {
        Self { idx: 1 }
    }

    pub(crate) fn target(&self) -> usize {
        COMMIT_SIZE_SCALE[self.idx]
    }

    pub(crate) fn update(&mut self, success: bool, nb_items: usize, duration_secs: f64) {
        if !success {
            self.idx = self.idx.saturating_sub(1);
        } else if nb_items >= COMMIT_SIZE_SCALE[self.idx] && duration_secs < 40.0 {
            self.idx = (self.idx + 1).min(COMMIT_SIZE_SCALE.len() - 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_sizer_grows_shrinks_clamps() {
        let mut s = CommitChunkSizer::new();
        assert_eq!(s.target(), 50); // index 1

        // Fast, full commit -> grow.
        s.update(true, 50, 10.0);
        assert_eq!(s.target(), 75);

        // Slow commit -> no growth.
        s.update(true, 75, 45.0);
        assert_eq!(s.target(), 75);

        // Failure -> shrink.
        s.update(false, 0, 1.0);
        assert_eq!(s.target(), 50);

        // Shrink never underflows past index 0.
        let mut low = CommitChunkSizer::new();
        low.update(false, 0, 1.0); // -> index 0 (20)
        low.update(false, 0, 1.0); // stays at 0
        assert_eq!(low.target(), 20);

        // Grow clamps at the top (1000).
        let mut high = CommitChunkSizer::new();
        for _ in 0..50 {
            let t = high.target();
            high.update(true, t, 1.0);
        }
        assert_eq!(high.target(), 1000);
    }
}
