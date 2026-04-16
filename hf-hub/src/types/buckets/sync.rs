use std::collections::HashMap;

use super::BucketTreeEntry;
use crate::types::bucket_params::SyncDirection;

/// Action to perform for a single file during sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncAction {
    Upload,
    Download,
    Delete,
    Skip,
}

/// A single operation within a sync plan.
#[derive(Debug, Clone)]
pub struct SyncOperation {
    /// What action to take.
    pub action: SyncAction,
    /// Relative file path (forward-slash separated).
    pub path: String,
    /// File size in bytes, if known.
    pub size: Option<u64>,
    /// Human-readable reason for this action (e.g. "new file", "size differs", "identical").
    pub reason: String,
}

/// The computed sync plan — describes what will happen (or has happened) during a sync.
///
/// Returned by [`HFBucket::sync`](crate::bucket::HFBucket::sync).
#[derive(Debug, Clone)]
pub struct SyncPlan {
    /// Sync direction that produced this plan.
    pub direction: SyncDirection,
    /// All operations in the plan.
    pub operations: Vec<SyncOperation>,
    /// Bucket tree entries for download operations, keyed by relative path.
    /// Used internally during execution to avoid re-fetching metadata.
    pub(crate) download_entries: HashMap<String, BucketTreeEntry>,
}

impl SyncPlan {
    pub fn uploads(&self) -> usize {
        self.operations.iter().filter(|op| op.action == SyncAction::Upload).count()
    }

    pub fn downloads(&self) -> usize {
        self.operations.iter().filter(|op| op.action == SyncAction::Download).count()
    }

    pub fn deletes(&self) -> usize {
        self.operations.iter().filter(|op| op.action == SyncAction::Delete).count()
    }

    pub fn skips(&self) -> usize {
        self.operations.iter().filter(|op| op.action == SyncAction::Skip).count()
    }

    /// Total bytes to transfer (upload + download operations).
    pub fn transfer_bytes(&self) -> u64 {
        self.operations
            .iter()
            .filter(|op| op.action == SyncAction::Upload || op.action == SyncAction::Download)
            .filter_map(|op| op.size)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plan(ops: Vec<(SyncAction, Option<u64>)>) -> SyncPlan {
        SyncPlan {
            direction: SyncDirection::Upload,
            operations: ops
                .into_iter()
                .enumerate()
                .map(|(i, (action, size))| SyncOperation {
                    action,
                    path: format!("file_{i}.txt"),
                    size,
                    reason: "test".to_string(),
                })
                .collect(),
            download_entries: HashMap::new(),
        }
    }

    #[test]
    fn test_plan_counts() {
        let plan = make_plan(vec![
            (SyncAction::Upload, Some(100)),
            (SyncAction::Upload, Some(200)),
            (SyncAction::Download, Some(300)),
            (SyncAction::Delete, Some(50)),
            (SyncAction::Skip, None),
        ]);
        assert_eq!(plan.uploads(), 2);
        assert_eq!(plan.downloads(), 1);
        assert_eq!(plan.deletes(), 1);
        assert_eq!(plan.skips(), 1);
    }

    #[test]
    fn test_transfer_bytes() {
        let plan = make_plan(vec![
            (SyncAction::Upload, Some(100)),
            (SyncAction::Download, Some(300)),
            (SyncAction::Delete, Some(50)),
            (SyncAction::Skip, None),
        ]);
        assert_eq!(plan.transfer_bytes(), 400);
    }

    #[test]
    fn test_empty_plan() {
        let plan = make_plan(vec![]);
        assert_eq!(plan.uploads(), 0);
        assert_eq!(plan.downloads(), 0);
        assert_eq!(plan.deletes(), 0);
        assert_eq!(plan.skips(), 0);
        assert_eq!(plan.transfer_bytes(), 0);
    }
}
