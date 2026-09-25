//! Progress reporting for upload and download operations.
//!
//! Implement [`ProgressHandler`] and pass the handler to the `.progress(...)` setter on any
//! method builder that supports progress reporting (upload, download, snapshot download,
//! `create_commit`, bucket sync, etc.). Each `.progress(...)` argument is converted to [`Progress`]
//! via [`Into`]: an owned handler, an `Arc<H>`, [`Progress::new`], or a [`Progress`] value. When no handler is set,
//! the library emits nothing — there is no runtime cost.
//!
//! # Event model
//!
//! Every operation that supports progress emits a stream of [`ProgressEvent`]s
//! framed by a `Start` event and a `Complete` event. If the operation returns
//! an error, `Complete` is **not** emitted — consumers should rely on the
//! returned `Result` for operation success, not on observing `Complete`.
//!
//! Upload and download events are distinct enums ([`UploadEvent`] and
//! [`DownloadEvent`]) wrapped in a [`ProgressEvent`] discriminator. An
//! individual operation only emits one variant family (e.g., `create_commit`
//! only emits `Upload(*)`, `snapshot_download` only emits `Download(*)`).
//!
//! ## Upload event sequence
//!
//! ```text
//!   Start ──┐
//!           │ (silent preflight: preupload API, LFS classification)
//!   Progress{phase: Preparing} ── … ── Progress{phase: Uploading} ── …
//!           │ (xet poll loop, ~every 100ms)
//!   Committing
//!           │ (silent: commit API round-trip, or bucket batch registration)
//!   Complete
//! ```
//!
//! Upload contract:
//!
//! - `Start` is emitted once, first. Its totals do not change afterwards; `Progress.total_bytes` is xet's own count of
//!   the same content and may differ slightly (e.g. while sizes are discovered for streams).
//! - Every `Progress` carries an [`UploadPhase`]. It is `Preparing` while xet is still reading, chunking and hashing
//!   and no byte counter has moved, and `Uploading` from the first moved byte on. It does not go back to `Preparing`,
//!   including in the final `Progress`; an upload whose counters never move (e.g. only empty files) stays `Preparing`.
//! - `bytes_completed` only counts content whose xorbs have been uploaded or deduplicated, so it can sit at 0 during
//!   hashing and lag the network; `transfer_bytes_*` tracks the network. Per-file `Complete` statuses can arrive late,
//!   often only in the final `Progress`.
//! - `Committing` is emitted once, after the last byte-level `Progress` and before the commit (repo) or batch
//!   registration (bucket) call. `Complete` follows on success.
//! - Uploads that do not go through xet (small inline files) may skip `Progress` entirely.
//!
//! ## Download event sequence
//!
//! ```text
//!   Start ──┐
//!           │ (HEAD fan-out precedes this for snapshot downloads)
//!   Progress ── Progress ── … ── AggregateProgress ── Progress ── …
//!           │ (Progress = per-file deltas; AggregateProgress = xet batch
//!           │  totals. Either or both, interleaved.)
//!   Complete
//! ```
//!
//! Download contract, for `download_file`, `snapshot_download` and `HFBucket::download_files`:
//!
//! - `Start` is emitted exactly once per call, before any other event. Its `total_files` and `total_bytes` are
//!   authoritative and never re-announced: a snapshot does not emit a `Start` per file. For `snapshot_download`,
//!   `total_files` counts every selected file that exists at the resolved commit (including files already present
//!   locally), and `total_bytes` is the sum of their sizes.
//! - Every file counted in `Start.total_files` gets at least one [`FileProgress`] with [`FileStatus::Complete`] before
//!   `Complete`, carrying `bytes_completed == total_bytes == <file size>`. That includes files that needed no transfer
//!   (already in the snapshot, blob already cached, destination already present, or content shared with another file of
//!   the same call). Repeats are possible; key by filename.
//! - Per file, `total_bytes` is constant and `bytes_completed` never decreases. The sum over files of the latest
//!   `bytes_completed` therefore rises monotonically to `Start.total_bytes`, and is the recommended overall byte
//!   counter.
//! - For xet files, per-file `bytes_completed` is bytes written to disk. Xet writes each file in order, so a single
//!   large file can report little progress while later parts are already downloaded;
//!   `AggregateProgress.transfer_bytes_completed` shows the network activity.
//! - `AggregateProgress` describes only the xet files of the call. Its `total_bytes` is not the operation total and
//!   should not replace `Start.total_bytes`; use it for rates and network activity.
//! - `Complete` is emitted once, last, on success only.
//! - `download_file` keeps this contract on cache hits too (snapshot already present, blob already cached, `304 Not
//!   Modified`, `local_files_only`), sizing the file from disk when no transfer happens. One exception: if a transient
//!   network error after `Start` makes it fall back to an older cached copy, that file's `Complete` reuses the size
//!   already announced in `Start` rather than re-reading it from disk, so it always matches `Start.total_bytes`. If no
//!   `Start` was announced yet and the cached file's size can't be read, the progress event for that file is skipped
//!   (logged as a warning) instead of failing the call.
//! - `snapshot_download` with `local_files_only` makes no network calls and emits no events at all.
//!
//! # Implementing a handler
//!
//! ```
//! use std::sync::Arc;
//!
//! use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler, UploadEvent};
//!
//! struct PrintHandler;
//!
//! impl ProgressHandler for PrintHandler {
//!     fn on_progress(&self, event: &ProgressEvent) {
//!         match event {
//!             ProgressEvent::Upload(UploadEvent::Start {
//!                 total_files,
//!                 total_bytes,
//!             }) => {
//!                 println!("Uploading {total_files} file(s), {total_bytes} bytes");
//!             },
//!             ProgressEvent::Upload(UploadEvent::Progress {
//!                 bytes_completed,
//!                 total_bytes,
//!                 ..
//!             }) => {
//!                 println!("  {bytes_completed}/{total_bytes}");
//!             },
//!             ProgressEvent::Upload(UploadEvent::Committing) => {
//!                 println!("Committing...");
//!             },
//!             ProgressEvent::Upload(UploadEvent::Complete) => {
//!                 println!("Done.");
//!             },
//!             _ => {},
//!         }
//!     }
//! }
//! ```
//!
//! # Thread safety and performance contract
//!
//! [`ProgressHandler`] requires `Send + Sync` because the library may invoke
//! `on_progress` from arbitrary tokio tasks, including background poll loops
//! running on ~100ms tick intervals during active transfers. Implementations
//! should:
//!
//! - **Never block.** Blocking `on_progress` blocks the emitting task, which for upload/download poll loops means
//!   delaying subsequent progress observations. For network streams the library calls `on_progress` on the stream-read
//!   path itself — slow handlers directly slow the transfer.
//! - **Not panic.** Panics propagate through the tokio runtime and can abort the operation.
//! - **Be idempotent / tolerant of redundant state.** The library guarantees event *ordering* but not *deduplication*;
//!   e.g., a file may receive multiple `FileStatus::Complete` events across `Progress` and a final cleanup emit in edge
//!   cases. Consumers that track completion should use a set keyed by filename to ignore repeats.

use std::sync::Arc;

/// Receives progress updates from long-running upload and download operations.
///
/// Register a handler by wrapping it in [`Progress`] and passing it to the
/// `.progress(...)` setter of any method builder that supports progress reporting.
/// See the [module-level docs](self) for the event model, ordering guarantees, and
/// the must-not-block / `Send + Sync` contract.
///
/// # Example
///
/// ```
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// use hf_hub::progress::{ProgressEvent, ProgressHandler, UploadEvent};
///
/// struct ByteCounter {
///     bytes: AtomicU64,
/// }
///
/// impl ProgressHandler for ByteCounter {
///     fn on_progress(&self, event: &ProgressEvent) {
///         if let ProgressEvent::Upload(UploadEvent::Progress {
///             bytes_completed, ..
///         }) = event
///         {
///             self.bytes.store(*bytes_completed, Ordering::Relaxed);
///         }
///     }
/// }
/// ```
pub trait ProgressHandler: Send + Sync {
    /// Invoked by the library for each progress event. The `event` reference is
    /// only valid for the duration of the call.
    fn on_progress(&self, event: &ProgressEvent);
}

/// Shared-ownership wrapper around a [`ProgressHandler`] trait object.
///
/// Internally an `Arc<dyn ProgressHandler>`, so cloning is cheap. `progress` setters
/// take `impl Into<Progress>`, so an owned handler, an `Arc<H>`, or an
/// `Arc<dyn ProgressHandler>` can be passed directly.
///
/// ```
/// use std::sync::Arc;
///
/// use hf_hub::progress::{Progress, ProgressEvent, ProgressHandler};
///
/// struct Noop;
/// impl ProgressHandler for Noop {
///     fn on_progress(&self, _event: &ProgressEvent) {}
/// }
///
/// let handler: Progress = Noop.into();
/// let shared: Progress = Arc::new(Noop).into();
/// let direct = Progress::new(Noop);
/// ```
pub struct Progress(Arc<dyn ProgressHandler>);

impl Progress {
    /// Wrap a handler value in a new `Progress`.
    pub fn new<H: ProgressHandler + 'static>(handler: H) -> Self {
        Self(Arc::new(handler))
    }
}

impl Clone for Progress {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress").finish_non_exhaustive()
    }
}

impl std::ops::Deref for Progress {
    type Target = dyn ProgressHandler;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<H: ProgressHandler + 'static> From<H> for Progress {
    fn from(handler: H) -> Self {
        Self(Arc::new(handler))
    }
}

impl<H: ProgressHandler + 'static> From<Arc<H>> for Progress {
    fn from(handler: Arc<H>) -> Self {
        Self(handler)
    }
}

impl From<Arc<dyn ProgressHandler>> for Progress {
    fn from(handler: Arc<dyn ProgressHandler>) -> Self {
        Self(handler)
    }
}

/// Top-level progress event dispatched to [`ProgressHandler::on_progress`].
///
/// A single operation emits only one variant family — uploads never produce
/// `Download(*)` and vice versa.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Emitted by upload operations (`upload_file`, `upload_folder`, `create_commit`,
    /// `HFBucket::upload_files`, bucket sync in the upload direction).
    Upload(UploadEvent),
    /// Emitted by download operations (`download_file`, `snapshot_download`,
    /// `HFBucket::download_files`, bucket sync in the download direction).
    Download(DownloadEvent),
}

/// Lifecycle events for a single upload operation. See the [module-level
/// docs](self) for the `Start` → `Progress` → `Committing` → `Complete` ordering and
/// the silent-gap caveats.
#[derive(Debug, Clone)]
pub enum UploadEvent {
    /// Upload has begun; totals are known.
    Start {
        /// Number of files the operation will upload (excludes deletes and other
        /// non-add operations in a commit).
        total_files: usize,
        /// Sum of source-content sizes in bytes, before xet deduplication.
        total_bytes: u64,
    },

    /// Byte-level progress during the active upload phase, emitted at ~10Hz by the
    /// xet upload poll loop.
    ///
    /// Two byte-count dimensions are reported because xet performs content-defined
    /// deduplication. The `bytes_completed` / `total_bytes` pair tracks logical
    /// content bytes (use for a "% processed" bar); the `transfer_bytes_*` triplet
    /// tracks post-dedup network bytes actually sent (use for a "network activity"
    /// bar). For deduplicated data, `transfer_bytes` ≪ `total_bytes`.
    ///
    /// `files` is a snapshot of every xet-tracked file's state at this event. May
    /// be empty for operations that don't go through xet (small inline files skip
    /// `Progress` entirely).
    Progress {
        /// Whether bytes have started moving. See [`UploadPhase`].
        phase: UploadPhase,
        /// Logical content bytes processed so far across all files.
        bytes_completed: u64,
        /// Total logical content bytes for the operation (matches `Start.total_bytes`).
        total_bytes: u64,
        /// Rate of logical content processing in bytes/sec. `None` during warm-up.
        bytes_per_sec: Option<f64>,
        /// Post-dedup network bytes actually sent so far.
        transfer_bytes_completed: u64,
        /// Total post-dedup network bytes the operation is expected to send.
        transfer_bytes: u64,
        /// Rate of network transfer in bytes/sec. `None` during warm-up.
        transfer_bytes_per_sec: Option<f64>,
        /// Per-file snapshot of every xet-tracked file in the upload.
        files: Vec<FileProgress>,
    },

    /// Emitted once, immediately before the commit API call. Signals that all byte
    /// transfer is done; the call itself is silent until `Complete`.
    Committing,

    /// Terminal event on success. Not emitted on failure — check the returned `Result`.
    Complete,
}

/// Lifecycle events for a single download operation. See the [module-level
/// docs](self) for ordering, the two-channel `Progress` vs `AggregateProgress`
/// model, and cache-hit fast paths.
#[derive(Debug, Clone)]
pub enum DownloadEvent {
    /// Download operation has begun; totals are known. Fires exactly once per call, after the
    /// HEAD round-trip (or HEAD fan-out for `snapshot_download`), and its totals are final.
    Start {
        /// Number of files the call covers, including ones that need no transfer.
        total_files: usize,
        /// Sum of those files' sizes in bytes.
        total_bytes: u64,
    },

    /// Per-file progress **delta** — `files` contains only files whose status or
    /// byte count changed since the previous `Progress` event. Consumers wanting a
    /// running view of every file must accumulate state by filename.
    Progress {
        /// Files whose state changed since the previous `Progress` event.
        files: Vec<FileProgress>,
    },

    /// Aggregate byte-level progress for the in-flight xet batch (~10Hz). Reports
    /// cumulative bytes for the entire batch with no per-file breakdown — xet
    /// reports aggregate stats only.
    AggregateProgress {
        /// File bytes written to disk so far across the in-flight xet batch. Xet writes each file
        /// in order, so this can trail the network while an early part of a file is still in flight.
        bytes_completed: u64,
        /// Total file bytes for the in-flight xet batch. Covers only the xet files of this call,
        /// not the operation total from `Start`.
        total_bytes: u64,
        /// Rate of `bytes_completed` in bytes/sec. `None` until enough samples accumulate.
        bytes_per_sec: Option<f64>,
        /// Network bytes received so far for the batch (compressed, after local chunk-cache hits).
        transfer_bytes_completed: u64,
        /// Network bytes the batch is expected to receive, as known so far. Grows while xet
        /// discovers the reconstruction plan.
        transfer_bytes: u64,
        /// Network receive rate in bytes/sec. `None` until enough samples accumulate.
        transfer_bytes_per_sec: Option<f64>,
    },

    /// Terminal event on success. Not emitted on failure — check the returned `Result`.
    Complete,
}

/// Stage of an upload reported on every [`UploadEvent::Progress`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UploadPhase {
    /// Xet is reading, chunking and hashing the sources; nothing has been uploaded or matched
    /// against existing content yet, so every byte counter is still zero. Large files can stay
    /// here for a while.
    Preparing,
    /// Content is being uploaded (or found to exist already). Byte counters move from here on.
    Uploading,
}

impl UploadPhase {
    pub(crate) fn from_counters(bytes_completed: u64, transfer_bytes_completed: u64) -> Self {
        if bytes_completed == 0 && transfer_bytes_completed == 0 {
            Self::Preparing
        } else {
            Self::Uploading
        }
    }
}

/// Phase reported across the `Progress` events of one upload: `Uploading` from the first moved
/// counter on, even if a later report reads lower counters.
#[derive(Debug, Default)]
pub(crate) struct UploadPhaseTracker(std::sync::atomic::AtomicBool);

impl UploadPhaseTracker {
    pub(crate) fn observe(&self, bytes_completed: u64, transfer_bytes_completed: u64) -> UploadPhase {
        use std::sync::atomic::Ordering;
        if UploadPhase::from_counters(bytes_completed, transfer_bytes_completed) == UploadPhase::Uploading {
            self.0.store(true, Ordering::Relaxed);
        }
        if self.0.load(Ordering::Relaxed) {
            UploadPhase::Uploading
        } else {
            UploadPhase::Preparing
        }
    }
}

/// Progress for a single file, carried in `Progress` events. See the parent
/// variant's docs for whether `files` is a snapshot ([`UploadEvent::Progress`]) or
/// a delta ([`DownloadEvent::Progress`]).
#[derive(Debug, Clone)]
pub struct FileProgress {
    /// Path as known by the repository or bucket (the `path_in_repo` used when
    /// uploading, or the remote path as returned from tree listing).
    pub filename: String,
    /// Bytes transferred so far for this file.
    pub bytes_completed: u64,
    /// Total bytes expected for this file. Zero when the size is unknown (e.g.,
    /// fast-path cached files emitted purely to signal completion).
    pub total_bytes: u64,
    /// Current lifecycle stage.
    pub status: FileStatus,
}

/// Lifecycle stage of an individual file within a transfer: `Started` →
/// `InProgress` → `Complete`. Not every stage is observed for every file (fast
/// transfers may skip from `Started` to `Complete`, cache hits emit only `Complete`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    /// File has been queued for transfer but no bytes have moved yet.
    Started,
    /// Bytes are actively being transferred.
    InProgress,
    /// All bytes for this file have been transferred. Terminal state.
    Complete,
}

impl From<UploadEvent> for ProgressEvent {
    fn from(event: UploadEvent) -> Self {
        ProgressEvent::Upload(event)
    }
}

impl From<DownloadEvent> for ProgressEvent {
    fn from(event: DownloadEvent) -> Self {
        ProgressEvent::Download(event)
    }
}

pub(crate) trait EmitEvent {
    fn emit(&self, event: impl Into<ProgressEvent>);
}

impl<T: ProgressHandler + ?Sized> EmitEvent for T {
    fn emit(&self, event: impl Into<ProgressEvent>) {
        self.on_progress(&event.into());
    }
}

impl EmitEvent for Option<Progress> {
    fn emit(&self, event: impl Into<ProgressEvent>) {
        if let Some(h) = self {
            h.on_progress(&event.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct RecordingHandler {
        events: Mutex<Vec<ProgressEvent>>,
    }

    impl RecordingHandler {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn events(&self) -> Vec<ProgressEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ProgressHandler for RecordingHandler {
        fn on_progress(&self, event: &ProgressEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    #[test]
    fn upload_phase_is_preparing_until_a_byte_counter_moves() {
        assert_eq!(UploadPhase::from_counters(0, 0), UploadPhase::Preparing);
        assert_eq!(UploadPhase::from_counters(0, 1), UploadPhase::Uploading);
        assert_eq!(UploadPhase::from_counters(1, 0), UploadPhase::Uploading);
    }

    #[test]
    fn upload_phase_tracker_never_goes_back_to_preparing() {
        use UploadPhase::{Preparing, Uploading};
        let tracker = UploadPhaseTracker::default();
        let phases: Vec<_> = [(0, 0), (0, 0), (0, 512), (0, 0), (1024, 512), (0, 0)]
            .into_iter()
            .map(|(bytes, transfer)| tracker.observe(bytes, transfer))
            .collect();
        assert_eq!(phases, [Preparing, Preparing, Uploading, Uploading, Uploading, Uploading]);
    }

    #[test]
    fn handler_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<RecordingHandler>>();
    }

    #[test]
    fn emit_with_none_is_noop() {
        let progress: Option<Progress> = None;
        progress.emit(DownloadEvent::Complete);
    }

    #[test]
    fn emit_records_events() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(UploadEvent::Start {
            total_files: 2,
            total_bytes: 1024,
        });
        progress.emit(UploadEvent::Progress {
            phase: UploadPhase::Uploading,
            bytes_completed: 512,
            total_bytes: 1024,
            bytes_per_sec: Some(100.0),
            transfer_bytes_completed: 0,
            transfer_bytes: 0,
            transfer_bytes_per_sec: None,
            files: vec![],
        });
        progress.emit(UploadEvent::Complete);

        let events = handler.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], ProgressEvent::Upload(UploadEvent::Start { .. })));
        assert!(matches!(events[1], ProgressEvent::Upload(UploadEvent::Progress { .. })));
        assert!(matches!(events[2], ProgressEvent::Upload(UploadEvent::Complete)));
    }

    #[test]
    fn download_file_lifecycle() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 1000,
        });
        progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "file.bin".to_string(),
                bytes_completed: 0,
                total_bytes: 1000,
                status: FileStatus::Started,
            }],
        });
        progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "file.bin".to_string(),
                bytes_completed: 500,
                total_bytes: 1000,
                status: FileStatus::InProgress,
            }],
        });
        progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "file.bin".to_string(),
                bytes_completed: 1000,
                total_bytes: 1000,
                status: FileStatus::Complete,
            }],
        });
        progress.emit(DownloadEvent::Complete);

        let events = handler.events();
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn upload_event_ordering() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(UploadEvent::Start {
            total_files: 1,
            total_bytes: 100,
        });
        progress.emit(UploadEvent::Progress {
            phase: UploadPhase::Uploading,
            bytes_completed: 50,
            total_bytes: 100,
            bytes_per_sec: None,
            transfer_bytes_completed: 0,
            transfer_bytes: 0,
            transfer_bytes_per_sec: None,
            files: vec![],
        });
        progress.emit(UploadEvent::Committing);
        progress.emit(UploadEvent::Complete);

        let events = handler.events();
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], ProgressEvent::Upload(UploadEvent::Start { .. })));
        assert!(matches!(events[1], ProgressEvent::Upload(UploadEvent::Progress { .. })));
        assert!(matches!(events[2], ProgressEvent::Upload(UploadEvent::Committing)));
        assert!(matches!(events[3], ProgressEvent::Upload(UploadEvent::Complete)));
    }

    #[test]
    fn upload_progress_with_per_file_data() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(UploadEvent::Progress {
            phase: UploadPhase::Uploading,
            bytes_completed: 500,
            total_bytes: 1000,
            bytes_per_sec: Some(100.0),
            transfer_bytes_completed: 250,
            transfer_bytes: 800,
            transfer_bytes_per_sec: Some(50.0),
            files: vec![
                FileProgress {
                    filename: "model/weights.bin".to_string(),
                    bytes_completed: 300,
                    total_bytes: 600,
                    status: FileStatus::InProgress,
                },
                FileProgress {
                    filename: "config.json".to_string(),
                    bytes_completed: 200,
                    total_bytes: 400,
                    status: FileStatus::InProgress,
                },
            ],
        });

        let events = handler.events();
        assert_eq!(events.len(), 1);
        if let ProgressEvent::Upload(UploadEvent::Progress {
            files,
            transfer_bytes_completed,
            transfer_bytes,
            transfer_bytes_per_sec,
            ..
        }) = &events[0]
        {
            assert_eq!(files.len(), 2);
            assert_eq!(files[0].filename, "model/weights.bin");
            assert_eq!(files[1].filename, "config.json");
            assert_eq!(*transfer_bytes_completed, 250);
            assert_eq!(*transfer_bytes, 800);
            assert_eq!(*transfer_bytes_per_sec, Some(50.0));
        } else {
            panic!("expected Upload(Progress)");
        }
    }
}
