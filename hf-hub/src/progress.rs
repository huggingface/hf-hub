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
//! individual operation only emits one variant family (e.g. `create_commit`
//! only emits `Upload(*)`, `snapshot_download` only emits `Download(*)`).
//!
//! ## Upload event sequence
//!
//! ```text
//!   Start ──┐
//!           │ (silent preflight: preupload API, LFS classification)
//!   Progress ── Progress ── … ── Progress
//!           │ (active upload — poll loop fires ~every 100ms)
//!   Committing
//!           │ (silent: commit API round-trip)
//!   Complete
//! ```
//!
//! ## Download event sequence
//!
//! ```text
//!   Start ──┐
//!           │ (HEAD fan-out may precede this for snapshot downloads)
//!   Progress ── Progress ── … ── AggregateProgress ── Progress ── …
//!           │ (Progress = per-file deltas; AggregateProgress = xet batch
//!           │  totals. Either or both, interleaved.)
//!   Complete
//! ```
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
//!   e.g. a file may receive multiple `FileStatus::Complete` events across `Progress` and a final cleanup emit in edge
//!   cases. Consumers that track completion should use a set keyed by filename to ignore repeats.

use std::sync::Arc;

/// Receives progress updates from long-running upload and download operations.
///
/// Register a handler by wrapping it in [`Progress`] (i.e. `Arc<dyn ProgressHandler>`)
/// and passing it to the `.progress(...)` setter of any method builder that
/// supports progress reporting (e.g. `repo.upload_file()`, `repo.snapshot_download()`,
/// `repo.create_commit()`, `bucket.upload_files()`, `bucket.sync()`).
///
/// The library calls [`on_progress`](Self::on_progress) by reference to avoid
/// cloning large per-file vectors; clone the event (or its fields) only if you
/// need to retain it past the call.
///
/// # Concurrency
///
/// `on_progress` may be called from any tokio task, including background poll
/// loops. The `Send + Sync` bound ensures this is sound, but implementations
/// must be safe under concurrent invocation — e.g. use interior mutability
/// with `Mutex` or atomics rather than relying on `&mut self`.
///
/// # Performance
///
/// Handlers sit on the upload/download hot path. During an active transfer a
/// poll task emits a `Progress` event roughly every 100ms, and for non-xet
/// downloads the library calls `on_progress` from inside the byte-streaming
/// loop itself. Treat `on_progress` as a hot callback: avoid blocking I/O,
/// heavy allocations, or lock contention.
///
/// # Error handling
///
/// `on_progress` cannot fail — there is no return value. If your handler needs
/// to report errors (e.g. a rendering failure), log internally; do not panic.
/// The library does not catch unwinds from handler calls.
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
    /// Invoked by the library for each progress event.
    ///
    /// The `event` reference is only valid for the duration of the call — clone
    /// it (or specific fields) if you need to keep it around.
    ///
    /// Must not block, panic, or perform heavy work. See the trait-level docs
    /// for the full contract.
    fn on_progress(&self, event: &ProgressEvent);
}

/// Shared-ownership wrapper around a [`ProgressHandler`] trait object.
///
/// Method builders accept `Option<Progress>` for the `progress` parameter —
/// not setting it (or passing `None` via `maybe_progress`) disables progress
/// emission entirely (zero cost). Internally it holds an `Arc<dyn ProgressHandler>`,
/// so cloning is cheap and the handler can be shared across concurrent tasks
/// within a single operation.
///
/// `progress` setters are annotated with `#[builder(into)]`, so any
/// [`ProgressHandler`] value, an `Arc<H>`, or an `Arc<dyn ProgressHandler>` can
/// be passed directly — `From` impls below handle the conversion.
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
/// // Owned handler — most common case.
/// let handler: Progress = Noop.into();
/// // Pre-shared via Arc, e.g. when the caller wants to inspect events later.
/// let shared: Progress = Arc::new(Noop).into();
/// // Wrap explicitly via the constructor.
/// let direct = Progress::new(Noop);
/// let maybe: Option<Progress> = Some(handler);
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
/// A single operation emits only one variant family — an upload operation
/// never produces `Download(*)` events and vice versa. Match on the outer
/// variant first to route to upload- or download-handling logic.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Emitted by upload operations: `upload_file`, `upload_folder`,
    /// `create_commit`, `HFBucket::upload_files`, and bucket sync in the
    /// upload direction.
    Upload(UploadEvent),
    /// Emitted by download operations: `download_file`, `snapshot_download`,
    /// `HFBucket::download_files`, and bucket sync in the download direction.
    Download(DownloadEvent),
}

/// Lifecycle events for a single upload operation.
///
/// # Event ordering
///
/// On success, every upload emits events in this exact order:
///
/// 1. [`Start`](Self::Start) — exactly once, with final totals known.
/// 2. Zero or more [`Progress`](Self::Progress) events — byte-level updates during active transfer. Not emitted for
///    operations that only delete or that contain no LFS/xet files. (Plain inline file uploads via the commit NDJSON do
///    not produce `Progress` events — they skip straight from `Start` to `Committing`.)
/// 3. [`Committing`](Self::Committing) — exactly once, immediately before the commit API call. Marks the transition out
///    of active byte transfer.
/// 4. [`Complete`](Self::Complete) — exactly once on successful completion.
///
/// If the operation fails, `Complete` is **not** emitted. Use the returned
/// `Result` to determine success, not the event stream.
///
/// # Silent gaps
///
/// There are two periods with no events by design:
///
/// - Between `Start` and the first `Progress`: the library is calling the preupload endpoint to classify files as LFS
///   vs. inline, building the xet session, and queuing file handles. For multi-file uploads this can take several
///   seconds. UIs should show a generic spinner here if visible feedback is desired.
/// - Between `Committing` and `Complete`: the server-side commit API call is in flight. Typically sub-second but can be
///   longer.
///
/// # Polling rate
///
/// During active xet uploads, `Progress` events fire from a background poll
/// task on a ~100ms interval. The handler should assume events arrive at up to
/// 10Hz.
#[derive(Debug, Clone)]
pub enum UploadEvent {
    /// Upload has begun; totals are known.
    ///
    /// - `total_files`: number of files the operation will attempt to upload (excludes deletes and other non-add
    ///   operations in a commit).
    /// - `total_bytes`: sum of source-content sizes in bytes, before any xet deduplication.
    Start {
        /// Number of files the operation will upload.
        total_files: usize,
        /// Sum of source-content sizes in bytes, before xet deduplication.
        total_bytes: u64,
    },

    /// Byte-level progress during the active upload phase.
    ///
    /// Emitted repeatedly (~10Hz) by the xet upload poll loop. The top-level
    /// byte fields represent aggregate totals across all files in the upload;
    /// `files` carries per-file breakdowns for UIs that display individual
    /// file progress bars.
    ///
    /// # Aggregate byte fields
    ///
    /// Two byte-count dimensions are reported because xet performs
    /// content-defined deduplication: the client fingerprints content locally
    /// and only sends chunks the server doesn't already have.
    ///
    /// - `bytes_completed` / `total_bytes`: **logical content bytes**. Progress against the original uncompressed file
    ///   sizes reported in `Start`. Use this for a "% of content processed" progress bar.
    /// - `transfer_bytes_completed` / `transfer_bytes` / `transfer_bytes_per_sec`: **post-dedup network bytes**. Only
    ///   the chunks that actually leave the machine. For deduplicated data, `transfer_bytes` ≪ `total_bytes`. Use this
    ///   for a "network activity" bar.
    /// - `bytes_per_sec`: rate of logical content processing (None during warm-up before enough samples exist).
    ///
    /// # Per-file progress (`files`)
    ///
    /// A snapshot of every xet-tracked file's current state at the moment of
    /// this event. Each entry's `status` reflects whether the file is
    /// [`Started`](FileStatus::Started) (queued, no bytes yet),
    /// [`InProgress`](FileStatus::InProgress) (actively transferring), or
    /// [`Complete`](FileStatus::Complete).
    ///
    /// `files` may be empty for operations that don't go through xet (e.g.
    /// small inline files — those skip `Progress` entirely and are reported
    /// only via the final commit).
    ///
    /// A given file may appear as `Complete` in multiple `Progress` events
    /// (once from the poll loop's last observation, once from a final
    /// cleanup emit). Handlers that track completion should use a set keyed
    /// by filename to dedupe.
    Progress {
        /// Logical content bytes processed so far across all files.
        bytes_completed: u64,
        /// Total logical content bytes for the operation (matches `Start.total_bytes`).
        total_bytes: u64,
        /// Rate of logical content processing in bytes/sec. `None` during warm-up.
        bytes_per_sec: Option<f64>,
        /// Post-dedup network bytes actually sent so far.
        transfer_bytes_completed: u64,
        /// Total post-dedup network bytes the operation is expected to send. Typically `≪ total_bytes`.
        transfer_bytes: u64,
        /// Rate of network transfer in bytes/sec. `None` during warm-up.
        transfer_bytes_per_sec: Option<f64>,
        /// Per-file snapshot of every xet-tracked file in the upload (may be empty for non-xet flows).
        files: Vec<FileProgress>,
    },

    /// Emitted once, immediately before the commit API call.
    ///
    /// Signals that all byte transfer is done and the server is being asked
    /// to finalize the commit. Consumers typically swap upload progress bars
    /// for a spinner at this point. The call itself is silent — no further
    /// events fire until `Complete`.
    Committing,

    /// Emitted once on successful completion. Terminal event.
    ///
    /// Not emitted if the operation fails — always check the returned
    /// `Result`.
    Complete,
}

/// Lifecycle events for a single download operation.
///
/// # Event ordering
///
/// On success, every download emits events in this order:
///
/// 1. [`Start`](Self::Start) — exactly once. For `snapshot_download`, fires **after** the HEAD fan-out so `total_bytes`
///    reflects the real size. For single-file downloads, fires after the HEAD round-trip. For instant cache hits (file
///    already in the snapshot cache, 304 Not-Modified, `local_files_only` hits), `Start` may be skipped and only
///    `Complete` fires — see "Cache hits" below.
/// 2. Zero or more [`Progress`](Self::Progress) and/or [`AggregateProgress`](Self::AggregateProgress) events —
///    interleaved (see "Two progress channels" below).
/// 3. [`Complete`](Self::Complete) — exactly once on success.
///
/// On error, `Complete` is not emitted.
///
/// # Two progress channels
///
/// Downloads emit **two distinct** kinds of progress events because the xet
/// protocol reports aggregate bytes but not per-file bytes, while non-xet
/// downloads are naturally per-file:
///
/// - [`Progress`](Self::Progress): per-file events. The `files` vec is a **delta** — it contains only files whose
///   status or byte count changed since the last `Progress` event, not a snapshot of all files. Consumers that want a
///   complete view must accumulate state by filename.
/// - [`AggregateProgress`](Self::AggregateProgress): emitted only during xet batch transfers, reports batch-wide byte
///   totals/rate. No per-file breakdown — xet reports aggregate stats only.
///
/// A mixed snapshot_download (some files via xet, others via plain HTTP)
/// produces both: `AggregateProgress` for the xet files' aggregate bytes,
/// and `Progress` for per-file status transitions. UIs typically track an
/// aggregate bar from `AggregateProgress` and per-file bars from `Progress`.
///
/// # Cache hits
///
/// The download APIs short-circuit when files are already present in the
/// local cache or local directory. For a snapshot_download where every file
/// is already cached, the sequence is just `Start` → one `Progress` event
/// listing the cached files as `Complete` → `Complete`. For a single-file
/// `download_file` whose content is already on disk, the fast path may emit
/// only `Complete` with no preceding `Start`.
#[derive(Debug, Clone)]
pub enum DownloadEvent {
    /// Download operation has begun; totals are known.
    ///
    /// - `total_files`: number of files to download.
    /// - `total_bytes`: sum of remote file sizes in bytes (reported by HEAD responses). For single-file downloads, the
    ///   size of that file.
    Start {
        /// Number of files to download.
        total_files: usize,
        /// Sum of remote file sizes in bytes, as reported by HEAD responses.
        total_bytes: u64,
    },

    /// Per-file progress delta for one or more files.
    ///
    /// `files` contains **only files whose state changed** since the
    /// previous `Progress` event (new `Started`, byte-count update, or
    /// transition to `Complete`). Consumers that want a running view of
    /// every file must accumulate state across events keyed by filename.
    ///
    /// Batched emission: during multi-file downloads the library coalesces
    /// per-tick updates into a single event with multiple entries rather
    /// than firing many small events.
    ///
    /// A single file may appear with `FileStatus::Complete` across multiple
    /// events. Handlers that want exactly-once completion handling should
    /// track a seen-set of filenames.
    Progress {
        /// Per-file delta — only files whose state changed since the previous `Progress` event.
        files: Vec<FileProgress>,
    },

    /// Aggregate byte-level progress for the in-flight xet batch.
    ///
    /// Emitted only for xet-backed downloads, roughly every 100ms by the
    /// xet poll loop. Reports cumulative bytes for the entire batch — there
    /// is no per-file breakdown from xet's perspective.
    ///
    /// `bytes_per_sec` is `None` until enough samples have accumulated to
    /// compute a rate.
    AggregateProgress {
        /// Bytes downloaded so far across the in-flight xet batch.
        bytes_completed: u64,
        /// Total bytes for the in-flight xet batch.
        total_bytes: u64,
        /// Download rate in bytes/sec. `None` until enough samples accumulate to compute a rate.
        bytes_per_sec: Option<f64>,
    },

    /// All downloads finished successfully. Terminal event.
    Complete,
}

/// Progress for a single file, carried in `Progress` events.
///
/// Used by both [`UploadEvent::Progress`] (where `files` is a per-event
/// snapshot of all tracked files) and [`DownloadEvent::Progress`] (where
/// `files` is a delta of files whose state changed since the last event).
/// See the parent variant's docs for the exact semantics.
///
/// # Bytes
///
/// - `bytes_completed` / `total_bytes` are logical file content bytes.
/// - For files reported as [`FileStatus::Complete`], both fields equal the file's size (unless the file's size was
///   unknown, in which case they may both be zero — e.g. pre-HEAD cached files in `snapshot_download`).
#[derive(Debug, Clone)]
pub struct FileProgress {
    /// Path as known by the repository or bucket — the `path_in_repo` used
    /// when uploading, or the remote path as returned from tree listing.
    pub filename: String,
    /// Bytes transferred so far for this file.
    pub bytes_completed: u64,
    /// Total bytes expected for this file. Zero when the size is unknown
    /// (e.g. certain fast-path cached files emitted purely to signal
    /// completion).
    pub total_bytes: u64,
    /// Current lifecycle stage.
    pub status: FileStatus,
}

/// Lifecycle stage of an individual file within a transfer.
///
/// A file progresses through the stages in order: `Started` → `InProgress`
/// → `Complete`. Not every stage is observed for every file — a fast
/// transfer may skip directly from `Started` to `Complete` between two
/// polls, and files served from cache emit a single `Complete` with no
/// prior states.
///
/// A file may be reported as `Complete` more than once across events.
/// Consumers that need exactly-once completion semantics should track a
/// per-filename set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    /// File has been queued for transfer but no bytes have moved yet.
    Started,
    /// Bytes are actively being transferred.
    InProgress,
    /// All bytes for this file have been transferred and written to disk.
    /// Terminal state.
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
