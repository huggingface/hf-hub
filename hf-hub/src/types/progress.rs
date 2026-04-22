use std::sync::Arc;

/// Trait implemented by consumers to receive progress updates.
/// Implementations must be fast — avoid blocking I/O in on_progress().
pub trait ProgressHandler: Send + Sync {
    /// Called by the library each time progress changes.
    /// Receives a reference to avoid allocation; clone if you need to store it.
    fn on_progress(&self, event: &ProgressEvent);
}

/// A clonable handle to a progress handler.
pub type Progress = Arc<dyn ProgressHandler>;

/// Top-level progress event — either an upload or download event.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Upload(UploadEvent),
    Download(DownloadEvent),
}

/// Progress events for upload operations.
///
/// `Start` and `Complete` bookend the operation. `Progress` always represents
/// byte-level upload activity; `Committing` is a marker (no payload) emitted
/// once the commit API call begins. The gap between `Start` and the first
/// `Progress`, and the gap between `Committing` and `Complete`, are silent —
/// consumers can show a generic spinner during those windows.
#[derive(Debug, Clone)]
pub enum UploadEvent {
    /// Upload operation has started; total file count and bytes are known.
    Start { total_files: usize, total_bytes: u64 },
    /// Byte-level progress during xet/LFS upload.
    /// `files` contains per-file progress for xet uploads (may be empty).
    Progress {
        bytes_completed: u64,
        total_bytes: u64,
        bytes_per_sec: Option<f64>,
        transfer_bytes_completed: u64,
        transfer_bytes: u64,
        transfer_bytes_per_sec: Option<f64>,
        files: Vec<FileProgress>,
    },
    /// Commit API call has begun; no byte-level activity follows until `Complete`.
    Committing,
    /// Entire upload operation finished (all files, commit created).
    Complete,
}

/// Progress events for download operations.
#[derive(Debug, Clone)]
pub enum DownloadEvent {
    /// Download operation has started; file count and total bytes known.
    Start { total_files: usize, total_bytes: u64 },
    /// Per-file progress update. Only includes files whose state changed
    /// since the last event (delta, not full snapshot). Batched for
    /// efficiency during multi-file downloads (snapshot_download).
    Progress { files: Vec<FileProgress> },
    /// Aggregate byte-level progress for xet batch transfers.
    /// Separate from per-file Progress because xet provides aggregate
    /// stats, not per-file byte counts.
    AggregateProgress {
        bytes_completed: u64,
        total_bytes: u64,
        bytes_per_sec: Option<f64>,
    },
    /// All downloads finished.
    Complete,
}

/// Per-file progress info, used inside [`DownloadEvent::Progress`].
#[derive(Debug, Clone)]
pub struct FileProgress {
    pub filename: String,
    pub bytes_completed: u64,
    pub total_bytes: u64,
    pub status: FileStatus,
}

/// Lifecycle status of a single file within a transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    /// File transfer has been queued but no bytes received yet.
    Started,
    /// Bytes are actively being transferred.
    InProgress,
    /// All bytes have been received and the file is written to disk.
    Complete,
}

pub(crate) fn emit(handler: &Option<Progress>, event: ProgressEvent) {
    if let Some(h) = handler {
        h.on_progress(&event);
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
        emit(&progress, ProgressEvent::Download(DownloadEvent::Complete));
    }

    #[test]
    fn emit_records_events() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone());

        emit(
            &progress,
            ProgressEvent::Upload(UploadEvent::Start {
                total_files: 2,
                total_bytes: 1024,
            }),
        );
        emit(
            &progress,
            ProgressEvent::Upload(UploadEvent::Progress {
                bytes_completed: 512,
                total_bytes: 1024,
                bytes_per_sec: Some(100.0),
                transfer_bytes_completed: 0,
                transfer_bytes: 0,
                transfer_bytes_per_sec: None,
                files: vec![],
            }),
        );
        emit(&progress, ProgressEvent::Upload(UploadEvent::Complete));

        let events = handler.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], ProgressEvent::Upload(UploadEvent::Start { .. })));
        assert!(matches!(events[1], ProgressEvent::Upload(UploadEvent::Progress { .. })));
        assert!(matches!(events[2], ProgressEvent::Upload(UploadEvent::Complete)));
    }

    #[test]
    fn download_file_lifecycle() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone());

        emit(
            &progress,
            ProgressEvent::Download(DownloadEvent::Start {
                total_files: 1,
                total_bytes: 1000,
            }),
        );
        emit(
            &progress,
            ProgressEvent::Download(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: "file.bin".to_string(),
                    bytes_completed: 0,
                    total_bytes: 1000,
                    status: FileStatus::Started,
                }],
            }),
        );
        emit(
            &progress,
            ProgressEvent::Download(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: "file.bin".to_string(),
                    bytes_completed: 500,
                    total_bytes: 1000,
                    status: FileStatus::InProgress,
                }],
            }),
        );
        emit(
            &progress,
            ProgressEvent::Download(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: "file.bin".to_string(),
                    bytes_completed: 1000,
                    total_bytes: 1000,
                    status: FileStatus::Complete,
                }],
            }),
        );
        emit(&progress, ProgressEvent::Download(DownloadEvent::Complete));

        let events = handler.events();
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn upload_event_ordering() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone());

        emit(
            &progress,
            ProgressEvent::Upload(UploadEvent::Start {
                total_files: 1,
                total_bytes: 100,
            }),
        );
        emit(
            &progress,
            ProgressEvent::Upload(UploadEvent::Progress {
                bytes_completed: 50,
                total_bytes: 100,
                bytes_per_sec: None,
                transfer_bytes_completed: 0,
                transfer_bytes: 0,
                transfer_bytes_per_sec: None,
                files: vec![],
            }),
        );
        emit(&progress, ProgressEvent::Upload(UploadEvent::Committing));
        emit(&progress, ProgressEvent::Upload(UploadEvent::Complete));

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
        let progress: Option<Progress> = Some(handler.clone());

        emit(
            &progress,
            ProgressEvent::Upload(UploadEvent::Progress {
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
            }),
        );

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
