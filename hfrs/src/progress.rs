use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::sync::Mutex;

use hf_hub::progress::{DownloadEvent, FileProgress, FileStatus, ProgressEvent, ProgressHandler, UploadEvent};
use indicatif::{HumanBytes, HumanDuration, MultiProgress, ProgressBar, ProgressStyle};

/// Renders indicatif progress bars in the terminal for download and upload operations.
const MAX_VISIBLE_FILE_BARS: usize = 10;
const MAX_VISIBLE_UPLOAD_BARS: usize = 10;

pub struct CliProgressHandler {
    multi: MultiProgress,
    state: Mutex<ProgressState>,
}

struct ProgressState {
    // Download state
    files_bar: Option<ProgressBar>,
    /// Byte bar for single-file downloads. Created from `Start`/`Started`
    /// when `total_files == 1`. Kept separate from `aggregate_bar` so per-file
    /// Complete/InProgress updates never clobber the aggregate position.
    single_file_bar: Option<ProgressBar>,
    /// Aggregate byte bar driven exclusively by `AggregateProgress` events
    /// (xet). Never touched by per-file events.
    aggregate_bar: Option<ProgressBar>,
    download_transfer_bar: Option<ProgressBar>,
    file_bars: HashMap<String, ProgressBar>,
    download_queue: VecDeque<(String, u64)>,
    total_files: usize,
    // Upload state
    processing_bar: Option<ProgressBar>,
    transfer_bar: Option<ProgressBar>,
    upload_file_bars: HashMap<String, ProgressBar>,
    upload_queue: VecDeque<(String, u64)>,
    upload_completed_files: HashSet<String>,
    spinner: Option<ProgressBar>,
    upload_bars_active: bool,
    upload_total_files: usize,
}

fn bytes_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg}: {percent}%|{wide_bar:.cyan/blue}| {bytes}/{total_bytes} [{elapsed}<{eta}, {bytes_per_sec}]",
    )
    .expect("hardcoded template")
    .progress_chars("##-")
}

fn aggregate_bytes_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg}: {percent}%|{wide_bar:.cyan/blue}| {bytes}/{total_bytes} [{elapsed}]")
        .expect("hardcoded template")
        .progress_chars("##-")
}

fn files_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg}: {percent}%|{wide_bar:.green/blue}| {pos}/{len} [{elapsed}<{eta}]")
        .expect("hardcoded template")
        .progress_chars("##-")
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} {msg}").expect("hardcoded template")
}

fn truncate_filename(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        return name.to_string();
    }
    let suffix = &name[name.len() - (max_len - 1)..];
    format!("…{suffix}")
}

fn format_rate(bytes_per_sec: Option<f64>) -> String {
    match bytes_per_sec {
        Some(r) if r.is_finite() && r > 0.0 => format!("{}/s", HumanBytes(r as u64)),
        _ => "--".to_string(),
    }
}

/// Hard cap: if the computed ETA is longer than this, show "--" instead.
/// Protects against the "ETA in years" display when the upstream rate
/// dips toward zero between progress events.
const MAX_REASONABLE_ETA_SECS: u64 = 24 * 60 * 60;

fn format_eta(remaining_bytes: u64, bytes_per_sec: Option<f64>) -> String {
    if remaining_bytes == 0 {
        return "0s".to_string();
    }
    let rate = match bytes_per_sec {
        Some(r) if r.is_finite() && r > 0.0 => r,
        _ => return "--".to_string(),
    };
    let secs = (remaining_bytes as f64 / rate).ceil();
    if !secs.is_finite() || secs < 0.0 || secs > MAX_REASONABLE_ETA_SECS as f64 {
        return "--".to_string();
    }
    format!("{}", HumanDuration(std::time::Duration::from_secs(secs as u64)))
}

impl CliProgressHandler {
    pub fn new(multi: MultiProgress) -> Self {
        Self {
            multi,
            state: Mutex::new(ProgressState {
                files_bar: None,
                single_file_bar: None,
                aggregate_bar: None,
                download_transfer_bar: None,
                file_bars: HashMap::new(),
                download_queue: VecDeque::new(),
                total_files: 0,
                processing_bar: None,
                transfer_bar: None,
                upload_file_bars: HashMap::new(),
                upload_queue: VecDeque::new(),
                upload_completed_files: HashSet::new(),
                spinner: None,
                upload_bars_active: false,
                upload_total_files: 0,
            }),
        }
    }

    fn handle_download(&self, event: &DownloadEvent) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match event {
            DownloadEvent::Start {
                total_files,
                total_bytes,
            } => {
                state.total_files = *total_files;
                if *total_files > 1 {
                    let bar = self.multi.add(ProgressBar::new(*total_files as u64));
                    bar.set_style(files_style());
                    bar.set_message(format!("Fetching {} files", total_files));
                    state.files_bar = Some(bar);
                }
                if *total_bytes > 0 && *total_files == 1 {
                    let bar = self.multi.add(ProgressBar::new(*total_bytes));
                    bar.set_style(bytes_style());
                    bar.set_message("Downloading");
                    state.single_file_bar = Some(bar);
                }
            },
            DownloadEvent::Progress { files } => {
                for fp in files {
                    if state.total_files == 1 && state.aggregate_bar.is_some() {
                        continue;
                    }
                    match fp.status {
                        FileStatus::Started => {
                            if state.total_files == 1 && state.single_file_bar.is_none() && fp.total_bytes > 0 {
                                let bar = self.multi.add(ProgressBar::new(fp.total_bytes));
                                bar.set_style(bytes_style());
                                bar.set_message("Downloading");
                                state.single_file_bar = Some(bar);
                            } else if state.file_bars.len() < MAX_VISIBLE_FILE_BARS {
                                let bar = self.multi.add(ProgressBar::new(fp.total_bytes));
                                bar.set_style(bytes_style());
                                bar.set_message(truncate_filename(&fp.filename, 40));
                                state.file_bars.insert(fp.filename.clone(), bar);
                            } else {
                                state.download_queue.push_back((fp.filename.clone(), fp.total_bytes));
                            }
                        },
                        FileStatus::InProgress => {
                            if let Some(bar) = state.file_bars.get(&fp.filename) {
                                bar.set_position(fp.bytes_completed);
                            } else if state.file_bars.len() < MAX_VISIBLE_FILE_BARS {
                                let bar = self.multi.add(ProgressBar::new(fp.total_bytes));
                                bar.set_style(bytes_style());
                                bar.set_message(truncate_filename(&fp.filename, 40));
                                bar.set_position(fp.bytes_completed);
                                state.file_bars.insert(fp.filename.clone(), bar);
                                state.download_queue.retain(|(n, _)| n != &fp.filename);
                            } else if let Some(ref bar) = state.single_file_bar {
                                bar.set_position(fp.bytes_completed);
                            }
                        },
                        FileStatus::Complete => {
                            if let Some(bar) = state.file_bars.remove(&fp.filename) {
                                bar.finish_and_clear();
                                self.multi.remove(&bar);
                            }
                            state.download_queue.retain(|(n, _)| n != &fp.filename);
                            if let Some(ref bar) = state.single_file_bar {
                                bar.set_position(fp.bytes_completed);
                            }
                            if let Some(ref bar) = state.files_bar {
                                bar.inc(1);
                            }
                            while state.file_bars.len() < MAX_VISIBLE_FILE_BARS {
                                if let Some((name, total)) = state.download_queue.pop_front() {
                                    let bar = self.multi.add(ProgressBar::new(total));
                                    bar.set_style(bytes_style());
                                    bar.set_message(truncate_filename(&name, 40));
                                    state.file_bars.insert(name, bar);
                                } else {
                                    break;
                                }
                            }
                        },
                    }
                }
            },
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                transfer_bytes_completed,
                transfer_bytes,
                transfer_bytes_per_sec,
                ..
            } => {
                if state.aggregate_bar.is_none() {
                    if state.total_files == 1 {
                        if let Some(bar) = state.single_file_bar.take() {
                            bar.finish_and_clear();
                            self.multi.remove(&bar);
                        }
                        for (_, bar) in state.file_bars.drain() {
                            bar.finish_and_clear();
                            self.multi.remove(&bar);
                        }
                        state.download_queue.clear();
                    }
                    let bar = self.multi.add(ProgressBar::new(*total_bytes));
                    bar.set_style(bytes_style());
                    bar.set_message("Reconstructing");
                    state.aggregate_bar = Some(bar);
                }
                if let Some(ref bar) = state.aggregate_bar {
                    bar.set_length(*total_bytes);
                    bar.set_position(*bytes_completed);
                }
                if state.download_transfer_bar.is_none() {
                    let bar = self.multi.add(ProgressBar::new(*transfer_bytes));
                    // Xet discovers transfer totals incrementally; do not derive a file ETA from them.
                    bar.set_style(aggregate_bytes_style());
                    state.download_transfer_bar = Some(bar);
                }
                if let Some(ref bar) = state.download_transfer_bar {
                    bar.set_length(*transfer_bytes);
                    bar.set_position(*transfer_bytes_completed);
                    bar.set_message(format!("Downloading ({})", format_rate(*transfer_bytes_per_sec)));
                }
            },
            DownloadEvent::Complete => {
                if let Some(ref bar) = state.files_bar {
                    bar.finish_and_clear();
                }
                if let Some(ref bar) = state.single_file_bar {
                    bar.finish_and_clear();
                }
                if let Some(ref bar) = state.aggregate_bar {
                    bar.finish_and_clear();
                }
                if let Some(ref bar) = state.download_transfer_bar {
                    bar.finish_and_clear();
                }
                for (_, bar) in state.file_bars.drain() {
                    bar.finish_and_clear();
                }
                state.download_queue.clear();
            },
        }
    }

    fn handle_upload(&self, event: &UploadEvent) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match event {
            UploadEvent::Start {
                total_files,
                total_bytes: _,
            } => {
                state.upload_total_files = *total_files;
                if let Some(spinner) = state.spinner.take() {
                    spinner.finish_and_clear();
                    self.multi.remove(&spinner);
                }
                let bar = self.multi.add(ProgressBar::new_spinner());
                bar.set_style(spinner_style());
                bar.set_message("Preparing upload...");
                bar.enable_steady_tick(std::time::Duration::from_millis(100));
                state.spinner = Some(bar);
            },
            UploadEvent::Progress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
                transfer_bytes_completed,
                transfer_bytes,
                transfer_bytes_per_sec,
                files,
            } => {
                if !state.upload_bars_active {
                    if let Some(spinner) = state.spinner.take() {
                        spinner.finish_and_clear();
                        self.multi.remove(&spinner);
                    }
                    let pbar = self.multi.add(ProgressBar::new(0));
                    pbar.set_style(aggregate_bytes_style());
                    pbar.set_message(format!("Processing Files (0 / {})", state.upload_total_files));
                    state.processing_bar = Some(pbar);

                    let tbar = self.multi.add(ProgressBar::new(0));
                    tbar.set_style(aggregate_bytes_style());
                    tbar.set_message("New Data Upload");
                    state.transfer_bar = Some(tbar);
                    state.upload_bars_active = true;
                }

                let completed_count = state.upload_completed_files.len();
                let total_count = state.upload_total_files;

                if let Some(ref bar) = state.processing_bar {
                    bar.set_length(*total_bytes);
                    bar.set_position(*bytes_completed);
                    let remaining = total_bytes.saturating_sub(*bytes_completed);
                    bar.set_message(format!(
                        "Processing Files ({} / {}) • {} • ETA {}",
                        completed_count,
                        total_count,
                        format_rate(*bytes_per_sec),
                        format_eta(remaining, *bytes_per_sec),
                    ));
                }

                if let Some(ref bar) = state.transfer_bar {
                    // No file count here: transfer bytes are post-dedup,
                    // so they don't map 1:1 to files.
                    bar.set_length(*transfer_bytes);
                    bar.set_position(*transfer_bytes_completed);
                    let remaining = transfer_bytes.saturating_sub(*transfer_bytes_completed);
                    bar.set_message(format!(
                        "New Data Upload • {} • ETA {}",
                        format_rate(*transfer_bytes_per_sec),
                        format_eta(remaining, *transfer_bytes_per_sec),
                    ));
                }

                for fp in files {
                    self.process_upload_file_progress(&mut state, fp);
                }
            },
            UploadEvent::Committing => {
                self.cleanup_upload_bars(&mut state);
                if let Some(spinner) = state.spinner.take() {
                    spinner.finish_and_clear();
                    self.multi.remove(&spinner);
                }
                let bar = self.multi.add(ProgressBar::new_spinner());
                bar.set_style(spinner_style());
                bar.set_message("Creating commit...");
                bar.enable_steady_tick(std::time::Duration::from_millis(100));
                state.spinner = Some(bar);
            },
            UploadEvent::Complete => {
                self.cleanup_upload_bars(&mut state);
                if let Some(spinner) = state.spinner.take() {
                    spinner.finish_and_clear();
                    self.multi.remove(&spinner);
                }
            },
        }
    }

    fn process_upload_file_progress(&self, state: &mut ProgressState, fp: &FileProgress) {
        if state.upload_completed_files.contains(&fp.filename) {
            return;
        }
        match fp.status {
            FileStatus::Started => {
                if !state.upload_file_bars.contains_key(&fp.filename) {
                    if state.upload_file_bars.len() < MAX_VISIBLE_UPLOAD_BARS {
                        let bar = self.multi.add(ProgressBar::new(fp.total_bytes));
                        bar.set_style(bytes_style());
                        bar.set_message(truncate_filename(&fp.filename, 40));
                        state.upload_file_bars.insert(fp.filename.clone(), bar);
                    } else {
                        state.upload_queue.push_back((fp.filename.clone(), fp.total_bytes));
                    }
                }
            },
            FileStatus::InProgress => {
                if let Some(bar) = state.upload_file_bars.get(&fp.filename) {
                    bar.set_position(fp.bytes_completed);
                } else if state.upload_file_bars.len() < MAX_VISIBLE_UPLOAD_BARS {
                    let bar = self.multi.add(ProgressBar::new(fp.total_bytes));
                    bar.set_style(bytes_style());
                    bar.set_message(truncate_filename(&fp.filename, 40));
                    bar.set_position(fp.bytes_completed);
                    state.upload_file_bars.insert(fp.filename.clone(), bar);
                    state.upload_queue.retain(|(n, _)| n != &fp.filename);
                }
            },
            FileStatus::Complete => {
                if state.upload_completed_files.insert(fp.filename.clone()) {
                    if let Some(bar) = state.upload_file_bars.remove(&fp.filename) {
                        bar.finish_and_clear();
                        self.multi.remove(&bar);
                    }
                    state.upload_queue.retain(|(n, _)| n != &fp.filename);
                    if let Some(ref bar) = state.files_bar {
                        bar.inc(1);
                    }
                    while state.upload_file_bars.len() < MAX_VISIBLE_UPLOAD_BARS {
                        if let Some((name, total)) = state.upload_queue.pop_front() {
                            let bar = self.multi.add(ProgressBar::new(total));
                            bar.set_style(bytes_style());
                            bar.set_message(truncate_filename(&name, 40));
                            state.upload_file_bars.insert(name, bar);
                        } else {
                            break;
                        }
                    }
                }
            },
        }
    }

    fn cleanup_upload_bars(&self, state: &mut ProgressState) {
        for (_, bar) in state.upload_file_bars.drain() {
            bar.finish_and_clear();
            self.multi.remove(&bar);
        }
        state.upload_queue.clear();
        state.upload_completed_files.clear();
        if let Some(bar) = state.processing_bar.take() {
            bar.finish_and_clear();
            self.multi.remove(&bar);
        }
        if let Some(bar) = state.transfer_bar.take() {
            bar.finish_and_clear();
            self.multi.remove(&bar);
        }
        state.upload_bars_active = false;
    }
}

impl ProgressHandler for CliProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(dl) => self.handle_download(dl),
            ProgressEvent::Upload(ul) => self.handle_upload(ul),
        }
    }
}

pub fn progress_disabled_by_env() -> bool {
    std::env::var("HF_HUB_DISABLE_PROGRESS_BARS").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[cfg(test)]
mod tests {
    use hf_hub::progress::{DownloadEvent, FileProgress, FileStatus};
    use indicatif::{MultiProgress, ProgressDrawTarget};

    use super::{CliProgressHandler, format_eta, format_rate};

    #[test]
    fn network_progress_does_not_advance_reconstruction_or_finish_download() {
        let handler = CliProgressHandler::new(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()));
        handler.handle_download(&DownloadEvent::Start {
            total_files: 1,
            total_bytes: 1_000,
        });
        handler.handle_download(&DownloadEvent::AggregateProgress {
            bytes_completed: 0,
            total_bytes: 1_000,
            bytes_per_sec: None,
            transfer_bytes_completed: 500,
            transfer_bytes: 500,
            transfer_bytes_per_sec: Some(250.0),
        });
        {
            let state = handler.state.lock().unwrap();
            let reconstruction = state.aggregate_bar.as_ref().unwrap();
            let transfer = state.download_transfer_bar.as_ref().unwrap();
            assert_eq!(reconstruction.position(), 0);
            assert_eq!(transfer.position(), 500);
            assert!(state.single_file_bar.is_none());
            assert!(transfer.message().contains(&format_rate(Some(250.0))));
            assert!(!reconstruction.is_finished());
            assert!(!transfer.is_finished());
        }
        handler.handle_download(&DownloadEvent::Complete);
        let state = handler.state.lock().unwrap();
        assert!(state.aggregate_bar.as_ref().unwrap().is_finished());
        assert!(state.download_transfer_bar.as_ref().unwrap().is_finished());
    }

    #[test]
    fn http_file_progress_does_not_create_xet_network_bar() {
        let handler = CliProgressHandler::new(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()));
        handler.handle_download(&DownloadEvent::Start {
            total_files: 1,
            total_bytes: 1_000,
        });
        handler.handle_download(&DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "config.json".to_string(),
                bytes_completed: 1_000,
                total_bytes: 1_000,
                status: FileStatus::Complete,
            }],
        });
        assert!(handler.state.lock().unwrap().download_transfer_bar.is_none());
    }

    #[test]
    fn format_rate_returns_placeholder_for_none() {
        assert_eq!(format_rate(None), "--");
    }

    #[test]
    fn format_rate_returns_placeholder_for_zero() {
        assert_eq!(format_rate(Some(0.0)), "--");
    }

    #[test]
    fn format_rate_returns_placeholder_for_negative() {
        assert_eq!(format_rate(Some(-1.0)), "--");
    }

    #[test]
    fn format_rate_returns_placeholder_for_nan() {
        assert_eq!(format_rate(Some(f64::NAN)), "--");
    }

    #[test]
    fn format_rate_formats_mb_per_sec() {
        let s = format_rate(Some(42_100_000.0));
        assert!(s.ends_with("/s"), "expected trailing /s, got {s}");
        assert!(s.contains("MB") || s.contains("MiB"), "expected MB unit, got {s}");
    }

    #[test]
    fn format_eta_returns_placeholder_for_no_rate() {
        assert_eq!(format_eta(1_000, None), "--");
    }

    #[test]
    fn format_eta_returns_placeholder_for_zero_rate() {
        assert_eq!(format_eta(1_000, Some(0.0)), "--");
    }

    #[test]
    fn format_eta_returns_zero_when_nothing_remaining() {
        assert_eq!(format_eta(0, Some(1_000_000.0)), "0s");
    }

    #[test]
    fn format_eta_formats_duration() {
        assert_eq!(format_eta(1_000_000, Some(1_000_000.0)), "1 second");
    }

    #[test]
    fn format_eta_caps_absurd_values() {
        let s = format_eta(1_000_000_000_000_000, Some(1.0));
        assert_eq!(s, "--");
    }
}

/// An `io::Write` adapter that routes output through `MultiProgress::println()`,
/// ensuring log lines appear above progress bars without visual corruption.
#[derive(Clone)]
pub struct MultiProgressWriter {
    multi: MultiProgress,
    buf: Vec<u8>,
}

impl MultiProgressWriter {
    pub fn new(multi: MultiProgress) -> Self {
        Self { multi, buf: Vec::new() }
    }
}

impl Write for MultiProgressWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.buf[..pos]).into_owned();
            self.multi.println(&line).map_err(std::io::Error::other)?;
            self.buf.drain(..=pos);
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.buf.is_empty() {
            let line = String::from_utf8_lossy(&self.buf).into_owned();
            self.multi.println(&line).map_err(std::io::Error::other)?;
            self.buf.clear();
        }
        Ok(())
    }
}
