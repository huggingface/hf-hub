use std::io::Write;

use indicatif::MultiProgress;

pub fn progress_disabled_by_env() -> bool {
    std::env::var("HF_HUB_DISABLE_PROGRESS_BARS").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
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
