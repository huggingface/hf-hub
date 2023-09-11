use std::io::{Read, BufRead};

pub struct ResumableReader<'a, R> {
    inner: R,
    last_written_byte: &'a mut usize,
}

impl<'a, R: Read> ResumableReader<'a, R> {
    pub fn new(inner: R, last_written_byte: &'a mut usize) -> Self {
        ResumableReader {
            inner,
            last_written_byte,
        }
    }
}

impl<'a, R: Read> Read for ResumableReader<'a, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let bytes_read = self.inner.read(buf)?;
        *self.last_written_byte += bytes_read;
        Ok(bytes_read)
    }

    fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        let bytes_read = self.inner.read_vectored(bufs)?;
        *self.last_written_byte += bytes_read;
        Ok(bytes_read)
    }

    fn read_to_string(&mut self, buf: &mut String) -> std::io::Result<usize> {
        let bytes_read = self.inner.read_to_string(buf)?;
        *self.last_written_byte += bytes_read;
        Ok(bytes_read)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.inner.read_exact(buf)?;
        *self.last_written_byte += buf.len();
        Ok(())
    }
}

impl<'a, R: BufRead> BufRead for ResumableReader<'a, R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt);
        *self.last_written_byte += amt;
    }
}
