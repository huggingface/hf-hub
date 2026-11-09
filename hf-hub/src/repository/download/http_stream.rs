use std::io;
use std::ops::Range;
use std::time::Duration;

use futures::StreamExt;
use reqwest::header::{
    CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_RANGE, ETAG, HeaderName, HeaderValue, IF_RANGE, RANGE,
};
use reqwest::{RequestBuilder, Response, StatusCode, Url};

use super::HFByteStream;
use crate::error::HFError;
use crate::retry;

struct ResumeRange {
    url: Url,
    etag: HeaderValue,
    remaining: Range<u64>,
    total: u64,
}

impl ResumeRange {
    fn new(response: &Response, requested: Option<&Range<u64>>) -> Option<Self> {
        let length = identity_length(response)?;
        let (remaining, total) = match (response.status(), requested) {
            (StatusCode::OK, None) => (0..length, length),
            (StatusCode::PARTIAL_CONTENT, Some(requested)) => {
                let (range, total) = content_range(response)?;
                if range != (requested.start..requested.end.min(total)) || length != range.end - range.start {
                    return None;
                }
                (range, total)
            },
            _ => return None,
        };
        Some(Self {
            url: response.url().clone(),
            etag: strong_etag(response)?.clone(),
            remaining,
            total,
        })
    }

    fn matches(&self, response: &Response) -> bool {
        // A strong validator identifies a representation of one resource.
        response.status() == StatusCode::PARTIAL_CONTENT
            && response.url() == &self.url
            && strong_etag(response) == Some(&self.etag)
            && identity_length(response) == Some(self.remaining.end - self.remaining.start)
            && content_range(response).is_some_and(|(range, total)| range == self.remaining && total == self.total)
    }
}

fn identity_length(response: &Response) -> Option<u64> {
    let headers = response.headers();
    if !headers
        .get_all(CONTENT_ENCODING)
        .iter()
        .all(|value| value.as_bytes().eq_ignore_ascii_case(b"identity"))
    {
        return None;
    }
    // Automatic decompression removes Content-Length. Decoded byte offsets
    // cannot be used as offsets in a subsequent HTTP range request.
    single_header(response, CONTENT_LENGTH)?.to_str().ok()?.parse().ok()
}

fn single_header(response: &Response, name: HeaderName) -> Option<&HeaderValue> {
    let all = response.headers().get_all(name);
    let mut values = all.iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn strong_etag(response: &Response) -> Option<&HeaderValue> {
    let value = single_header(response, ETAG)?;
    let [b'"', tag @ .., b'"'] = value.as_bytes() else {
        return None;
    };
    tag.iter()
        .all(|&byte| byte == b'!' || (b'#'..=b'~').contains(&byte) || byte >= 0x80)
        .then_some(value)
}

fn content_range(response: &Response) -> Option<(Range<u64>, u64)> {
    let value = single_header(response, CONTENT_RANGE)?.to_str().ok()?;
    let (range, total) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (start, last) = range.split_once('-')?;
    let (start, last, total): (u64, u64, u64) = (start.parse().ok()?, last.parse().ok()?, total.parse().ok()?);
    (start <= last && last < total).then(|| (start..last + 1, total))
}

pub(super) fn resume(
    response: Response,
    requested: Option<&Range<u64>>,
    request: impl Fn() -> RequestBuilder + Send + Sync + 'static,
    max_attempts: usize,
    delays: impl Iterator<Item = Duration> + Send + 'static,
) -> HFByteStream {
    let range = ResumeRange::new(&response, requested);
    let stream = Box::pin(response.bytes_stream());
    let Some(range) = range else {
        return Box::new(stream.map(|r| r.map_err(HFError::from)));
    };

    let stream = futures::stream::try_unfold(
        (stream, range, request, delays),
        move |(mut stream, mut range, request, mut delays)| async move {
            loop {
                let error = match stream.next().await {
                    Some(Ok(chunk)) => {
                        let length = chunk.len() as u64;
                        if length > range.remaining.end - range.remaining.start {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "HTTP body exceeded its advertised length",
                            )
                            .into());
                        }
                        range.remaining.start += length;
                        return Ok(Some((chunk, (stream, range, request, delays))));
                    },
                    Some(Err(error)) => error,
                    None if range.remaining.is_empty() => return Ok(None),
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "HTTP body ended before its advertised length",
                        )
                        .into());
                    },
                };

                if range.remaining.is_empty() || !retry::is_transient_body_error(&error) {
                    return Err(error.into());
                }
                let Some(delay) = delays.next() else {
                    return Err(error.into());
                };
                drop(stream);
                tokio::time::sleep(delay).await;
                let header = format!("bytes={}-{}", range.remaining.start, range.remaining.end - 1);
                let response = retry::retry_with_delays(max_attempts, &mut delays, || {
                    request().header(RANGE, &header).header(IF_RANGE, &range.etag).send()
                })
                .await?;
                if !range.matches(&response) {
                    tracing::debug!(status = %response.status(), "cannot safely resume HTTP download");
                    return Err(error.into());
                }
                stream = Box::pin(response.bytes_stream());
            }
        },
    );
    Box::new(Box::pin(stream))
}
