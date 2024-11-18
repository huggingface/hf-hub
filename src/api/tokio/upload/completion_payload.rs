use http::HeaderMap;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct CompletionPayload {
    oid: String,
    parts: Vec<PayloadPart>,
}

#[derive(Debug, Serialize)]
struct PayloadPart {
    #[serde(rename = "partNumber")]
    part_number: u32,
    etag: String,
}

pub fn get_completion_payload(response_headers: &[HeaderMap], sha256: &[u8]) -> CompletionPayload {
    let parts: Vec<PayloadPart> = response_headers
        .iter()
        .enumerate()
        .map(|(part_number, headers)| {
            let etag = headers
                .get("etag")
                .and_then(|h| h.to_str().ok())
                .filter(|&s| !s.is_empty())
                .ok_or_else(|| format!("Invalid etag returned for part {}", part_number + 1))
                .unwrap(); // You might want to handle this error differently

            PayloadPart {
                part_number: (part_number + 1) as u32,
                etag: etag.to_string(),
            }
        })
        .collect();

    CompletionPayload {
        oid: hex::encode(sha256),
        parts,
    }
}
