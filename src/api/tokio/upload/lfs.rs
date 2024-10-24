use http::HeaderMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::ErrorKind};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::{
    api::tokio::{ApiError, ApiRepo, ReqwestBadResponse},
    RepoType,
};

use super::{
    commit_api::{CommitOperationAdd, UploadInfo, UploadSource},
    commit_info::fix_hf_endpoint_in_url,
    completion_payload::get_completion_payload,
};

#[derive(Debug, Deserialize)]
pub struct BatchInfo {
    pub objects: Vec<BatchObject>,
}

#[derive(Debug, Deserialize)]
pub struct BatchObject {
    pub oid: String,
    pub size: u64,
    #[serde(default)]
    pub error: Option<BatchError>,
    pub actions: Option<BatchActions>,
}

#[derive(Debug, Deserialize)]
pub struct BatchError {
    pub code: i32,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct BatchActions {
    pub upload: Option<LfsAction>,
    pub verify: Option<LfsAction>,
}

#[derive(Debug, Deserialize)]
pub struct LfsAction {
    pub href: String,
    pub header: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
struct BatchRequest {
    operation: String,
    transfers: Vec<String>,
    objects: Vec<BatchRequestObject>,
    hash_algo: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    r#ref: Option<BatchRequestRef>,
}

#[derive(Debug, Serialize)]
struct BatchRequestObject {
    oid: String,
    size: u64,
}

impl From<BatchObject> for BatchRequestObject {
    fn from(value: BatchObject) -> Self {
        Self {
            oid: value.oid,
            size: value.size,
        }
    }
}

#[derive(Debug, Serialize)]
struct BatchRequestRef {
    name: String,
}

fn lfs_endpoint(repo_type: RepoType, repo_id: &str) -> String {
    let prefix = match repo_type {
        RepoType::Model => "",
        RepoType::Dataset => "datasets/",
        RepoType::Space => "spaces/",
    };
    format!("{}{}.git", prefix, repo_id)
}

impl ApiRepo {
    /// Requests the LFS batch endpoint to retrieve upload instructions
    ///
    /// Learn more: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md
    pub(crate) async fn post_lfs_batch_info(
        &self,
        additions: &[CommitOperationAdd],
    ) -> Result<Vec<BatchObject>, ApiError> {
        let batch_url = format!(
            "{}/{}/info/lfs/objects/batch",
            self.api.endpoint,
            lfs_endpoint(self.repo.repo_type, &self.repo.repo_id)
        );

        let objects: Vec<BatchRequestObject> = additions
            .iter()
            .map(|op| BatchRequestObject {
                oid: hex::encode(&op.upload_info.sha256),
                size: op.upload_info.size,
            })
            .collect();

        let payload = BatchRequest {
            operation: "upload".to_string(),
            transfers: vec!["basic".to_string(), "multipart".to_string()],
            objects,
            hash_algo: "sha256".to_string(),
            r#ref: None, // Add revision handling if needed
        };

        let headers = make_lfs_headers();

        let response = self
            .api
            .client
            .post(&batch_url)
            .headers(headers)
            .json(&payload)
            .send()
            .await?
            .maybe_err()
            .await?;

        let batch_info: BatchInfo = response.json().await?;
        Ok(batch_info.objects)
    }
}

fn make_lfs_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("Accept", "application/vnd.git-lfs+json".parse().unwrap());
    headers.insert(
        "Content-Type",
        "application/vnd.git-lfs+json".parse().unwrap(),
    );
    headers
}

fn get_sorted_parts_urls(
    headers: &HashMap<String, String>,
    upload_info: &UploadInfo,
    chunk_size: u64,
) -> Result<Vec<String>, ApiError> {
    let mut part_urls: Vec<(u32, String)> = headers
        .iter()
        .filter_map(|(key, value)| {
            if let Ok(part_num) = key.parse::<u32>() {
                Some((part_num, value.clone()))
            } else {
                None
            }
        })
        .collect();

    part_urls.sort_by_key(|&(num, _)| num);
    let sorted_urls: Vec<String> = part_urls.into_iter().map(|(_, url)| url).collect();

    let expected_parts = ((upload_info.size as f64) / (chunk_size as f64)).ceil() as usize;
    log::trace!("chunk size is {chunk_size} and the whole file is {size}, and {size}/{chunk_size} === {result}", size = upload_info.size, result=expected_parts);
    if sorted_urls.len() != expected_parts {
        return Err(ApiError::InvalidResponse(
            "Invalid server response to upload large LFS file".into(),
        ));
    }

    Ok(sorted_urls)
}

async fn upload_part(
    client: Client,
    part_upload_url: &str,
    data: Vec<u8>,
) -> Result<reqwest::Response, ApiError> {
    let l = data.len();
    log::trace!("uploading part ({} bytes)", l);
    let response = client
        .put(part_upload_url)
        .body(data)
        .send()
        .await?
        .maybe_err()
        .await?;

    log::trace!("uploaded ({} bytes)", l);
    Ok(response)
}

async fn upload_multi_part(
    hf_client: Client,
    s3_client: Client,
    operation: CommitOperationAdd,
    headers: &HashMap<String, String>,
    chunk_size: u64,
    upload_url: &str,
) -> Result<CommitOperationAdd, ApiError> {
    // 1. Get upload URLs for each part
    log::trace!("getting upload urls..");
    let sorted_parts_urls = get_sorted_parts_urls(headers, &operation.upload_info, chunk_size)?;
    log::trace!("got upload URLs: {sorted_parts_urls:?}.");

    // 2. Upload parts
    log::trace!("uploading parts...");
    let sha256 = operation.upload_info.sha256.clone();
    let (operation, response_headers) =
        upload_parts_iteratively(s3_client, operation, &sorted_parts_urls, chunk_size).await?;
    log::trace!("parts uploaded.");

    // 3. Send completion request
    let completion_payload = get_completion_payload(&response_headers, &sha256);
    log::trace!("sending completion request: {completion_payload:?}");

    let headers = make_lfs_headers();

    let response = hf_client
        .post(upload_url)
        .headers(headers)
        .json(&completion_payload)
        .send()
        .await?
        .maybe_err()
        .await?;

    log::trace!("completion response: {:?}", response.text().await?);

    Ok(operation)
}

async fn upload_parts_iteratively(
    client: Client,
    mut operation: CommitOperationAdd,
    sorted_parts_urls: &[String],
    chunk_size: u64,
) -> Result<(CommitOperationAdd, Vec<HeaderMap>), ApiError> {
    let mut response_headers = Vec::new();

    match &operation.source {
        UploadSource::File(path) => {
            let file = tokio::fs::File::open(path).await?;
            let mut reader = tokio::io::BufReader::new(file);

            for (part_idx, part_upload_url) in sorted_parts_urls.iter().enumerate() {
                let mut buffer = vec![0u8; chunk_size as usize];
                let start_pos = part_idx as u64 * chunk_size;
                log::trace!("uploading path {path:?} chunk {part_idx}, start_pos {start_pos}");
                reader.seek(std::io::SeekFrom::Start(start_pos)).await?;

                // read either until the chunk is done or we hit EoF
                let bytes_read = {
                    let mut bytes_read = 0;
                    while bytes_read < chunk_size as usize {
                        match reader.read(&mut buffer[bytes_read..]).await? {
                            0 => break, // EOF reached
                            n => bytes_read += n,
                        }
                    }
                    bytes_read
                };
                buffer.truncate(bytes_read);

                let response = upload_part(client.clone(), part_upload_url, buffer).await?;
                response_headers.push(response.headers().clone());
            }
        }
        UploadSource::Bytes(bytes) => {
            for (part_idx, part_upload_url) in sorted_parts_urls.iter().enumerate() {
                let start = (part_idx as u64 * chunk_size) as usize;
                let end = ((part_idx + 1) as u64 * chunk_size) as usize;
                let chunk = bytes[start..std::cmp::min(end, bytes.len())].to_vec();

                let response = upload_part(client.clone(), part_upload_url, chunk).await?;
                response_headers.push(response.headers().clone());
            }
        }
        UploadSource::Emptied => {
            return Err(ApiError::IoError(std::io::Error::new(
                ErrorKind::NotFound,
                "File has already been emptied!",
            )));
        }
    }

    operation.source = UploadSource::Emptied;

    Ok((operation, response_headers))
}

async fn upload_single_part(
    s3_client: Client,
    mut operation: CommitOperationAdd,
    upload_url: &str,
) -> Result<CommitOperationAdd, ApiError> {
    let body = match &operation.source {
        UploadSource::File(path) => tokio::fs::read(path).await?,
        UploadSource::Bytes(bytes) => bytes.clone(),
        UploadSource::Emptied => {
            return Err(ApiError::IoError(std::io::Error::new(
                ErrorKind::NotFound,
                "File has already been emptied!".to_string(),
            )));
        }
    };

    let _ = s3_client
        .put(upload_url)
        .body(body)
        .send()
        .await?
        .maybe_err()
        .await?;

    operation.source = UploadSource::Emptied;

    Ok(operation)
}

pub(crate) async fn lfs_upload(
    hf_client: Client,
    s3_client: Client,
    operation: CommitOperationAdd,
    batch_action: BatchObject,
    endpoint: String,
) -> Result<CommitOperationAdd, ApiError> {
    // Skip if already uploaded
    if batch_action.actions.is_none() {
        log::debug!(
            "Content of file {} is already present upstream - skipping upload",
            operation.path_in_repo
        );
        return Ok(operation);
    }
    let path = operation.path_in_repo.clone();

    let actions = batch_action.actions.as_ref().unwrap();
    let upload_action = actions.upload.as_ref().ok_or_else(|| {
        ApiError::InvalidResponse("Missing upload action in LFS batch response".into())
    })?;

    let multipart = upload_action.header.as_ref().and_then(|h| {
        h.get("chunk_size")
            .and_then(|size| size.parse::<u64>().ok().map(|s| (h, s)))
    });
    let finished_upload = if let Some((header, chunk_size)) = multipart {
        const ONE_MB: u64 = 1024 * 1024;
        const MIN_UPLOAD_SIZE: u64 = 5 * ONE_MB;
        if chunk_size < MIN_UPLOAD_SIZE {
            return Err(ApiError::InvalidResponse(format!("API gave us a chunk size of {chunk_size}, but the smallest allowed chunk size for AWS multipart uploads is {MIN_UPLOAD_SIZE}")));
        } else {
            log::trace!("chunk size for {path}: {chunk_size}")
        }
        // Handle multipart upload if chunk_size is present
        log::debug!("starting multipart upload for {path}");
        upload_multi_part(
            hf_client.clone(),
            s3_client.clone(),
            operation,
            header,
            chunk_size,
            &upload_action.href,
        )
        .await
    } else {
        // Fall back to single-part upload
        log::debug!("starting single-part upload for {}", path);
        upload_single_part(s3_client.clone(), operation, &upload_action.href).await
    }?;

    if let Some(verify) = &actions.verify {
        log::debug!("running verify for {}", path);
        let verify_url = fix_hf_endpoint_in_url(&verify.href, &endpoint);
        let verify_body: BatchRequestObject = batch_action.into();
        let res = hf_client.post(verify_url).json(&verify_body).send().await?;
        log::debug!("verify result: {}", res.text().await?)
    }

    log::debug!("{}: Upload successful", path);

    Ok(finished_upload)
}
