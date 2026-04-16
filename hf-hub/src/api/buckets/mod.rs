mod sync;

use futures::Stream;
use url::Url;

use crate::bucket::HFBucket;
use crate::client::HFClient;
use crate::error::{HFError, NotFoundContext, Result};
use crate::types::progress::{self, DownloadEvent, Progress, ProgressEvent, UploadEvent};
use crate::types::{
    BatchBucketFilesParams, BucketFileMetadata, BucketInfo, BucketSyncParams, BucketTreeEntry, BucketUrl,
    CreateBucketParams, ListBucketTreeParams, SyncPlan,
};

const BUCKET_BATCH_CHUNK_SIZE: usize = 1000;
const BUCKET_PATHS_INFO_BATCH_SIZE: usize = 1000;

impl HFClient {
    /// Create a new bucket on the Hub.
    ///
    /// Endpoint: `POST /api/buckets/{namespace}/{name}`
    pub async fn create_bucket(&self, params: &CreateBucketParams) -> Result<BucketUrl> {
        let url = format!("{}/api/buckets/{}/{}", self.endpoint(), params.namespace, params.name);

        let mut body = serde_json::json!({});
        if params.private {
            body["private"] = serde_json::Value::Bool(true);
        }
        if let Some(ref rg) = params.resource_group_id {
            body["resourceGroupId"] = serde_json::Value::String(rg.clone());
        }

        let response = self
            .http_client()
            .post(&url)
            .headers(self.auth_headers())
            .json(&body)
            .send()
            .await?;

        let bucket_id = format!("{}/{}", params.namespace, params.name);

        if response.status().as_u16() == 409 && params.exist_ok {
            return Ok(BucketUrl {
                url: format!("{}/buckets/{}", self.endpoint(), bucket_id),
            });
        }

        let response = self
            .check_response(response, Some(&bucket_id), NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Delete a bucket from the Hub.
    ///
    /// Endpoint: `DELETE /api/buckets/{bucket_id}`
    pub async fn delete_bucket(&self, bucket_id: &str, missing_ok: bool) -> Result<()> {
        let url = self.bucket_api_url(bucket_id);

        let response = self.http_client().delete(&url).headers(self.auth_headers()).send().await?;

        if response.status().as_u16() == 404 && missing_ok {
            return Ok(());
        }

        self.check_response(response, Some(bucket_id), NotFoundContext::Bucket).await?;
        Ok(())
    }

    /// List buckets in a namespace.
    ///
    /// Endpoint: `GET /api/buckets/{namespace}` (paginated)
    pub fn list_buckets(&self, namespace: &str) -> Result<impl Stream<Item = Result<BucketInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/buckets/{}", self.endpoint(), namespace))?;
        Ok(self.paginate(url, vec![], None))
    }

    /// Move (rename) a bucket.
    ///
    /// Endpoint: `POST /api/repos/move` with `type: "bucket"`
    pub async fn move_bucket(&self, from_id: &str, to_id: &str) -> Result<()> {
        let url = format!("{}/api/repos/move", self.endpoint());
        let body = serde_json::json!({
            "fromRepo": from_id,
            "toRepo": to_id,
            "type": "bucket",
        });

        let response = self
            .http_client()
            .post(&url)
            .headers(self.auth_headers())
            .json(&body)
            .send()
            .await?;

        self.check_response(response, None, NotFoundContext::Generic).await?;
        Ok(())
    }
}

impl HFBucket {
    /// Get metadata about this bucket.
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}`
    pub async fn info(&self) -> Result<BucketInfo> {
        let bucket_id = self.bucket_id();
        let url = self.hf_client.bucket_api_url(&bucket_id);

        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;

        let response = self
            .hf_client
            .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
            .await?;
        Ok(response.json().await?)
    }

    /// List files and directories in this bucket.
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}/tree[/{prefix}]` (paginated)
    pub fn list_tree(&self, params: &ListBucketTreeParams) -> Result<impl Stream<Item = Result<BucketTreeEntry>> + '_> {
        let bucket_id = self.bucket_id();
        let mut url_str = format!("{}/api/buckets/{}/tree", self.hf_client.endpoint(), bucket_id);
        if let Some(ref prefix) = params.prefix {
            url_str = format!("{}/{}", url_str, prefix);
        }

        let url = Url::parse(&url_str)?;

        let mut query = vec![];
        if params.recursive == Some(true) {
            query.push(("recursive".to_string(), "true".to_string()));
        }

        Ok(self.hf_client.paginate(url, query, None))
    }

    /// Get info about specific paths in this bucket.
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/paths-info`
    pub async fn get_paths_info(&self, paths: &[String]) -> Result<Vec<BucketTreeEntry>> {
        let bucket_id = self.bucket_id();
        let url = format!("{}/api/buckets/{}/paths-info", self.hf_client.endpoint(), bucket_id);

        let mut all_entries = Vec::new();
        for chunk in paths.chunks(BUCKET_PATHS_INFO_BATCH_SIZE) {
            let body = serde_json::json!({ "paths": chunk });

            let response = self
                .hf_client
                .http_client()
                .post(&url)
                .headers(self.hf_client.auth_headers())
                .json(&body)
                .send()
                .await?;

            let response = self
                .hf_client
                .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
                .await?;
            let entries: Vec<BucketTreeEntry> = response.json().await?;
            all_entries.extend(entries);
        }

        Ok(all_entries)
    }

    /// Get metadata for a single file in this bucket via a HEAD request.
    ///
    /// Endpoint: `HEAD /buckets/{bucket_id}/resolve/{path}`
    pub async fn get_file_metadata(&self, remote_path: &str) -> Result<BucketFileMetadata> {
        let bucket_id = self.bucket_id();
        let url = format!("{}/buckets/{}/resolve/{}", self.hf_client.endpoint(), bucket_id, remote_path);

        let response = self
            .hf_client
            .no_redirect_client()
            .head(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;

        let response = self
            .hf_client
            .check_response(
                response,
                Some(&bucket_id),
                NotFoundContext::Entry {
                    path: remote_path.to_string(),
                },
            )
            .await?;

        let size = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        let xet_hash = response
            .headers()
            .get("x-xet-hash")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        Ok(BucketFileMetadata { size, xet_hash })
    }

    /// Execute batch file operations (add, delete, copy) on this bucket.
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/batch` (NDJSON)
    pub async fn batch(&self, params: &BatchBucketFilesParams) -> Result<()> {
        let bucket_id = self.bucket_id();
        let url = format!("{}/api/buckets/{}/batch", self.hf_client.endpoint(), bucket_id);

        let mut lines: Vec<String> = Vec::new();

        for add in &params.add {
            let mut obj = serde_json::json!({
                "type": "addFile",
                "path": add.path,
                "xetHash": add.xet_hash,
                "size": add.size,
            });
            if let Some(mtime) = add.mtime {
                obj["mtime"] = serde_json::Value::Number(mtime.into());
            }
            if let Some(ref ct) = add.content_type {
                obj["contentType"] = serde_json::Value::String(ct.clone());
            }
            lines.push(serde_json::to_string(&obj)?);
        }

        for path in &params.delete {
            let obj = serde_json::json!({
                "type": "deleteFile",
                "path": path,
            });
            lines.push(serde_json::to_string(&obj)?);
        }

        for copy in &params.copy {
            let obj = serde_json::json!({
                "type": "copyFile",
                "path": copy.path,
                "xetHash": copy.xet_hash,
                "sourceRepoType": copy.source_repo_type,
                "sourceRepoId": copy.source_repo_id,
            });
            lines.push(serde_json::to_string(&obj)?);
        }

        for chunk in lines.chunks(BUCKET_BATCH_CHUNK_SIZE) {
            let body = chunk.join("\n") + "\n";

            let response = self
                .hf_client
                .http_client()
                .post(&url)
                .headers(self.hf_client.auth_headers())
                .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
                .body(body)
                .send()
                .await?;

            self.hf_client
                .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
                .await?;
        }

        Ok(())
    }

    /// Delete files from this bucket by path.
    pub async fn delete_files(&self, paths: &[String]) -> Result<()> {
        let params = BatchBucketFilesParams {
            add: vec![],
            delete: paths.to_vec(),
            copy: vec![],
        };
        self.batch(&params).await
    }

    /// Upload local files to the bucket.
    ///
    /// Uploads file contents to xet, then registers them via the batch endpoint.
    #[cfg(feature = "xet")]
    pub async fn upload_files(&self, files: &[(std::path::PathBuf, String)], progress: &Progress) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }

        let total_bytes: u64 = files
            .iter()
            .filter_map(|(p, _)| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        progress::emit(
            progress,
            ProgressEvent::Upload(UploadEvent::Start {
                total_files: files.len(),
                total_bytes,
            }),
        );

        let xet_files: Vec<(String, crate::types::AddSource)> = files
            .iter()
            .map(|(local_path, remote_path)| (remote_path.clone(), crate::types::AddSource::File(local_path.clone())))
            .collect();

        let xet_infos = self.xet_upload(&xet_files, progress).await?;

        let add_files: Vec<crate::types::BucketAddFile> = files
            .iter()
            .zip(xet_infos.iter())
            .map(|((local_path, remote_path), xet_info)| {
                let metadata = std::fs::metadata(local_path).ok();
                let size = metadata.as_ref().map(|m| m.len()).or(xet_info.file_size).unwrap_or(0);
                let mtime = metadata
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());

                crate::types::BucketAddFile {
                    path: remote_path.clone(),
                    xet_hash: xet_info.hash.clone(),
                    size,
                    mtime,
                    content_type: None,
                }
            })
            .collect();

        let batch_params = crate::types::BatchBucketFilesParams {
            add: add_files,
            ..Default::default()
        };
        self.batch(&batch_params).await?;

        progress::emit(progress, ProgressEvent::Upload(UploadEvent::Complete));
        Ok(())
    }

    /// Upload local files to the bucket (stub when xet feature is disabled).
    #[cfg(not(feature = "xet"))]
    pub async fn upload_files(&self, _files: &[(std::path::PathBuf, String)], _progress: &Progress) -> Result<()> {
        Err(HFError::XetNotEnabled)
    }

    /// Download files from the bucket to local paths.
    ///
    /// Resolves xet hashes via `get_paths_info`, then downloads via xet.
    #[cfg(feature = "xet")]
    pub async fn download_files(
        &self,
        params: &crate::types::BucketDownloadFilesParams,
        progress: &Progress,
    ) -> Result<()> {
        if params.files.is_empty() {
            return Ok(());
        }

        let remote_paths: Vec<String> = params.files.iter().map(|(r, _)| r.clone()).collect();
        let entries = self.get_paths_info(&remote_paths).await?;

        let entry_map: std::collections::HashMap<String, BucketTreeEntry> = entries
            .into_iter()
            .map(|e| {
                let path = match &e {
                    BucketTreeEntry::File { path, .. } => path.clone(),
                    BucketTreeEntry::Directory { path, .. } => path.clone(),
                };
                (path, e)
            })
            .collect();

        let mut xet_batch_files = Vec::new();
        let mut total_bytes: u64 = 0;

        for (remote_path, local_path) in &params.files {
            match entry_map.get(remote_path) {
                Some(BucketTreeEntry::File { xet_hash, size, .. }) => {
                    total_bytes += size;
                    xet_batch_files.push(crate::xet::XetBatchFile {
                        hash: xet_hash.clone(),
                        file_size: *size,
                        path: local_path.clone(),
                        filename: remote_path.clone(),
                    });
                },
                Some(BucketTreeEntry::Directory { path, .. }) => {
                    return Err(HFError::InvalidParameter(format!("Cannot download directory entry: {path}")));
                },
                None => {
                    return Err(HFError::EntryNotFound {
                        path: remote_path.clone(),
                        repo_id: self.bucket_id(),
                    });
                },
            }
        }

        progress::emit(
            progress,
            ProgressEvent::Download(DownloadEvent::Start {
                total_files: xet_batch_files.len(),
                total_bytes,
            }),
        );

        self.xet_download_batch(&xet_batch_files, progress).await?;

        progress::emit(progress, ProgressEvent::Download(DownloadEvent::Complete));
        Ok(())
    }

    /// Download files from the bucket (stub when xet feature is disabled).
    #[cfg(not(feature = "xet"))]
    pub async fn download_files(
        &self,
        _params: &crate::types::BucketDownloadFilesParams,
        _progress: &Progress,
    ) -> Result<()> {
        Err(HFError::XetNotEnabled)
    }
}

sync_api! {
    impl HFClient -> HFClientSync {
        fn create_bucket(&self, params: &CreateBucketParams) -> Result<BucketUrl>;
        fn delete_bucket(&self, bucket_id: &str, missing_ok: bool) -> Result<()>;
        fn move_bucket(&self, from_id: &str, to_id: &str) -> Result<()>;
    }
}

sync_api_stream! {
    impl HFClient -> HFClientSync {
        fn list_buckets(&self, namespace: &str) -> BucketInfo;
    }
}

sync_api! {
    impl HFBucket -> HFBucketSync {
        fn info(&self) -> Result<BucketInfo>;
        fn get_file_metadata(&self, remote_path: &str) -> Result<BucketFileMetadata>;
        fn get_paths_info(&self, paths: &[String]) -> Result<Vec<BucketTreeEntry>>;
        fn batch(&self, params: &BatchBucketFilesParams) -> Result<()>;
        fn upload_files(&self, files: &[(std::path::PathBuf, String)], progress: &Progress) -> Result<()>;
        fn download_files(&self, params: &crate::types::BucketDownloadFilesParams, progress: &Progress) -> Result<()>;
        fn delete_files(&self, paths: &[String]) -> Result<()>;
        fn sync(&self, params: &BucketSyncParams) -> Result<SyncPlan>;
    }
}

sync_api_stream! {
    impl HFBucket -> HFBucketSync {
        fn list_tree(&self, params: &ListBucketTreeParams) -> BucketTreeEntry;
    }
}
