use super::{exponential_backoff, symlink_or_rename, ApiError, ApiRepo, RepoInfo};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{header::RANGE, RequestBuilder};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::Semaphore;

impl ApiRepo {
    /// Get the fully qualified URL of the remote filename
    /// ```
    /// # use hf_hub::api::tokio::Api;
    /// let api = Api::new().unwrap();
    /// let url = api.model("gpt2".to_string()).url("model.safetensors");
    /// assert_eq!(url, "https://huggingface.co/gpt2/resolve/main/model.safetensors");
    /// ```
    pub fn file_url(&self, filename: &str) -> String {
        let endpoint = &self.api.endpoint;
        let revision = &self.repo.url_revision();
        format!(
            "{endpoint}/{}/resolve/{revision}/{filename}",
            self.repo.url()
        )
        .replace("{endpoint}", endpoint)
        .replace("{repo_id}", &self.repo.url())
        .replace("{revision}", revision)
        .replace("{filename}", filename)
    }

    /// Get the fully qualified URL for a preupload
    /// ```
    /// # use hf_hub::api::tokio::Api;
    /// let api = Api::new().unwrap();
    /// let url = api.model("gpt2".to_string()).url("model.safetensors");
    /// assert_eq!(url, "https://huggingface.co/api/models/gpt2/model.safetensors/preupload/main");
    /// ```
    pub fn preupload_url(&self) -> String {
        let endpoint = &self.api.endpoint;
        let repo_id = self.repo.url();
        let repo_type = self.repo.repo_type.to_string();
        let revision = &self.repo.url_revision();
        format!("{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}")
    }

    async fn download_tempfile(
        &self,
        url: &str,
        length: usize,
        progressbar: Option<ProgressBar>,
    ) -> Result<PathBuf, ApiError> {
        let mut handles = vec![];
        let semaphore = Arc::new(Semaphore::new(self.api.max_files));
        let parallel_failures_semaphore = Arc::new(Semaphore::new(self.api.parallel_failures));
        let filename = self.api.cache.temp_path();

        // Create the file and set everything properly
        tokio::fs::File::create(&filename)
            .await?
            .set_len(length as u64)
            .await?;

        let chunk_size = self.api.chunk_size;
        for start in (0..length).step_by(chunk_size) {
            let url = url.to_string();
            let filename = filename.clone();
            let client = self.api.client.clone();

            let stop = std::cmp::min(start + chunk_size - 1, length);
            let permit = semaphore.clone().acquire_owned().await?;
            let parallel_failures = self.api.parallel_failures;
            let max_retries = self.api.max_retries;
            let parallel_failures_semaphore = parallel_failures_semaphore.clone();
            let progress = progressbar.clone();
            handles.push(tokio::spawn(async move {
                let mut chunk = Self::download_chunk(&client, &url, &filename, start, stop).await;
                let mut i = 0;
                if parallel_failures > 0 {
                    while let Err(dlerr) = chunk {
                        let parallel_failure_permit =
                            parallel_failures_semaphore.clone().try_acquire_owned()?;

                        let wait_time = exponential_backoff(300, i, 10_000);
                        tokio::time::sleep(tokio::time::Duration::from_millis(wait_time as u64))
                            .await;

                        chunk = Self::download_chunk(&client, &url, &filename, start, stop).await;
                        i += 1;
                        if i > max_retries {
                            return Err(ApiError::TooManyRetries(dlerr.into()));
                        }
                        drop(parallel_failure_permit);
                    }
                }
                drop(permit);
                if let Some(p) = progress {
                    p.inc((stop - start) as u64);
                }
                chunk
            }));
        }

        // Output the chained result
        let results: Vec<Result<Result<(), ApiError>, tokio::task::JoinError>> =
            futures::future::join_all(handles).await;
        let results: Result<(), ApiError> = results.into_iter().flatten().collect();
        results?;
        if let Some(p) = progressbar {
            p.finish();
        }
        Ok(filename)
    }

    async fn download_chunk(
        client: &reqwest::Client,
        url: &str,
        filename: &PathBuf,
        start: usize,
        stop: usize,
    ) -> Result<(), ApiError> {
        // Process each socket concurrently.
        let range = format!("bytes={start}-{stop}");
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(filename)
            .await?;
        file.seek(SeekFrom::Start(start as u64)).await?;
        let response = client
            .get(url)
            .header(RANGE, range)
            .send()
            .await?
            .error_for_status()?;
        let content = response.bytes().await?;
        file.write_all(&content).await?;
        Ok(())
    }

    /// This will attempt the fetch the file locally first, then [`Api.download`]
    /// if the file is not present.
    /// ```no_run
    /// # use hf_hub::api::tokio::Api;
    /// # tokio_test::block_on(async {
    /// let api = Api::new().unwrap();
    /// let local_filename = api.model("gpt2".to_string()).get("model.safetensors").await.unwrap();
    /// # })
    pub async fn get(&self, filename: &str) -> Result<PathBuf, ApiError> {
        if let Some(path) = self.api.cache.repo(self.repo.clone()).get(filename) {
            Ok(path)
        } else {
            self.download(filename).await
        }
    }

    /// Downloads a remote file (if not already present) into the cache directory
    /// to be used locally.
    /// This functions require internet access to verify if new versions of the file
    /// exist, even if a file is already on disk at location.
    /// ```no_run
    /// # use hf_hub::api::tokio::Api;
    /// # tokio_test::block_on(async {
    /// let api = Api::new().unwrap();
    /// let local_filename = api.model("gpt2".to_string()).download("model.safetensors").await.unwrap();
    /// # })
    /// ```
    pub async fn download(&self, filename: &str) -> Result<PathBuf, ApiError> {
        let url = self.file_url(filename);
        let metadata = self.api.metadata(&url).await?;
        let cache = self.api.cache.repo(self.repo.clone());

        let blob_path = cache.blob_path(&metadata.etag);
        std::fs::create_dir_all(blob_path.parent().unwrap())?;

        let progressbar = if self.api.progress {
            let progress = ProgressBar::new(metadata.size as u64);
            progress.set_style(
                ProgressStyle::with_template(
                    "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec} ({eta})",
                )
                    .unwrap(), // .progress_chars("━ "),
            );
            let maxlength = 30;
            let message = if filename.len() > maxlength {
                format!("..{}", &filename[filename.len() - maxlength..])
            } else {
                filename.to_string()
            };
            progress.set_message(message);
            Some(progress)
        } else {
            None
        };

        let tmp_filename = self
            .download_tempfile(&url, metadata.size, progressbar)
            .await?;

        tokio::fs::rename(&tmp_filename, &blob_path).await?;

        let mut pointer_path = cache.pointer_path(&metadata.commit_hash);
        pointer_path.push(filename);
        std::fs::create_dir_all(pointer_path.parent().unwrap()).ok();

        symlink_or_rename(&blob_path, &pointer_path)?;
        cache.create_ref(&metadata.commit_hash)?;

        Ok(pointer_path)
    }

    /// Get information about the Repo
    /// ```
    /// # use hf_hub::api::tokio::Api;
    /// # tokio_test::block_on(async {
    /// let api = Api::new().unwrap();
    /// api.model("gpt2".to_string()).info();
    /// # })
    /// ```
    pub async fn info(&self) -> Result<RepoInfo, ApiError> {
        Ok(self.info_request().send().await?.json().await?)
    }

    /// Get the raw [`reqwest::RequestBuilder`] with the url and method already set
    /// ```
    /// # use hf_hub::api::tokio::Api;
    /// # tokio_test::block_on(async {
    /// let api = Api::new().unwrap();
    /// api.model("gpt2".to_owned())
    ///     .info_request()
    ///     .query(&[("blobs", "true")])
    ///     .send()
    ///     .await;
    /// # })
    /// ```
    pub fn info_request(&self) -> RequestBuilder {
        let url = format!("{}/api/{}", self.api.endpoint, self.repo.api_url());
        self.api.client.get(url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::{tokio::ApiBuilder, Siblings},
        Repo, RepoType,
    };
    use hex_literal::hex;
    use rand::{distributions::Alphanumeric, Rng};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn new() -> Self {
            let s: String = rand::thread_rng()
                .sample_iter(&Alphanumeric)
                .take(7)
                .map(char::from)
                .collect();
            let mut path = std::env::temp_dir();
            path.push(s);
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).unwrap();
        }
    }

    #[tokio::test]
    async fn simple() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let model_id = "julien-c/dummy-unknown".to_string();
        let repo = Repo::new(model_id.clone(), RepoType::Model);
        let downloaded_path = api.model(model_id).download("config.json").await.unwrap();
        assert!(downloaded_path.exists());
        let val = Sha256::digest(std::fs::read(&*downloaded_path).unwrap());
        assert_eq!(
            val[..],
            hex!("b908f2b7227d4d31a2105dfa31095e28d304f9bc938bfaaa57ee2cacf1f62d32")
        );

        // Make sure the file is now seeable without connection
        let cache_path = api.cache.repo(repo.clone()).get("config.json").unwrap();
        assert_eq!(cache_path, downloaded_path);
    }

    #[tokio::test]
    async fn revision() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let model_id = "BAAI/bge-base-en".to_string();
        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, "refs/pr/2".to_string());
        let downloaded_path = api
            .repo(repo.clone())
            .download("tokenizer.json")
            .await
            .unwrap();
        assert!(downloaded_path.exists());
        let val = Sha256::digest(std::fs::read(&*downloaded_path).unwrap());
        assert_eq!(
            val[..],
            hex!("d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66")
        );

        // Make sure the file is now seeable without connection
        let cache_path = api.cache.repo(repo).get("tokenizer.json").unwrap();
        assert_eq!(cache_path, downloaded_path);
    }

    #[tokio::test]
    async fn dataset() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::with_revision(
            "wikitext".to_string(),
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let downloaded_path = api
            .repo(repo)
            .download("wikitext-103-v1/test/0000.parquet")
            .await
            .unwrap();
        assert!(downloaded_path.exists());
        let val = Sha256::digest(std::fs::read(&*downloaded_path).unwrap());
        assert_eq!(
            val[..],
            hex!("ABDFC9F83B1103B502924072460D4C92F277C9B49C313CEF3E48CFCF7428E125")
        );
    }

    #[tokio::test]
    async fn models() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::with_revision(
            "BAAI/bGe-reRanker-Base".to_string(),
            RepoType::Model,
            "refs/pr/5".to_string(),
        );
        let downloaded_path = api.repo(repo).download("tokenizer.json").await.unwrap();
        assert!(downloaded_path.exists());
        let val = Sha256::digest(std::fs::read(&*downloaded_path).unwrap());
        assert_eq!(
            val[..],
            hex!("9EB652AC4E40CC093272BBBE0F55D521CF67570060227109B5CDC20945A4489E")
        );
    }

    #[tokio::test]
    async fn info() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::with_revision(
            "wikitext".to_string(),
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let model_info = api.repo(repo).info().await.unwrap();
        assert_eq!(
            model_info,
            RepoInfo {
                siblings: vec![
                    Siblings {
                        rfilename: ".gitattributes".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/test/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/train/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/train/0001.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/validation/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/test/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/train/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/train/0001.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/validation/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/test/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/train/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/validation/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/test/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/train/0000.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/validation/0000.parquet".to_string()
                    }
                ],
                sha: "3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e".to_string()
            }
        );
    }

    #[tokio::test]
    async fn info_request() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_token(None)
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::with_revision(
            "mcpotato/42-eicar-street".to_string(),
            RepoType::Model,
            "8b3861f6931c4026b0cd22b38dbc09e7668983ac".to_string(),
        );
        let blobs_info: Value = api
            .repo(repo)
            .info_request()
            .query(&[("blobs", "true")])
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            blobs_info,
            json!({
                "_id": "621ffdc136468d709f17ddb4",
                "author": "mcpotato",
                "createdAt": "2022-03-02T23:29:05.000Z",
                "disabled": false,
                "downloads": 0,
                "gated": false,
                "id": "mcpotato/42-eicar-street",
                "lastModified": "2022-11-30T19:54:16.000Z",
                "likes": 0,
                "modelId": "mcpotato/42-eicar-street",
                "private": false,
                "sha": "8b3861f6931c4026b0cd22b38dbc09e7668983ac",
                "siblings": [
                    {
                        "blobId": "6d34772f5ca361021038b404fb913ec8dc0b1a5a",
                        "rfilename": ".gitattributes",
                        "size": 1175
                    },
                    {
                        "blobId": "be98037f7c542112c15a1d2fc7e2a2427e42cb50",
                        "rfilename": "build_pickles.py",
                        "size": 304
                    },
                    {
                        "blobId": "8acd02161fff53f9df9597e377e22b04bc34feff",
                        "rfilename": "danger.dat",
                        "size": 66
                    },
                    {
                        "blobId": "86b812515e075a1ae216e1239e615a1d9e0b316e",
                        "rfilename": "eicar_test_file",
                        "size": 70
                    },
                    {
                        "blobId": "86b812515e075a1ae216e1239e615a1d9e0b316e",
                        "rfilename": "eicar_test_file_bis",
                        "size":70
                    },
                    {
                        "blobId": "cd1c6d8bde5006076655711a49feae66f07d707e",
                        "lfs": {
                            "pointerSize": 127,
                            "sha256": "f9343d7d7ec5c3d8bcced056c438fc9f1d3819e9ca3d42418a40857050e10e20",
                            "size": 22
                        },
                        "rfilename": "pytorch_model.bin",
                        "size": 22
                    },
                    {
                        "blobId": "8ab39654695136173fee29cba0193f679dfbd652",
                        "rfilename": "supposedly_safe.pkl",
                        "size": 31
                    }
                ],
                "spaces": [],
                "tags": ["pytorch", "region:us"],
            })
        );
    }
}
