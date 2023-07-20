use crate::Error;
use crate::{Cache, Repo};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{distributions::Alphanumeric, thread_rng, Rng};
use std::collections::HashMap;
// use reqwest::{
//     blocking::Agent,
//     header::{
//         HeaderMap, HeaderName, HeaderValue, InvalidHeaderValue, ToStrError, AUTHORIZATION,
//         CONTENT_RANGE, LOCATION, RANGE, USER_AGENT,
//     },
//     redirect::Policy,
//     Error as ReqwestError,
// };
use serde::Deserialize;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use ureq::Agent;

/// Current version (used in user-agent)
const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Current name (used in user-agent)
const NAME: &str = env!("CARGO_PKG_NAME");

const RANGE: &str = "Range";
const CONTENT_RANGE: &str = "Content-Range";
const LOCATION: &str = "Location";
const USER_AGENT: &str = "User-Agent";
const AUTHORIZATION: &str = "Authorization";
type HeaderMap = HashMap<&'static str, String>;
pub(crate) type HeaderName = &'static str;

/// Simple wrapper over [`ureq::Agent`] to include default headers
#[derive(Clone)]
pub struct HeaderAgent {
    agent: Agent,
    headers: HeaderMap,
}

impl HeaderAgent {
    fn new(agent: Agent, headers: HeaderMap) -> Self {
        Self { agent, headers }
    }

    fn get(&self, url: &str) -> ureq::Request {
        let mut request = self.agent.get(url);
        for (header, value) in &self.headers {
            request = request.set(header, &value);
        }
        request
    }
}

/// Siblings are simplified file descriptions of remote files on the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Siblings {
    /// The path within the repo.
    pub rfilename: String,
}

/// The description of the repo given by the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// See [`Siblings`]
    pub siblings: Vec<Siblings>,
}

/// Helper to create [`Api`] with all the options.
pub struct ApiBuilder {
    endpoint: String,
    cache: Cache,
    url_template: String,
    token: Option<String>,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    progress: bool,
}

impl Default for ApiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiBuilder {
    /// Default api builder
    /// ```
    /// use hf_hub::api::sync::ApiBuilder;
    /// let api = ApiBuilder::new().build().unwrap();
    /// ```
    pub fn new() -> Self {
        let cache = Cache::default();
        let mut token_filename = cache.path().clone();
        token_filename.push("token");
        let token = match std::fs::read_to_string(token_filename) {
            Ok(token_content) => {
                let token_content = token_content.trim();
                if !token_content.is_empty() {
                    Some(token_content.to_string())
                } else {
                    None
                }
            }
            Err(_) => None,
        };

        let progress = true;

        Self {
            endpoint: "https://huggingface.co".to_string(),
            url_template: "{endpoint}/{repo_id}/resolve/{revision}/{filename}".to_string(),
            cache,
            token,
            chunk_size: 10_000_000,
            parallel_failures: 0,
            max_retries: 0,
            progress,
        }
    }

    /// Wether to show a progressbar
    pub fn with_progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }

    /// Changes the location of the cache directory. Defaults is `~/.cache/huggingface/`.
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache = Cache::new(cache_dir);
        self
    }

    /// Sets the token to be used in the API
    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    fn build_headers(&self) -> Result<HeaderMap, Error> {
        let mut headers = HeaderMap::new();
        let user_agent = format!("unkown/None; {NAME}/{VERSION}; rust/unknown");
        headers.insert(USER_AGENT, user_agent);
        if let Some(token) = &self.token {
            headers.insert(AUTHORIZATION, format!("Bearer {token}"));
        }
        Ok(headers)
    }

    /// Consumes the builder and buids the final [`Api`]
    pub fn build(self) -> Result<Api, Error> {
        let headers = self.build_headers()?;
        let client = HeaderAgent::new(ureq::builder().build(), headers.clone());
        let no_redirect_client = HeaderAgent::new(ureq::builder().redirects(0).build(), headers);
        Ok(Api {
            endpoint: self.endpoint,
            url_template: self.url_template,
            cache: self.cache,
            client,

            no_redirect_client,
            chunk_size: self.chunk_size,
            parallel_failures: self.parallel_failures,
            max_retries: self.max_retries,
            progress: self.progress,
        })
    }
}

#[derive(Debug)]
struct Metadata {
    commit_hash: String,
    etag: String,
    size: usize,
}

/// The actual Api used to interacto with the hub.
/// You can inspect repos with [`Api::info`]
/// or download files with [`Api::download`]
pub struct Api {
    endpoint: String,
    url_template: String,
    cache: Cache,
    client: HeaderAgent,
    no_redirect_client: HeaderAgent,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    progress: bool,
}

fn temp_filename() -> PathBuf {
    let s: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect();
    let mut path = std::env::temp_dir();
    path.push(s);
    path
}

fn make_relative(src: &Path, dst: &Path) -> PathBuf {
    let path = src;
    let base = dst;

    if path.is_absolute() != base.is_absolute() {
        panic!("This function is made to look at absolute paths only");
    }
    let mut ita = path.components();
    let mut itb = base.components();

    loop {
        match (ita.next(), itb.next()) {
            (Some(a), Some(b)) if a == b => (),
            (some_a, _) => {
                // Ignoring b, because 1 component is the filename
                // for which we don't need to go back up for relative
                // filename to work.
                let mut new_path = PathBuf::new();
                for _ in itb {
                    new_path.push(Component::ParentDir);
                }
                if let Some(a) = some_a {
                    new_path.push(a);
                    for comp in ita {
                        new_path.push(comp);
                    }
                }
                return new_path;
            }
        }
    }
}

fn symlink_or_rename(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    if dst.exists() {
        return Ok(());
    }

    let src = make_relative(src, dst);
    #[cfg(target_os = "windows")]
    std::os::windows::fs::symlink_file(src, dst)?;

    #[cfg(target_family = "unix")]
    std::os::unix::fs::symlink(src, dst)?;

    #[cfg(not(any(target_family = "unix", target_os = "windows")))]
    std::fs::rename(src, dst)?;

    Ok(())
}

fn jitter() -> usize {
    thread_rng().gen_range(0..=500)
}

fn exponential_backoff(base_wait_time: usize, n: usize, max: usize) -> usize {
    (base_wait_time + n.pow(2) + jitter()).min(max)
}

impl Api {
    /// Creates a default Api, for Api options See [`ApiBuilder`]
    pub fn new() -> Result<Self, Error> {
        ApiBuilder::new().build()
    }

    /// Get the fully qualified URL of the remote filename
    /// ```
    /// # use hf_hub::{api::sync::Api, Repo};
    /// let api = Api::new().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// let url = api.url(&repo, "model.safetensors");
    /// assert_eq!(url, "https://huggingface.co/gpt2/resolve/main/model.safetensors");
    /// ```
    pub fn url(&self, repo: &Repo, filename: &str) -> String {
        let endpoint = &self.endpoint;
        let revision = &repo.url_revision();
        self.url_template
            .replace("{endpoint}", endpoint)
            .replace("{repo_id}", &repo.url())
            .replace("{revision}", revision)
            .replace("{filename}", filename)
    }

    /// Get the underlying api client
    /// Allows for lower level access
    pub fn client(&self) -> &HeaderAgent {
        &self.client
    }

    fn metadata(&self, url: &str) -> Result<Metadata, Error> {
        let response = self
            .no_redirect_client
            .get(url)
            .set(RANGE, "bytes=0-0")
            .call()?;
        // let headers = response.headers();
        let header_commit = "x-repo-commit";
        let header_linked_etag = "x-linked-etag";
        let header_etag = "etag";

        let etag = match response.header(&header_linked_etag) {
            Some(etag) => etag,
            None => response
                .header(&header_etag)
                .ok_or(Error::MissingHeader(header_etag))?,
        };
        // Cleaning extra quotes
        let etag = etag.to_string().replace('"', "");
        let commit_hash = response
            .header(&header_commit)
            .ok_or(Error::MissingHeader(header_commit))?
            .to_string();

        // The response was redirected o S3 most likely which will
        // know about the size of the file
        let status = response.status();
        let is_redirection = status >= 300 && status < 400;
        let response = if is_redirection {
            self.client
                .get(response.header(LOCATION).unwrap())
                .set(RANGE, "bytes=0-0")
                .call()?
        } else {
            response
        };
        let content_range = response
            .header(CONTENT_RANGE)
            .ok_or(Error::MissingHeader(CONTENT_RANGE))?;

        let size = content_range
            .split('/')
            .last()
            .ok_or(Error::InvalidHeader(CONTENT_RANGE))?
            .parse()?;
        Ok(Metadata {
            commit_hash,
            etag,
            size,
        })
    }

    fn download_tempfile(
        &self,
        url: &str,
        length: usize,
        progressbar: Option<ProgressBar>,
    ) -> Result<PathBuf, Error> {
        let filename = temp_filename();

        // Create the file and set everything properly
        std::fs::File::create(&filename)?.set_len(length as u64)?;

        let chunk_size = self.chunk_size;

        let n_chunks = (length + chunk_size - 1) / chunk_size;
        let n_threads = num_cpus::get();
        let chunks_per_thread = (n_chunks + n_threads - 1) / n_threads;
        let handles = (0..n_threads).map(|thread_id| {
            let url = url.to_string();
            let filename = filename.clone();
            let client = self.client.clone();
            let parallel_failures = self.parallel_failures;
            let max_retries = self.max_retries;
            let progress = progressbar.clone();
            std::thread::spawn(move || {
                for chunk_id in chunks_per_thread * thread_id
                    ..std::cmp::min(chunks_per_thread * (thread_id + 1), n_chunks)
                {
                    let start = chunk_id * chunk_size;
                    let stop = std::cmp::min(start + chunk_size - 1, length);
                    let mut chunk = Self::download_chunk(&client, &url, &filename, start, stop);
                    let mut i = 0;
                    if parallel_failures > 0 {
                        while let Err(dlerr) = chunk {
                            let wait_time = exponential_backoff(300, i, 10_000);
                            std::thread::sleep(std::time::Duration::from_millis(wait_time as u64));

                            chunk = Self::download_chunk(&client, &url, &filename, start, stop);
                            i += 1;
                            if i > max_retries {
                                return Err(Error::TooManyRetries(dlerr.into()));
                            }
                        }
                    }
                    if let Some(p) = &progress {
                        p.inc((stop - start) as u64);
                    }
                    chunk?
                }
                Ok(())
            })
        });

        let results: Result<Vec<()>, Error> = handles.into_iter().flat_map(|h| h.join()).collect();

        results?;
        if let Some(p) = progressbar {
            p.finish()
        }
        Ok(filename)
    }

    fn download_chunk(
        client: &HeaderAgent,
        url: &str,
        filename: &PathBuf,
        start: usize,
        stop: usize,
    ) -> Result<(), Error> {
        // Process each socket concurrently.
        let range = format!("bytes={start}-{stop}");
        let mut file = std::fs::OpenOptions::new().write(true).open(filename)?;
        file.seek(SeekFrom::Start(start as u64))?;
        let response = client.get(url).set(RANGE, &range).call()?;

        const MAX: usize = 4096;
        let mut buffer: [u8; MAX] = [0; MAX];
        let mut reader = response.into_reader();
        let mut remaining = stop - start;
        while remaining > 0 {
            let to_read = if remaining > MAX { MAX } else { remaining };

            reader.read_exact(&mut buffer[0..to_read])?;
            remaining -= to_read;
            file.write_all(&buffer[0..to_read])?;
        }
        // file.write_all(&content)?;
        Ok(())
    }

    /// This will attempt the fetch the file locally first, then [`Api.download`]
    /// if the file is not present.
    /// ```no_run
    /// use hf_hub::{api::sync::ApiBuilder, Repo};
    /// let api = ApiBuilder::new().build().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// let local_filename = api.get(&repo, "model.safetensors").unwrap();
    pub fn get(&self, repo: &Repo, filename: &str) -> Result<PathBuf, Error> {
        if let Some(path) = self.cache.get(repo, filename) {
            Ok(path)
        } else {
            self.download(repo, filename)
        }
    }

    /// Downloads a remote file (if not already present) into the cache directory
    /// to be used locally.
    /// This functions require internet access to verify if new versions of the file
    /// exist, even if a file is already on disk at location.
    /// ```no_run
    /// # use hf_hub::{api::sync::ApiBuilder, Repo};
    /// let api = ApiBuilder::new().build().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// let local_filename = api.download(&repo, "model.safetensors").unwrap();
    /// ```
    pub fn download(&self, repo: &Repo, filename: &str) -> Result<PathBuf, Error> {
        let url = self.url(repo, filename);
        let metadata = self.metadata(&url)?;

        let blob_path = self.cache.blob_path(repo, &metadata.etag);
        std::fs::create_dir_all(blob_path.parent().unwrap())?;

        let progressbar = if self.progress {
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

        let tmp_filename = self.download_tempfile(&url, metadata.size, progressbar)?;

        if std::fs::rename(&tmp_filename, &blob_path).is_err() {
            // Renaming may fail if locations are different mount points
            std::fs::File::create(&blob_path)?;
            std::fs::copy(tmp_filename, &blob_path)?;
        }

        let mut pointer_path = self.cache.pointer_path(repo, &metadata.commit_hash);
        pointer_path.push(filename);
        std::fs::create_dir_all(pointer_path.parent().unwrap()).ok();

        symlink_or_rename(&blob_path, &pointer_path)?;
        self.cache.create_ref(repo, &metadata.commit_hash)?;

        Ok(pointer_path)
    }

    /// Get information about the Repo
    /// ```
    /// use hf_hub::{api::sync::Api, Repo};
    /// let api = Api::new().unwrap();
    /// let repo = Repo::model("gpt2".to_string());
    /// api.info(&repo);
    /// ```
    pub fn info(&self, repo: &Repo) -> Result<ModelInfo, Error> {
        let url = format!("{}/api/{}", self.endpoint, repo.api_url());
        let response = self.client.get(&url).call()?;

        let model_info = response.into_json()?;

        Ok(model_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RepoType;
    use hex_literal::hex;
    use rand::{distributions::Alphanumeric, Rng};
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
            std::fs::remove_dir_all(&self.path).unwrap()
        }
    }

    #[test]
    fn simple() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_cache_dir(tmp.path.clone())
            .build()
            .unwrap();
        let repo = Repo::new("julien-c/dummy-unknown".to_string(), RepoType::Model);
        let downloaded_path = api.download(&repo, "config.json").unwrap();
        assert!(downloaded_path.exists());
        let val = Sha256::digest(std::fs::read(&*downloaded_path).unwrap());
        assert_eq!(
            val[..],
            hex!("b908f2b7227d4d31a2105dfa31095e28d304f9bc938bfaaa57ee2cacf1f62d32")
        );

        // Make sure the file is now seeable without connection
        let cache_path = api.cache.get(&repo, "config.json").unwrap();
        assert_eq!(cache_path, downloaded_path);
    }

    #[test]
    fn dataset() {
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
            .download(&repo, "wikitext-103-v1/wikitext-test.parquet")
            .unwrap();
        assert!(downloaded_path.exists());
        let val = Sha256::digest(std::fs::read(&*downloaded_path).unwrap());
        assert_eq!(
            val[..],
            hex!("59ce09415ad8aa45a9e34f88cec2548aeb9de9a73fcda9f6b33a86a065f32b90")
        )
    }

    #[test]
    fn info() {
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
        let model_info = api.info(&repo).unwrap();
        assert_eq!(
            model_info,
            ModelInfo {
                siblings: vec![
                    Siblings {
                        rfilename: ".gitattributes".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-train-00000-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-train-00001-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-raw-v1/wikitext-validation.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/test/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/validation/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-train-00000-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-train-00001-of-00002.parquet"
                            .to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-103-v1/wikitext-validation.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/test/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/train/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/validation/index.duckdb".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/wikitext-train.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-raw-v1/wikitext-validation.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/wikitext-test.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/wikitext-train.parquet".to_string()
                    },
                    Siblings {
                        rfilename: "wikitext-2-v1/wikitext-validation.parquet".to_string()
                    }
                ],
            }
        )
    }
}
