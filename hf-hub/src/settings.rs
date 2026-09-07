//! Account settings: per-repository storage usage.
//!
//! This module exposes the [`RepoStorageEntry`] type and
//! [`HFClient::list_settings_repositories`], the paginated listing behind the Hub's
//! storage settings page for a user or an organization.

use bon::bon;
use futures::Stream;
use serde::Deserialize;
use url::Url;

use crate::client::HFClient;
use crate::error::HFResult;

/// Storage usage for a single repository, as reported by the storage settings endpoints.
///
/// Yielded by [`HFClient::list_settings_repositories`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoStorageEntry {
    /// Repo ID, in the form `owner/name`.
    pub id: String,
    /// Repo kind, one of `"model"`, `"dataset"`, or `"space"`.
    #[serde(rename = "type")]
    pub repo_type: String,
    /// ISO-8601 timestamp of the most recent commit to the repo.
    pub updated_at: Option<String>,
    /// Visibility of the repo, either `"public"` or `"private"`.
    pub visibility: String,
    /// Total size of the repo on disk, in bytes.
    pub storage: Option<u64>,
    /// Share of the namespace's total storage this repo accounts for, as a percentage.
    pub storage_percent: Option<f64>,
}

#[bon]
impl HFClient {
    /// Stream per-repository storage usage for a namespace.
    ///
    /// Endpoint: `GET /api/settings/repositories` for the authenticated caller's own
    /// namespace, or `GET /api/organizations/{namespace}/settings/repositories` when
    /// `namespace` is set.
    ///
    /// # Parameters
    ///
    /// - `namespace`: Hub handle of an organization. Omit it to list the caller's own repositories. For a value that is
    ///   already an `Option<String>` at runtime, pass it through `maybe_namespace(opt)`.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_settings_repositories(
        &self,
        /// Hub handle of an organization. Omit it to list the caller's own repositories.
        namespace: Option<String>,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<RepoStorageEntry>> + '_> {
        let url = match namespace {
            Some(namespace) => {
                format!("{}/api/organizations/{}/settings/repositories", self.endpoint(), namespace)
            },
            None => format!("{}/api/settings/repositories", self.endpoint()),
        };
        Ok(self.paginate(Url::parse(&url)?, vec![], limit))
    }
}

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::list_settings_repositories`]. Collects the stream into a
    /// `Vec<RepoStorageEntry>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_settings_repositories(
        &self,
        namespace: Option<String>,
        limit: Option<usize>,
    ) -> HFResult<Vec<RepoStorageEntry>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_settings_repositories()
                .maybe_namespace(namespace)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::RepoStorageEntry;
    use crate::client::HFClient;

    /// Bind a loopback listener that answers one request with `body`, and return the
    /// endpoint to point a client at plus a handle yielding the request line it saw.
    async fn serve_one_json(body: &'static str) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = BufReader::new(socket);
            let mut request_line = String::new();
            socket.read_line(&mut request_line).await.unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.get_mut().write_all(response.as_bytes()).await.unwrap();
            request_line
        });
        (endpoint, server)
    }

    const BODY: &str = r#"[
        {
            "id":"alice/thing",
            "type":"model",
            "updatedAt":"2026-01-02T03:04:05.000Z",
            "visibility":"private",
            "storage":2048,
            "storagePercent":12.5
        }
    ]"#;

    #[test]
    fn test_repo_storage_entry_optional_fields() {
        let json = r#"{"id":"alice/thing","type":"dataset","visibility":"public"}"#;
        let entry: RepoStorageEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.id, "alice/thing");
        assert_eq!(entry.repo_type, "dataset");
        assert_eq!(entry.visibility, "public");
        assert!(entry.updated_at.is_none());
        assert!(entry.storage.is_none());
        assert!(entry.storage_percent.is_none());
    }

    #[tokio::test]
    async fn list_settings_repositories_without_namespace_uses_the_caller_endpoint() {
        let (endpoint, server) = serve_one_json(BODY).await;

        let client = HFClient::builder().endpoint(endpoint).build().unwrap();
        let stream = client.list_settings_repositories().send().unwrap();
        futures::pin_mut!(stream);
        let entry = stream.next().await.unwrap().unwrap();

        assert_eq!(server.await.unwrap(), "GET /api/settings/repositories HTTP/1.1\r\n");
        assert_eq!(entry.id, "alice/thing");
        assert_eq!(entry.repo_type, "model");
        assert_eq!(entry.updated_at.as_deref(), Some("2026-01-02T03:04:05.000Z"));
        assert_eq!(entry.visibility, "private");
        assert_eq!(entry.storage, Some(2048));
        assert_eq!(entry.storage_percent, Some(12.5));
    }

    #[tokio::test]
    async fn list_settings_repositories_with_namespace_uses_the_organization_endpoint() {
        let (endpoint, server) = serve_one_json(BODY).await;

        let client = HFClient::builder().endpoint(endpoint).build().unwrap();
        let stream = client
            .list_settings_repositories()
            .namespace("acme".to_string())
            .send()
            .unwrap();
        futures::pin_mut!(stream);
        let entry = stream.next().await.unwrap().unwrap();

        assert_eq!(server.await.unwrap(), "GET /api/organizations/acme/settings/repositories HTTP/1.1\r\n");
        assert_eq!(entry.id, "alice/thing");
    }

    #[tokio::test]
    async fn list_settings_repositories_accepts_a_runtime_optional_namespace() {
        let (endpoint, server) = serve_one_json(BODY).await;
        let namespace: Option<String> = Some("acme".to_string());

        let client = HFClient::builder().endpoint(endpoint).build().unwrap();
        let stream = client.list_settings_repositories().maybe_namespace(namespace).send().unwrap();
        futures::pin_mut!(stream);
        stream.next().await.unwrap().unwrap();

        assert_eq!(server.await.unwrap(), "GET /api/organizations/acme/settings/repositories HTTP/1.1\r\n");
    }
}
