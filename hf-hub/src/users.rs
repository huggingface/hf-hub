//! Users and organizations: identity, profile lookup, and social listings.
//!
//! This module exposes the [`User`], [`OrgMembership`], and [`Organization`]
//! types and the corresponding [`HFClient`] methods:
//!
//! - [`HFClient::whoami`] — identify the caller and verify that the current token is valid.
//! - [`HFClient::user_overview`] / [`HFClient::organization_overview`] — fetch a public profile by username or
//!   organization name.
//! - [`HFClient::list_user_followers`] / [`HFClient::list_user_following`] / [`HFClient::list_organization_members`] /
//!   [`HFClient::list_organization_followers`] — paginated listings that yield [`User`] entries one page at a time.
//! - [`HFClient::list_user_likes`] — paginated listing of the repos a user has liked.

use bon::bon;
use futures::Stream;
use serde::Deserialize;
use url::Url;

use crate::client::HFClient;
use crate::error::HFResult;
use crate::retry;

/// A Hugging Face Hub user account.
///
/// Returned by [`HFClient::whoami`] and the various user-lookup endpoints.
/// Only [`username`](Self::username) is guaranteed to be set; the remaining
/// fields are populated for the authenticated caller's own `whoami` response
/// or when the field is publicly visible on the target user's profile.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct User {
    /// Hub handle (slug) of the user — the name used in URLs such as
    /// `https://huggingface.co/<username>`.
    #[serde(alias = "login", alias = "user", alias = "name")]
    pub username: String,
    /// Display name as shown on the user's profile, when set.
    pub fullname: Option<String>,
    /// URL to the user's avatar image.
    pub avatar_url: Option<String>,
    /// Account type, typically `"user"` or `"org"`.
    #[serde(rename = "type")]
    pub user_type: Option<String>,
    /// Free-text bio shown on the user's profile.
    pub details: Option<String>,
    /// Whether the authenticated caller follows this user.
    pub is_following: Option<bool>,
    /// Whether the user is on a Pro plan.
    pub is_pro: Option<bool>,
    /// Number of models created by the user.
    pub num_models: Option<u64>,
    /// Number of datasets created by the user.
    pub num_datasets: Option<u64>,
    /// Number of Spaces created by the user.
    pub num_spaces: Option<u64>,
    /// Number of discussions initiated by the user.
    pub num_discussions: Option<u64>,
    /// Number of papers authored by the user.
    pub num_papers: Option<u64>,
    /// Upvotes the user has received.
    pub num_upvotes: Option<u64>,
    /// Likes the user has given.
    pub num_likes: Option<u64>,
    /// Number of users this user is following.
    pub num_following: Option<u64>,
    /// Number of users following this user.
    pub num_followers: Option<u64>,
    /// Email address — only returned by `whoami` for the authenticated user.
    pub email: Option<String>,
    /// Whether the email has been verified — only returned by `whoami`.
    pub email_verified: Option<bool>,
    /// Billing plan identifier — only returned by `whoami`.
    pub plan: Option<String>,
    /// Whether the account has a valid payment method — only returned by `whoami`.
    pub can_pay: Option<bool>,
    /// Organizations the authenticated user belongs to. Only populated by
    /// `whoami` for the caller themselves.
    pub orgs: Option<Vec<OrgMembership>>,
    /// Details about the token used for the request — only returned by `whoami`.
    pub auth: Option<AuthInfo>,
}

/// The token that authenticated a `whoami` request.
///
/// Returned inside [`User::auth`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthInfo {
    /// The access token the request was made with, when the request used one.
    pub access_token: Option<AccessTokenInfo>,
}

/// Metadata about the access token that authenticated a `whoami` request.
///
/// Returned inside [`AuthInfo::access_token`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccessTokenInfo {
    /// Name the token was given when it was created.
    pub display_name: Option<String>,
    /// Token role, typically `"read"`, `"write"`, or `"fineGrained"`.
    pub role: Option<String>,
    /// ISO-8601 timestamp when the token was created.
    pub created_at: Option<String>,
    /// Fine-grained permission scopes, present only for fine-grained tokens. Left untyped
    /// because the Hub's shape here is not stable; most callers only need to know whether
    /// it is set.
    #[serde(default)]
    pub fine_grained: Option<serde_json::Value>,
}

/// Summary entry for an organization the authenticated user belongs to.
///
/// Returned inside [`User::orgs`]. This is a lighter-weight shape than
/// [`Organization`] — use [`HFClient::organization_overview`] to fetch the
/// full record by name.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrgMembership {
    /// Hub handle (slug) of the organization.
    pub name: Option<String>,
    /// Display name as shown on the organization's profile.
    pub fullname: Option<String>,
    /// URL to the organization's avatar image.
    pub avatar_url: Option<String>,
    /// The caller's role in the organization, typically `"admin"`, `"write"`, `"contributor"`,
    /// or `"read"`.
    #[serde(rename = "roleInOrg")]
    pub role_in_org: Option<String>,
}

/// A repository a user has liked.
///
/// Returned inside [`LikedRepoEntry::repo`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LikedRepoRef {
    /// Repo ID, in the form `owner/name`.
    pub name: String,
    /// Repo kind, one of `"model"`, `"dataset"`, or `"space"`.
    #[serde(rename = "type")]
    pub repo_type: String,
}

/// One entry in a user's list of liked repositories.
///
/// Yielded by [`HFClient::list_user_likes`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LikedRepoEntry {
    /// ISO-8601 timestamp when the like was recorded.
    pub created_at: Option<String>,
    /// The liked repository.
    pub repo: LikedRepoRef,
}

/// A Hugging Face Hub organization.
///
/// Returned by [`HFClient::organization_overview`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Organization {
    /// Hub handle (slug) of the organization — the name used in URLs such as
    /// `https://huggingface.co/<name>`.
    pub name: String,
    /// Display name as shown on the organization's profile, when set.
    pub fullname: Option<String>,
    /// URL to the organization's avatar image.
    pub avatar_url: Option<String>,
    /// Account type, typically `"org"`.
    #[serde(rename = "type")]
    pub org_type: Option<String>,
    /// Free-text description shown on the organization's profile.
    pub details: Option<String>,
    /// Whether the organization is verified.
    pub is_verified: Option<bool>,
    /// Whether the authenticated caller follows this organization.
    pub is_following: Option<bool>,
    /// Number of members in the organization.
    pub num_users: Option<u64>,
    /// Number of models owned by the organization.
    pub num_models: Option<u64>,
    /// Number of Spaces owned by the organization.
    pub num_spaces: Option<u64>,
    /// Number of datasets owned by the organization.
    pub num_datasets: Option<u64>,
    /// Number of followers of the organization.
    pub num_followers: Option<u64>,
    /// Number of papers authored by the organization.
    pub num_papers: Option<u64>,
    /// Plan identifier (e.g., `"enterprise"`, `"team"`).
    pub plan: Option<String>,
}

#[bon]
impl HFClient {
    /// Fetch the profile of the user that owns the current token.
    ///
    /// Returns the authenticated [`User`], including private fields like
    /// [`email`](User::email) and the caller's [`orgs`](User::orgs) list. Fails with
    /// [`HFError::AuthRequired`](crate::HFError::AuthRequired) if no valid token is configured.
    ///
    /// Endpoint: `GET /api/whoami-v2`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn whoami(&self) -> HFResult<User> {
        let url = format!("{}/api/whoami-v2", self.endpoint());
        let headers = self.auth_headers();
        let response =
            retry::retry(self.retry_config(), || self.http_client().get(&url).headers(headers.clone()).send()).await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch the public profile of a user by Hub handle.
    ///
    /// Endpoint: `GET /api/users/{username}/overview`.
    ///
    /// # Parameters
    ///
    /// - `username` (required): Hub handle (slug) of the user.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn user_overview(
        &self,
        /// Hub handle (slug) of the user.
        username: &str,
    ) -> HFResult<User> {
        let url = format!("{}/api/users/{}/overview", self.endpoint(), username);
        let headers = self.auth_headers();
        let response =
            retry::retry(self.retry_config(), || self.http_client().get(&url).headers(headers.clone()).send()).await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch the public profile of an organization by Hub handle.
    ///
    /// Endpoint: `GET /api/organizations/{organization}/overview`.
    ///
    /// # Parameters
    ///
    /// - `organization` (required): Hub handle (slug) of the organization.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn organization_overview(
        &self,
        /// Hub handle (slug) of the organization.
        organization: &str,
    ) -> HFResult<Organization> {
        let url = format!("{}/api/organizations/{}/overview", self.endpoint(), organization);
        let headers = self.auth_headers();
        let response =
            retry::retry(self.retry_config(), || self.http_client().get(&url).headers(headers.clone()).send()).await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Stream the followers of a user.
    ///
    /// Endpoint: `GET /api/users/{username}/followers`.
    ///
    /// # Parameters
    ///
    /// - `username` (required): Hub handle of the user.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_followers(
        &self,
        /// Hub handle of the user.
        #[builder(into)]
        username: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!("{}/api/users/{}/followers", self.endpoint(), username))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the users that a user is following.
    ///
    /// Endpoint: `GET /api/users/{username}/following`.
    ///
    /// # Parameters
    ///
    /// - `username` (required): Hub handle of the user.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_following(
        &self,
        /// Hub handle of the user.
        #[builder(into)]
        username: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!("{}/api/users/{}/following", self.endpoint(), username))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the repositories a user has liked.
    ///
    /// Endpoint: `GET /api/users/{username}/likes`.
    ///
    /// # Parameters
    ///
    /// - `username` (required): Hub handle of the user.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_likes(
        &self,
        /// Hub handle of the user.
        #[builder(into)]
        username: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<LikedRepoEntry>> + '_> {
        let url = Url::parse(&format!("{}/api/users/{}/likes", self.endpoint(), username))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the members of an organization.
    ///
    /// Endpoint: `GET /api/organizations/{organization}/members`.
    ///
    /// # Parameters
    ///
    /// - `organization` (required): Hub handle of the organization.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_organization_members(
        &self,
        /// Hub handle of the organization.
        #[builder(into)]
        organization: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!("{}/api/organizations/{}/members", self.endpoint(), organization))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the followers of an organization.
    ///
    /// Endpoint: `GET /api/organizations/{organization}/followers`.
    ///
    /// # Parameters
    ///
    /// - `organization` (required): Hub handle of the organization.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_organization_followers(
        &self,
        /// Hub handle of the organization.
        #[builder(into)]
        organization: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!("{}/api/organizations/{}/followers", self.endpoint(), organization))?;
        Ok(self.paginate(url, vec![], limit))
    }
}

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::whoami`].
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn whoami(&self) -> HFResult<User> {
        self.runtime.block_on(self.inner.whoami().send())
    }

    /// Blocking counterpart of [`HFClient::user_overview`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn user_overview(&self, username: &str) -> HFResult<User> {
        self.runtime.block_on(self.inner.user_overview().username(username).send())
    }

    /// Blocking counterpart of [`HFClient::organization_overview`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn organization_overview(&self, organization: &str) -> HFResult<Organization> {
        self.runtime
            .block_on(self.inner.organization_overview().organization(organization).send())
    }

    /// Blocking counterpart of [`HFClient::list_user_followers`]. Collects the stream into a
    /// `Vec<User>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_followers(&self, #[builder(into)] username: String, limit: Option<usize>) -> HFResult<Vec<User>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self.inner.list_user_followers().username(username).maybe_limit(limit).send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_user_following`]. Collects the stream into a
    /// `Vec<User>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_following(&self, #[builder(into)] username: String, limit: Option<usize>) -> HFResult<Vec<User>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self.inner.list_user_following().username(username).maybe_limit(limit).send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_user_likes`]. Collects the stream into a
    /// `Vec<LikedRepoEntry>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_likes(
        &self,
        #[builder(into)] username: String,
        limit: Option<usize>,
    ) -> HFResult<Vec<LikedRepoEntry>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self.inner.list_user_likes().username(username).maybe_limit(limit).send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_organization_members`]. Collects the stream into a
    /// `Vec<User>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_organization_members(
        &self,
        #[builder(into)] organization: String,
        limit: Option<usize>,
    ) -> HFResult<Vec<User>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_organization_members()
                .organization(organization)
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

    /// Blocking counterpart of [`HFClient::list_organization_followers`]. Collects the stream into a
    /// `Vec<User>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_organization_followers(
        &self,
        #[builder(into)] organization: String,
        limit: Option<usize>,
    ) -> HFResult<Vec<User>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_organization_followers()
                .organization(organization)
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

    use super::{LikedRepoEntry, OrgMembership, Organization, User};
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

    #[test]
    fn test_user_full_profile() {
        let json = r#"{
            "user":"alice",
            "fullname":"Alice Anderson",
            "avatarUrl":"https://example/a.png",
            "type":"user",
            "details":"researcher",
            "isFollowing":true,
            "isPro":false,
            "numModels":12,
            "numDatasets":3,
            "numSpaces":1,
            "numDiscussions":4,
            "numPapers":2,
            "numUpvotes":5,
            "numLikes":6,
            "numFollowing":7,
            "numFollowers":8
        }"#;
        let user: User = serde_json::from_str(json).unwrap();
        assert_eq!(user.username, "alice");
        assert_eq!(user.details.as_deref(), Some("researcher"));
        assert_eq!(user.is_following, Some(true));
        assert_eq!(user.num_models, Some(12));
        assert_eq!(user.num_datasets, Some(3));
        assert_eq!(user.num_spaces, Some(1));
        assert_eq!(user.num_discussions, Some(4));
        assert_eq!(user.num_papers, Some(2));
        assert_eq!(user.num_upvotes, Some(5));
        assert_eq!(user.num_likes, Some(6));
        assert_eq!(user.num_following, Some(7));
        assert_eq!(user.num_followers, Some(8));
    }

    #[test]
    fn test_organization_full_profile() {
        let json = r#"{
            "name":"acme",
            "fullname":"Acme Corp",
            "avatarUrl":"https://example/o.png",
            "type":"org",
            "details":"description",
            "isVerified":true,
            "isFollowing":false,
            "numUsers":42,
            "numModels":7,
            "numSpaces":2,
            "numDatasets":3,
            "numFollowers":100,
            "numPapers":5,
            "plan":"enterprise"
        }"#;
        let org: Organization = serde_json::from_str(json).unwrap();
        assert_eq!(org.name, "acme");
        assert_eq!(org.details.as_deref(), Some("description"));
        assert_eq!(org.is_verified, Some(true));
        assert_eq!(org.is_following, Some(false));
        assert_eq!(org.num_users, Some(42));
        assert_eq!(org.num_models, Some(7));
        assert_eq!(org.num_spaces, Some(2));
        assert_eq!(org.num_datasets, Some(3));
        assert_eq!(org.num_followers, Some(100));
        assert_eq!(org.num_papers, Some(5));
        assert_eq!(org.plan.as_deref(), Some("enterprise"));
    }

    #[test]
    fn test_whoami_auth_block() {
        let json = r#"{
            "name":"alice",
            "auth":{
                "accessToken":{
                    "displayName":"laptop",
                    "role":"fineGrained",
                    "createdAt":"2026-01-02T03:04:05.000Z",
                    "fineGrained":{"scoped":[{"entity":{"type":"user","name":"alice"}}]}
                }
            }
        }"#;
        let user: User = serde_json::from_str(json).unwrap();
        let token = user.auth.unwrap().access_token.unwrap();
        assert_eq!(token.display_name.as_deref(), Some("laptop"));
        assert_eq!(token.role.as_deref(), Some("fineGrained"));
        assert_eq!(token.created_at.as_deref(), Some("2026-01-02T03:04:05.000Z"));
        assert!(token.fine_grained.is_some());
    }

    #[test]
    fn test_whoami_auth_block_absent_and_partial() {
        let user: User = serde_json::from_str(r#"{"name":"alice"}"#).unwrap();
        assert!(user.auth.is_none());

        let user: User = serde_json::from_str(r#"{"name":"alice","auth":{}}"#).unwrap();
        assert!(user.auth.unwrap().access_token.is_none());

        let user: User = serde_json::from_str(r#"{"name":"alice","auth":{"accessToken":{"role":"read"}}}"#).unwrap();
        let token = user.auth.unwrap().access_token.unwrap();
        assert_eq!(token.role.as_deref(), Some("read"));
        assert!(token.display_name.is_none());
        assert!(token.fine_grained.is_none());
    }

    #[test]
    fn test_org_membership_role_in_org() {
        let json = r#"{"name":"acme","fullname":"Acme Corp","roleInOrg":"admin"}"#;
        let membership: OrgMembership = serde_json::from_str(json).unwrap();
        assert_eq!(membership.name.as_deref(), Some("acme"));
        assert_eq!(membership.role_in_org.as_deref(), Some("admin"));

        let membership: OrgMembership = serde_json::from_str(r#"{"name":"acme"}"#).unwrap();
        assert!(membership.role_in_org.is_none());
    }

    #[test]
    fn test_liked_repo_entry_deserialization() {
        let json = r#"{"createdAt":"2026-01-02T03:04:05.000Z","repo":{"name":"acme/thing","type":"dataset"}}"#;
        let entry: LikedRepoEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.created_at.as_deref(), Some("2026-01-02T03:04:05.000Z"));
        assert_eq!(entry.repo.name, "acme/thing");
        assert_eq!(entry.repo.repo_type, "dataset");

        let entry: LikedRepoEntry = serde_json::from_str(r#"{"repo":{"name":"o/m","type":"model"}}"#).unwrap();
        assert!(entry.created_at.is_none());
    }

    #[tokio::test]
    async fn list_user_likes_streams_entries_and_honors_limit() {
        let body = r#"[
            {"createdAt":"2026-01-02T03:04:05.000Z","repo":{"name":"acme/thing","type":"dataset"}},
            {"createdAt":"2026-01-03T03:04:05.000Z","repo":{"name":"o/m","type":"model"}},
            {"createdAt":"2026-01-04T03:04:05.000Z","repo":{"name":"o/s","type":"space"}}
        ]"#;
        let (endpoint, server) = serve_one_json(body).await;

        let client = HFClient::builder().endpoint(endpoint).build().unwrap();
        let stream = client.list_user_likes().username("alice").limit(2_usize).send().unwrap();
        futures::pin_mut!(stream);
        let mut entries = Vec::new();
        while let Some(entry) = stream.next().await {
            entries.push(entry.unwrap());
        }

        assert_eq!(server.await.unwrap(), "GET /api/users/alice/likes HTTP/1.1\r\n");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].repo.name, "acme/thing");
        assert_eq!(entries[0].repo.repo_type, "dataset");
        assert_eq!(entries[1].repo.name, "o/m");
    }

    #[tokio::test]
    async fn list_organization_followers_requests_the_followers_endpoint() {
        let (endpoint, server) = serve_one_json(r#"[{"user":"alice"},{"user":"bob"}]"#).await;

        let client = HFClient::builder().endpoint(endpoint).build().unwrap();
        let stream = client.list_organization_followers().organization("acme").send().unwrap();
        futures::pin_mut!(stream);
        let mut usernames = Vec::new();
        while let Some(user) = stream.next().await {
            usernames.push(user.unwrap().username);
        }

        assert_eq!(server.await.unwrap(), "GET /api/organizations/acme/followers HTTP/1.1\r\n");
        assert_eq!(usernames, ["alice", "bob"]);
    }
}
