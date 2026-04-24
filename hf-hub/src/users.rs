//! Users and organizations: identity, profile lookup, and social listings.
//!
//! This module exposes the [`User`], [`OrgMembership`], and [`Organization`]
//! types and the corresponding [`HFClient`] methods:
//!
//! - [`HFClient::whoami`] / [`HFClient::auth_check`] — identify the caller and verify that the current token is valid.
//! - [`HFClient::get_user_overview`] / [`HFClient::get_organization_overview`] — fetch a public profile by username or
//!   organization name.
//! - [`HFClient::list_user_followers`] / [`HFClient::list_user_following`] / [`HFClient::list_organization_members`] —
//!   paginated listings that yield [`User`] entries one page at a time.

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
    /// Whether the user is on a Pro plan.
    pub is_pro: Option<bool>,
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
}

/// Summary entry for an organization the authenticated user belongs to.
///
/// Returned inside [`User::orgs`]. This is a lighter-weight shape than
/// [`Organization`] — use [`HFClient::get_organization_overview`] to fetch the
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
}

/// A Hugging Face Hub organization.
///
/// Returned by [`HFClient::get_organization_overview`].
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
}

impl HFClient {
    /// Fetch the profile of the user that owns the current token.
    ///
    /// Returns the authenticated [`User`], including private fields like
    /// [`email`](User::email) and the caller's [`orgs`](User::orgs) list.
    /// Fails with [`HFError::AuthRequired`](crate::HFError::AuthRequired) if no
    /// valid token is configured.
    ///
    /// Endpoint: `GET /api/whoami-v2`
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

    /// Verify that the current token is valid.
    ///
    /// Returns `Ok(())` if the token authenticates successfully, or
    /// [`HFError::AuthRequired`](crate::HFError::AuthRequired) if it is missing
    /// or rejected. Equivalent to calling [`whoami`](Self::whoami) and
    /// discarding the response.
    ///
    /// Endpoint: `GET /api/whoami-v2`
    pub async fn auth_check(&self) -> HFResult<()> {
        self.whoami().await?;
        Ok(())
    }

    /// Fetch the public profile of a user by Hub handle.
    ///
    /// Private fields such as [`email`](User::email) are never populated for
    /// other users; use [`whoami`](Self::whoami) to retrieve them for the
    /// authenticated caller.
    ///
    /// Endpoint: `GET /api/users/{username}/overview`
    pub async fn get_user_overview(&self, username: &str) -> HFResult<User> {
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
    /// Endpoint: `GET /api/organizations/{organization}/overview`
    pub async fn get_organization_overview(&self, organization: &str) -> HFResult<Organization> {
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
    /// `limit` caps the total number of items yielded across all pages; pass
    /// `None` to iterate until the API runs out. Use the
    /// [`futures::StreamExt`](https://docs.rs/futures) adapters to consume the
    /// stream.
    ///
    /// Endpoint: `GET /api/users/{username}/followers`
    pub fn list_user_followers(
        &self,
        username: &str,
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!("{}/api/users/{}/followers", self.endpoint(), username))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the users that a user is following.
    ///
    /// `limit` caps the total number of items yielded across all pages; pass
    /// `None` to iterate until the API runs out.
    ///
    /// Endpoint: `GET /api/users/{username}/following`
    pub fn list_user_following(
        &self,
        username: &str,
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!("{}/api/users/{}/following", self.endpoint(), username))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the members of an organization.
    ///
    /// `limit` caps the total number of items yielded across all pages; pass
    /// `None` to iterate until the API runs out. Visibility of members
    /// depends on the organization's settings and the caller's permissions.
    ///
    /// Endpoint: `GET /api/organizations/{organization}/members`
    pub fn list_organization_members(
        &self,
        organization: &str,
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!("{}/api/organizations/{}/members", self.endpoint(), organization))?;
        Ok(self.paginate(url, vec![], limit))
    }
}

sync_api! {
    impl HFClient -> HFClientSync {
        fn whoami(&self) -> HFResult<User>;
        fn auth_check(&self) -> HFResult<()>;
        fn get_user_overview(&self, username: &str) -> HFResult<User>;
        fn get_organization_overview(&self, organization: &str) -> HFResult<Organization>;
    }
}

sync_api_stream! {
    impl HFClient -> HFClientSync {
        fn list_user_followers(&self, username: &str, limit: Option<usize>) -> User;
        fn list_user_following(&self, username: &str, limit: Option<usize>) -> User;
        fn list_organization_members(&self, organization: &str, limit: Option<usize>) -> User;
    }
}
