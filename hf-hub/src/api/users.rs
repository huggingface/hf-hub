use futures::Stream;
use url::Url;

use crate::client::HFClient;
use crate::error::Result;
use crate::types::{Organization, User};

impl HFClient {
    /// Get authenticated user info.
    /// Endpoint: GET /api/whoami-v2
    pub async fn whoami(&self) -> Result<User> {
        let url = format!("{}/api/whoami-v2", self.endpoint());
        let response = self.http_client().get(&url).headers(self.auth_headers()).send().await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Check if the current token is valid.
    /// Endpoint: GET /api/whoami-v2
    /// Returns Ok(()) on success, Err(AuthRequired) if invalid.
    pub async fn auth_check(&self) -> Result<()> {
        self.whoami().await?;
        Ok(())
    }

    /// Get overview of a user.
    /// Endpoint: GET /api/users/{username}/overview
    pub async fn get_user_overview(&self, username: &str) -> Result<User> {
        let url = format!("{}/api/users/{}/overview", self.endpoint(), username);
        let response = self.http_client().get(&url).headers(self.auth_headers()).send().await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Get overview of an organization.
    /// Endpoint: GET /api/organizations/{organization}/overview
    pub async fn get_organization_overview(&self, organization: &str) -> Result<Organization> {
        let url = format!("{}/api/organizations/{}/overview", self.endpoint(), organization);
        let response = self.http_client().get(&url).headers(self.auth_headers()).send().await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// List followers of a user.
    /// Endpoint: GET /api/users/{username}/followers
    pub fn list_user_followers(
        &self,
        username: &str,
        limit: Option<usize>,
    ) -> Result<impl Stream<Item = Result<User>> + '_> {
        let url = Url::parse(&format!("{}/api/users/{}/followers", self.endpoint(), username))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// List users that a user is following.
    /// Endpoint: GET /api/users/{username}/following
    pub fn list_user_following(
        &self,
        username: &str,
        limit: Option<usize>,
    ) -> Result<impl Stream<Item = Result<User>> + '_> {
        let url = Url::parse(&format!("{}/api/users/{}/following", self.endpoint(), username))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// List members of an organization.
    /// Endpoint: GET /api/organizations/{organization}/members
    pub fn list_organization_members(
        &self,
        organization: &str,
        limit: Option<usize>,
    ) -> Result<impl Stream<Item = Result<User>> + '_> {
        let url = Url::parse(&format!("{}/api/organizations/{}/members", self.endpoint(), organization))?;
        Ok(self.paginate(url, vec![], limit))
    }
}

sync_api! {
    impl HFClient -> HFClientSync {
        fn whoami(&self) -> Result<User>;
        fn auth_check(&self) -> Result<()>;
        fn get_user_overview(&self, username: &str) -> Result<User>;
        fn get_organization_overview(&self, organization: &str) -> Result<Organization>;
    }
}

sync_api_stream! {
    impl HFClient -> HFClientSync {
        fn list_user_followers(&self, username: &str, limit: Option<usize>) -> User;
        fn list_user_following(&self, username: &str, limit: Option<usize>) -> User;
        fn list_organization_members(&self, organization: &str, limit: Option<usize>) -> User;
    }
}
