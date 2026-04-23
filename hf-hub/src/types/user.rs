use serde::Deserialize;

/// A Hugging Face Hub user account.
///
/// Returned by [`HFClient::whoami`](crate::client::HFClient::whoami) and user-lookup methods.
/// Fields beyond `username` are only populated for authenticated `whoami` calls or when the
/// caller has permission to see them.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct User {
    #[serde(alias = "login", alias = "user", alias = "name")]
    pub username: String,
    pub fullname: Option<String>,
    pub avatar_url: Option<String>,
    #[serde(rename = "type")]
    pub user_type: Option<String>,
    pub is_pro: Option<bool>,
    pub email: Option<String>,
    pub email_verified: Option<bool>,
    pub plan: Option<String>,
    pub can_pay: Option<bool>,
    pub orgs: Option<Vec<OrgMembership>>,
}

/// Summary entry for an organization the authenticated user belongs to.
///
/// Attached to [`User::orgs`] — a lighter-weight shape than [`Organization`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrgMembership {
    pub name: Option<String>,
    pub fullname: Option<String>,
    pub avatar_url: Option<String>,
}

/// A Hugging Face Hub organization.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Organization {
    pub name: String,
    pub fullname: Option<String>,
    pub avatar_url: Option<String>,
    #[serde(rename = "type")]
    pub org_type: Option<String>,
}
