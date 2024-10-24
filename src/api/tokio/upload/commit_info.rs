use std::{error::Error, fmt, num::ParseIntError};

use crate::RepoType;
use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug)]
pub struct RepoUrl {
    pub endpoint: String,
    pub namespace: Option<String>,
    pub repo_name: String,
    pub repo_id: String,
    pub repo_type: Option<RepoType>,
    pub url: String,
}

const HF_DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const HF_DEFAULT_STAGING_ENDPOINT: &str = "https://hub-ci.huggingface.co";

impl RepoUrl {
    pub fn new(url: &str) -> Result<Self, InvalidHfIdError> {
        Self::new_with_endpoint(url, HF_DEFAULT_ENDPOINT)
    }
    pub fn new_with_endpoint(url: &str, endpoint: &str) -> Result<Self, InvalidHfIdError> {
        let url = fix_hf_endpoint_in_url(url, endpoint);
        let (repo_type, namespace, repo_name) =
            repo_type_and_name_from_hf_id(&url, Some(endpoint))?;
        let repo_id = if let Some(ns) = &namespace {
            format!("{ns}/{repo_name}")
        } else {
            repo_name.clone()
        };

        Ok(Self {
            url,
            endpoint: endpoint.into(),
            namespace,
            repo_id,
            repo_type,
            repo_name,
        })
    }
}

#[derive(Debug)]
pub struct InvalidHfIdError(String);

impl fmt::Display for InvalidHfIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unable to retrieve user and repo ID from the passed HF ID: {}",
            self.0
        )
    }
}

impl Error for InvalidHfIdError {}

/// Returns the repo type and ID from a huggingface.co URL linking to a repository
///
/// # Arguments
///
/// * `hf_id` - An URL or ID of a repository on the HF hub. Accepted values are:
///   - https://huggingface.co/<repo_type>/<namespace>/<repo_id>
///   - https://huggingface.co/<namespace>/<repo_id>
///   - hf://<repo_type>/<namespace>/<repo_id>
///   - hf://<namespace>/<repo_id>
///   - <repo_type>/<namespace>/<repo_id>
///   - <namespace>/<repo_id>
///   - <repo_id>
/// * `hub_url` - The URL of the HuggingFace Hub, defaults to https://huggingface.co
///
/// # Returns
///
/// A tuple with three items: (repo_type, namespace, repo_name)
fn repo_type_and_name_from_hf_id(
    hf_id: &str,
    hub_url: Option<&str>,
) -> Result<(Option<RepoType>, Option<String>, String), InvalidHfIdError> {
    let hub_url = hub_url.unwrap_or(HF_DEFAULT_ENDPOINT);
    let hub_url = Regex::new(r"https?://")
        .unwrap()
        .replace(hub_url, "")
        .into_owned();

    let is_hf_url = hf_id.contains(&hub_url) && !hf_id.contains('@');

    const HFFS_PREFIX: &str = "hf://";
    let hf_id = hf_id.strip_prefix(HFFS_PREFIX).unwrap_or(hf_id);

    let url_segments: Vec<&str> = hf_id.split('/').collect();
    let is_hf_id = url_segments.len() <= 3;

    let (repo_type, namespace, repo_id) = if is_hf_url {
        let (namespace, repo_id) = (
            url_segments[url_segments.len() - 2],
            url_segments.last().unwrap(),
        );
        let namespace = if namespace == hub_url {
            None
        } else {
            Some(namespace.to_string())
        };

        let repo_type: Option<RepoType> =
            if url_segments.len() > 2 && !url_segments[url_segments.len() - 3].contains(&hub_url) {
                url_segments[url_segments.len() - 3]
                    .to_string()
                    .parse()
                    .ok()
            } else {
                namespace
                    .clone()
                    .unwrap_or("".to_string())
                    .parse::<RepoType>()
                    .ok()
            };

        (repo_type, namespace, repo_id.to_string())
    } else if is_hf_id {
        match url_segments.len() {
            3 => {
                let (repo_type, namespace, repo_id) = (
                    url_segments[0].parse().ok(),
                    Some(url_segments[1].to_string()),
                    url_segments[2].to_string(),
                );
                (repo_type, namespace, repo_id)
            }
            2 => {
                if let Ok(repo_type) = url_segments[0].parse() {
                    (Some(repo_type), None, url_segments[1].to_string())
                } else {
                    (
                        None,
                        Some(url_segments[0].to_string()),
                        url_segments[1].to_string(),
                    )
                }
            }
            1 => (None, None, url_segments[0].to_string()),
            _ => return Err(InvalidHfIdError(hf_id.to_string())),
        }
    } else {
        return Err(InvalidHfIdError(hf_id.to_string()));
    };

    Ok((repo_type, namespace, repo_id))
}

/// Replace the default endpoint in a URL by a custom one.
/// This is useful when using a proxy and the Hugging Face Hub returns a URL with the default endpoint.
pub fn fix_hf_endpoint_in_url(url: &str, endpoint: &str) -> String {
    // check if a proxy has been set => if yes, update the returned URL to use the proxy
    let mut url = url.to_string();
    if endpoint != HF_DEFAULT_ENDPOINT {
        url = url.replace(HF_DEFAULT_ENDPOINT, endpoint);
    } else if endpoint != HF_DEFAULT_STAGING_ENDPOINT {
        url = url.replace(HF_DEFAULT_STAGING_ENDPOINT, endpoint);
    }
    url
}

/// Data structure containing information about a newly created commit.
/// Returned by any method that creates a commit on the Hub.
#[derive(Debug)]
pub struct CommitInfo {
    /// Url where to find the commit.
    pub commit_url: String,
    /// The summary (first line) of the commit that has been created.
    pub commit_message: String,
    ///  Description of the commit that has been created. Can be empty.
    pub commit_description: String,
    ///  Commit hash id. Example: `"91c54ad1727ee830252e457677f467be0bfd8a57"`.
    pub oid: String,

    /// Repo URL of the commit containing info like repo_id, repo_type, etc.
    pub repo_url: RepoUrl,

    // Info about the associated pull request
    pub pull_request: Option<PullRequestInfo>,
}
#[derive(Debug)]
pub struct PullRequestInfo {
    pub url: String,
    pub revision: String,
    pub num: u32,
}

impl PullRequestInfo {
    fn new(pr_url: &str) -> Result<Self, ParseIntError> {
        let pr_revision = parse_revision_from_pr_url(pr_url);
        let pr_num: u32 = pr_revision.split("/").last().unwrap().parse()?;
        Ok(PullRequestInfo {
            num: pr_num,
            revision: pr_revision,
            url: pr_url.into(),
        })
    }
}

lazy_static! {
    static ref REGEX_DISCUSSION_URL: Regex = Regex::new(r".*/discussions/(\d+)$").unwrap();
}

/// Safely parse revision number from a PR url.
/// # Example
/// ```
///    assert_eq!(parse_revision_from_pr_url("https://huggingface.co/bigscience/bloom/discussions/2"), "refs/pr/2");
/// ```
fn parse_revision_from_pr_url(pr_url: &str) -> String {
    let re_match = REGEX_DISCUSSION_URL.captures(pr_url).unwrap_or_else(|| {
        panic!(
            "Unexpected response from the hub, expected a Pull Request URL but got: '{}'",
            pr_url
        )
    });

    format!("refs/pr/{}", &re_match[1])
}

impl CommitInfo {
    pub fn new(
        url: &str,
        commit_description: &str,
        commit_message: &str,
        oid: String,
    ) -> Result<Self, InvalidHfIdError> {
        Ok(Self {
            commit_url: url.into(),
            commit_description: commit_description.into(),
            commit_message: commit_message.into(),
            oid,
            pull_request: None,
            repo_url: RepoUrl::new(url)?,
        })
    }

    pub fn set_pr_info(&mut self, pr_url: &str) -> Result<(), ParseIntError> {
        self.pull_request = Some(PullRequestInfo::new(pr_url)?);
        Ok(())
    }
}
