pub mod token;

use hf_hub::{HFClient, HFRepository, RepoType};

/// Split a repo ID like "owner/name" into (owner, name).
/// If no slash is present, treats the whole string as the name with an empty owner.
pub fn split_repo_id(repo_id: &str) -> (&str, &str) {
    match repo_id.split_once('/') {
        Some((owner, name)) => (owner, name),
        None => ("", repo_id),
    }
}

/// Create an HFRepository handle from an client client, repo_id string, and repo_type.
pub fn make_repo(client: &HFClient, repo_id: &str, repo_type: RepoType) -> HFRepository {
    let (owner, name) = split_repo_id(repo_id);
    client.repo(repo_type, owner, name)
}
