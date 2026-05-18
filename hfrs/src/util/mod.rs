pub mod token;

use hf_hub::{HFClient, HFRepository, RepoTypeAny};

use crate::cli::RepoTypeArg;

/// Split a repo ID like "owner/name" into (owner, name).
/// If no slash is present, treats the whole string as the name with an empty owner.
pub fn split_repo_id(repo_id: &str) -> (&str, &str) {
    match repo_id.split_once('/') {
        Some((owner, name)) => (owner, name),
        None => ("", repo_id),
    }
}

/// Build an [`HFRepository<RepoTypeAny>`] from a CLI `--type` flag and a repo id string.
///
/// The handle carries the kind at runtime via [`RepoTypeAny`], so the file/commit/ref/settings
/// methods on it hit the right Hub endpoint without the call site needing to match per kind.
pub fn typed_repo(client: &HFClient, repo_id: &str, arg: RepoTypeArg) -> HFRepository<RepoTypeAny> {
    let (owner, name) = split_repo_id(repo_id);
    client.repository(arg.into(), owner, name)
}
