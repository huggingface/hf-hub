pub mod token;

/// Split a repo ID like "owner/name" into (owner, name).
/// If no slash is present, treats the whole string as the name with an empty owner.
pub fn split_repo_id(repo_id: &str) -> (&str, &str) {
    match repo_id.split_once('/') {
        Some((owner, name)) => (owner, name),
        None => ("", repo_id),
    }
}

/// Match on a [`crate::cli::RepoTypeArg`] and run `$body` with `$repo` bound to a typed
/// `HFRepository<T>` for each arm. The body must compile generically over `T: RepoType`.
///
/// Used at CLI dispatch sites to convert the user-supplied `--type model|dataset|space`
/// flag into a concrete typed repo handle without each command repeating the match.
#[macro_export]
macro_rules! with_typed_repo {
    ($client:expr, $repo_id:expr, $arg:expr, |$repo:ident| $body:expr) => {{
        let (owner, name) = $crate::util::split_repo_id($repo_id);
        match $arg {
            $crate::cli::RepoTypeArg::Model => {
                let $repo = $client.repository::<hf_hub::RepoTypeModel>(owner, name);
                $body
            },
            $crate::cli::RepoTypeArg::Dataset => {
                let $repo = $client.repository::<hf_hub::RepoTypeDataset>(owner, name);
                $body
            },
            $crate::cli::RepoTypeArg::Space => {
                let $repo = $client.repository::<hf_hub::RepoTypeSpace>(owner, name);
                $body
            },
        }
    }};
}

/// Match on a [`crate::cli::RepoTypeArg`] and call a method that is generic over the repo
/// kind via turbofish. Used for client-level methods like `create_repo::<T>()` and
/// `delete_repo::<T>()` which don't take a typed repo handle.
///
/// Expands to `match $arg { Model => $body::<RepoTypeModel>, ... }`.
#[macro_export]
macro_rules! dispatch_repo_type {
    ($arg:expr, $body:ident) => {
        match $arg {
            $crate::cli::RepoTypeArg::Model => $body::<hf_hub::RepoTypeModel>(),
            $crate::cli::RepoTypeArg::Dataset => $body::<hf_hub::RepoTypeDataset>(),
            $crate::cli::RepoTypeArg::Space => $body::<hf_hub::RepoTypeSpace>(),
        }
    };
}
