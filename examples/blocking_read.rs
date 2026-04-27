//! Synchronous read operations using HFClientSync.
//!
//! Demonstrates repo info, file listing, downloads, user info, and
//! paginated endpoints — all without an async runtime.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --features blocking --example blocking_read

use hf_hub::HFClientSync;
use hf_hub::repository::{RepoInfo, RepoTreeEntry};

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;

    // --- Repo info ---

    let model = client.model("openai-community", "gpt2");
    let dataset = client.dataset("rajpurkar", "squad");
    let space = client.space("huggingface", "transformers-benchmarks");

    match model.info().send()? {
        RepoInfo::Model(info) => println!("Model: {} (downloads: {:?})", info.id, info.downloads),
        _ => unreachable!(),
    }

    match dataset.info().send()? {
        RepoInfo::Dataset(info) => println!("Dataset: {} (downloads: {:?})", info.id, info.downloads),
        _ => unreachable!(),
    }

    match space.info().send()? {
        RepoInfo::Space(info) => println!("Space: {} (sdk: {:?})", info.id, info.sdk),
        _ => unreachable!(),
    }

    let exists = model.exists().send()?;
    println!("gpt2 exists: {exists}");

    // --- Listing (streams collected to Vec) ---

    let models = client.list_models().author("openai").send()?;
    println!("\nModels by openai ({} total):", models.len());
    for m in models.iter().take(3) {
        println!("  - {}", m.id);
    }

    let datasets = client.list_datasets().search("squad").send()?;
    println!("\nDatasets matching 'squad' ({} total):", datasets.len());
    for ds in datasets.iter().take(3) {
        println!("  - {}", ds.id);
    }

    // --- Files ---

    let files = model.list_files().send()?;
    println!("\nFiles in gpt2: {}", files.len());
    for f in files.iter().take(5) {
        println!("  - {f}");
    }

    let tree = model.list_tree().recursive(true).send()?;
    println!("\nTree entries in gpt2:");
    for entry in tree.iter().take(5) {
        match entry {
            RepoTreeEntry::File { path, size, .. } => println!("  file: {path} ({size} bytes)"),
            RepoTreeEntry::Directory { path, .. } => println!("  dir:  {path}"),
        }
    }

    let paths_info = model
        .get_paths_info()
        .paths(vec!["config.json".to_string(), "README.md".to_string()])
        .send()?;
    println!("\nPaths info ({} entries):", paths_info.len());

    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");
    let downloaded = model
        .download_file()
        .filename("config.json")
        .local_dir(tmp_dir.path().to_path_buf())
        .send()?;
    println!("\nDownloaded gpt2/config.json to: {}", downloaded.display());

    // --- Commits ---

    let commits = model.list_commits().send()?;
    println!("\nRecent commits on gpt2 ({} total):", commits.len());
    for c in commits.iter().take(3) {
        println!("  - {} {}", &c.id[..8], c.title);
    }

    // --- Users ---

    let me = client.whoami().send()?;
    println!("\nLogged in as: {}", me.username);

    let user = client.get_user_overview().username("julien-c").send()?;
    println!("User: {} (fullname: {:?})", user.username, user.fullname);

    let org = client.get_organization_overview().organization("huggingface").send()?;
    println!("Org: {} (fullname: {:?})", org.name, org.fullname);

    let followers = client.list_user_followers().username("julien-c").send()?;
    println!("\nFollowers of julien-c ({} total):", followers.len());
    for u in followers.iter().take(3) {
        println!("  - {}", u.username);
    }

    let members = client.list_organization_members().organization("huggingface").send()?;
    println!("\nMembers of huggingface ({} total):", members.len());
    for m in members.iter().take(3) {
        println!("  - {}", m.username);
    }

    Ok(())
}
