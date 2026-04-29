//! Repository operations: info, listing, existence checks, and CRUD.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example repo

use futures::StreamExt;
use hf_hub::{HFClient, RepoTypeModel};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    // --- Read operations ---

    let model = client.model("openai-community", "gpt2");
    let info = model.info().send().await?;
    println!("Model: {} (downloads: {:?})", info.id, info.downloads);

    let dataset = client.dataset("rajpurkar", "squad");
    let info = dataset.info().send().await?;
    println!("Dataset: {} (downloads: {:?})", info.id, info.downloads);

    let space = client.space("huggingface", "transformers-benchmarks");
    let info = space.info().send().await?;
    println!("Space: {} (sdk: {:?})", info.id, info.sdk);

    let exists = model.exists().send().await?;
    println!("gpt2 exists: {exists}");

    let rev_exists = model.revision_exists().revision("main").send().await?;
    println!("gpt2@main exists: {rev_exists}");

    let file_exists = model.file_exists().filename("config.json").send().await?;
    println!("gpt2/config.json exists: {file_exists}");

    let models_stream = client.list_models().author("openai").send()?;
    futures::pin_mut!(models_stream);
    println!("\nModels by openai:");
    let mut count = 0;
    while let Some(Ok(model)) = models_stream.next().await {
        println!("  - {}", model.id);
        count += 1;
        if count >= 3 {
            break;
        }
    }

    let datasets_stream = client.list_datasets().search("squad").send()?;
    futures::pin_mut!(datasets_stream);
    println!("\nDatasets matching 'squad':");
    let mut count = 0;
    while let Some(Ok(ds)) = datasets_stream.next().await {
        println!("  - {}", ds.id);
        count += 1;
        if count >= 3 {
            break;
        }
    }

    let spaces_stream = client.list_spaces().author("huggingface").send()?;
    futures::pin_mut!(spaces_stream);
    println!("\nSpaces by huggingface:");
    let mut count = 0;
    while let Some(Ok(sp)) = spaces_stream.next().await {
        println!("  - {}", sp.id);
        count += 1;
        if count >= 3 {
            break;
        }
    }

    // --- Write operations (creates real resources on the Hub) ---

    let user = client.whoami().send().await?;
    let unique = std::process::id();
    let repo = client.model(&user.username, format!("example-repo-{unique}"));

    let repo_url = client
        .create_repo::<RepoTypeModel>()
        .repo_id(repo.repo_path())
        .private(true)
        .exist_ok(true)
        .send()
        .await?;
    println!("\nCreated repo: {}", repo_url.url);

    repo.update_settings().description("Temporary example repo").send().await?;
    println!("Updated repo description");

    let new_name = format!("{}/example-repo-renamed-{unique}", user.username);
    let moved = client
        .move_repo::<RepoTypeModel>()
        .from_id(repo.repo_path())
        .to_id(&new_name)
        .send()
        .await?;
    println!("Moved repo to: {}", moved.url);

    client
        .delete_repo::<RepoTypeModel>()
        .repo_id(&new_name)
        .missing_ok(true)
        .send()
        .await?;
    println!("Deleted repo");

    Ok(())
}
