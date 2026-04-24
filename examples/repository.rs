//! Repository operations: info, listing, existence checks, and CRUD.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example repo

use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::repository::{
    CreateRepoParams, DeleteRepoParams, ListDatasetsParams, ListModelsParams, ListSpacesParams, MoveRepoParams,
    RepoFileExistsParams, RepoInfo, RepoInfoParams, RepoRevisionExistsParams, RepoUpdateSettingsParams,
};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    // --- Read operations ---

    let model = client.model("openai-community", "gpt2");
    match model.info(RepoInfoParams::default()).await? {
        RepoInfo::Model(info) => println!("Model: {} (downloads: {:?})", info.id, info.downloads),
        _ => unreachable!(),
    }

    let dataset = client.dataset("rajpurkar", "squad");
    match dataset.info(RepoInfoParams::default()).await? {
        RepoInfo::Dataset(info) => println!("Dataset: {} (downloads: {:?})", info.id, info.downloads),
        _ => unreachable!(),
    }

    let space = client.space("huggingface", "transformers-benchmarks");
    match space.info(RepoInfoParams::default()).await? {
        RepoInfo::Space(info) => println!("Space: {} (sdk: {:?})", info.id, info.sdk),
        _ => unreachable!(),
    }

    let exists = model.exists().await?;
    println!("gpt2 exists: {exists}");

    let rev_exists = model
        .revision_exists(RepoRevisionExistsParams::builder().revision("main").build())
        .await?;
    println!("gpt2@main exists: {rev_exists}");

    let file_exists = model
        .file_exists(RepoFileExistsParams::builder().filename("config.json").build())
        .await?;
    println!("gpt2/config.json exists: {file_exists}");

    let models_stream = client.list_models(ListModelsParams::builder().author("openai").build())?;
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

    let datasets_stream = client.list_datasets(ListDatasetsParams::builder().search("squad").build())?;
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

    let spaces_stream = client.list_spaces(ListSpacesParams::builder().author("huggingface").build())?;
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

    let user = client.whoami().await?;
    let unique = std::process::id();
    let repo = client.model(&user.username, format!("example-repo-{unique}"));

    let repo_url = client
        .create_repo(
            CreateRepoParams::builder()
                .repo_id(repo.repo_path())
                .private(true)
                .exist_ok(true)
                .build(),
        )
        .await?;
    println!("\nCreated repo: {}", repo_url.url);

    repo.update_settings(
        RepoUpdateSettingsParams::builder()
            .description("Temporary example repo")
            .build(),
    )
    .await?;
    println!("Updated repo description");

    let new_name = format!("{}/example-repo-renamed-{unique}", user.username);
    let moved = client
        .move_repo(MoveRepoParams::builder().from_id(repo.repo_path()).to_id(&new_name).build())
        .await?;
    println!("Moved repo to: {}", moved.url);

    client
        .delete_repo(DeleteRepoParams::builder().repo_id(&new_name).missing_ok(true).build())
        .await?;
    println!("Deleted repo");

    Ok(())
}
