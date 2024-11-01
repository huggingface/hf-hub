use std::time::Instant;

use hf_hub::{
    api::tokio::{ApiBuilder, ApiError},
    Repo,
};
use rand::Rng;

const ONE_MB: usize = 1024 * 1024;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let token = std::env::var("HF_TOKEN")
        .map_err(|_| "HF_TOKEN environment variable not set".to_string())?;
    let hf_repo = std::env::var("HF_REPO")
        .map_err(|_| "HF_REPO environment variable not set, e.g. apyh/gronk".to_string())?;

    let api = ApiBuilder::new().with_token(Some(token)).build()?;
    let repo = Repo::model(hf_repo);
    let api_repo = api.repo(repo);

    let exists = api_repo.exists().await;
    if !exists {
        return Err(ApiError::GatedRepoError("repo does not exist".to_string()).into());
    } else {
        println!("repo exists!");
    }

    let is_writable = api_repo.is_writable().await;
    if !is_writable {
        return Err(ApiError::GatedRepoError("repo is not writable".to_string()).into());
    } else {
        println!("repo is writable!");
    }
    let files = [
        (
            format!("im a tiny file {:?}", Instant::now())
                .as_bytes()
                .to_vec(),
            "tiny_file.txt",
        ),
        (
            {
                let mut data = vec![0u8; ONE_MB];
                rand::thread_rng().fill(&mut data[..]);
                data
            },
            "1m_file.txt",
        ),
        (
            {
                let mut data = vec![0u8; 10 * ONE_MB];
                rand::thread_rng().fill(&mut data[..]);
                data
            },
            "10m_file.txt",
        ),
        (
            {
                let mut data = vec![0u8; 20 * ONE_MB];
                rand::thread_rng().fill(&mut data[..]);
                data
            },
            "20m_file.txt",
        ),
    ];
    let res = api_repo
        .upload_files(
            files
                .into_iter()
                .map(|(data, path)| (data.into(), path.into()))
                .collect(),
            None,
            "update multiple files!".to_string().into(),
            false,
        )
        .await?;
    log::info!("{:?}", res);
    log::info!("Success!!");
    Ok(())
}
