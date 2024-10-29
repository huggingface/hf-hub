use std::time::Instant;

use hf_hub::{api::tokio::ApiBuilder, Repo};
use rand::Rng;

const ONE_MB: usize = 1024 * 1024;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let token =
        std::env::var("HF_TOKEN").map_err(|_| format!("HF_TOKEN environment variable not set"))?;
    let hf_repo = std::env::var("HF_REPO")
        .map_err(|_| format!("HF_REPO environment variable not set, e.g. apyh/gronk"))?;

    let api = ApiBuilder::new().with_token(Some(token)).build()?;
    let repo = Repo::model(hf_repo);
    let api_repo = api.repo(repo);
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
