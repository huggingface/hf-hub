use hf_hub::{Repo, RepoType};

#[cfg(not(feature = "tokio"))]
fn main() {
    let api = hf_hub::api::sync::Api::new().unwrap();
    let repo = Repo::new("meta-llama/Llama-2-7b-hf".to_string(), RepoType::Model);

    let _filename = api.get(&repo, "model-00001-of-00002.safetensors").unwrap();
}

#[cfg(feature = "tokio")]
#[tokio::main]
async fn main() {
    let api = hf_hub::api::tokio::Api::new().unwrap();
    let repo = Repo::new("meta-llama/Llama-2-7b-hf".to_string(), RepoType::Model);

    let _filename = api
        .get(&repo, "model-00001-of-00002.safetensors")
        .await
        .unwrap();
}
