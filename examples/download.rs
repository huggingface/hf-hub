#[cfg(not(feature = "tokio"))]
fn main() {
    let api = hf_hub::api::sync::Api::new().unwrap();

    let _filename = api
        .model("meta-llama/Llama-2-7b-hf".to_string())
        .get("model-00001-of-00002.safetensors")
        .unwrap();
}

#[cfg(feature = "tokio")]
#[tokio::main]
async fn main() {
    let api = hf_hub::api::tokio::Api::new().unwrap();

    let _filename = api
        .model("meta-llama/Llama-2-7b-hf".to_string())
        .get("model-00001-of-00002.safetensors")
        .await
        .unwrap();
}
