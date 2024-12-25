#[tokio::main]
async fn main() {
    let _proxy = std::env::var("HTTPS_PROXY").expect("This example expects a HTTPS_PROXY environment variable to be defined to test that the routing happens correctly. Starts a socks servers and use point HTTPS_PROXY to that server to see the routing in action.");

    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_progress(true)
        .build()
        .unwrap();

    let _filename = api
        .model("meta-llama/Llama-2-7b-hf".to_string())
        .get("model-00001-of-00002.safetensors")
        .await
        .unwrap();
}
