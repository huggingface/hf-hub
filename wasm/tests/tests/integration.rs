//! wasm32-unknown-unknown integration tests for `hf-hub`.
//!
//! These run against the live Hub via a headless browser (configured by
//! `run_tests.sh` / the CI `wasm-test` job). Tests are wasm-bindgen-test
//! futures and target stable public resources so no auth is required.

#![cfg(target_family = "wasm")]

use futures_util::StreamExt;
use hf_hub::HFClientBuilder;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

const ENDPOINT: &str = "https://huggingface.co";
const TEST_MODEL_OWNER: &str = "openai-community";
const TEST_MODEL_NAME: &str = "gpt2";

fn client() -> hf_hub::HFClient {
    HFClientBuilder::new()
        .endpoint(ENDPOINT.to_string())
        .build()
        .expect("build HFClient")
}

#[wasm_bindgen_test]
async fn download_file_to_bytes_works() {
    let bytes = client()
        .model(TEST_MODEL_OWNER, TEST_MODEL_NAME)
        .download_file_to_bytes()
        .filename("config.json")
        .send()
        .await
        .expect("download config.json");

    assert!(!bytes.is_empty(), "config.json was empty");
    let s = std::str::from_utf8(&bytes).expect("config.json is utf-8");
    assert!(s.trim_start().starts_with('{'), "config.json doesn't look like JSON: {s:.80}");
    assert!(s.contains("\"model_type\""), "config.json missing model_type field");
}

#[wasm_bindgen_test]
async fn model_info_works() {
    let info = client()
        .model(TEST_MODEL_OWNER, TEST_MODEL_NAME)
        .info()
        .send()
        .await
        .expect("model info");

    assert_eq!(info.id, format!("{TEST_MODEL_OWNER}/{TEST_MODEL_NAME}"));
    assert!(info.pipeline_tag.is_some(), "expected gpt2 to have a pipeline_tag, got None",);
}

#[wasm_bindgen_test]
async fn raw_diff_works() {
    // Two old, immutable gpt2 commit SHAs — the older commit added the ONNX
    // file, the newer one added tokenizer_config.json. The diff between them
    // is small, stable, and non-empty.
    let diff = client()
        .model(TEST_MODEL_OWNER, TEST_MODEL_NAME)
        .get_raw_diff()
        .compare("11c5a3d5811f..607a30d783df")
        .send()
        .await
        .expect("get_raw_diff");

    assert!(!diff.is_empty(), "raw diff between known gpt2 commits was empty");
    assert!(
        diff.contains("tokenizer_config.json"),
        "expected the diff to mention tokenizer_config.json, got: {diff:.200}",
    );
}

/// Small xet-backed file in a stable public test repo
/// (`hf-internal-testing/tiny-random-bert/pytorch_model.bin`, ~528 KiB).
/// Exercising this drives the wasm download dispatch through the
/// `paths-info`-detected xet path in `repository/download.rs`, which
/// goes through `hf-xet`'s threaded-wasm runtime.
const XET_REPO_OWNER: &str = "hf-internal-testing";
const XET_REPO_NAME: &str = "tiny-random-bert";
const XET_FILE_NAME: &str = "pytorch_model.bin";
const XET_FILE_SIZE: usize = 540_217;

#[wasm_bindgen_test]
async fn download_xet_file_works() {
    let bytes = client()
        .model(XET_REPO_OWNER, XET_REPO_NAME)
        .download_file_to_bytes()
        .filename(XET_FILE_NAME)
        .send()
        .await
        .expect("download xet-backed pytorch_model.bin");

    assert_eq!(
        bytes.len(),
        XET_FILE_SIZE,
        "xet-backed file size mismatch: got {} bytes, want {XET_FILE_SIZE}",
        bytes.len(),
    );
    // PyTorch pickle archives are zip files, so the first two bytes are the
    // ZIP local file header magic. A sanity check that we got the actual
    // file content, not an LFS pointer or an error body.
    assert_eq!(&bytes[..2], b"PK", "pytorch_model.bin did not start with the ZIP magic");
}

#[wasm_bindgen_test]
async fn list_models_pagination_works() {
    // Passing `limit >= 1000` short-circuits the `?limit=` query
    // parameter (see `HFClient::list_models`), so the server returns its
    // default page size (1000) and the pagination loop has to follow the
    // `Link: rel="next"` header to satisfy the requested count.
    const REQUESTED: usize = 1100;

    let client = client();
    let stream = client.list_models().limit(REQUESTED).send().expect("list_models builder");
    futures_util::pin_mut!(stream);

    let mut count = 0usize;
    while let Some(item) = stream.next().await {
        let _model = item.expect("model item");
        count += 1;
    }

    assert_eq!(count, REQUESTED, "pagination short-changed us: got {count} of {REQUESTED} requested models");
}

#[wasm_bindgen_test]
async fn list_spaces_search_works() {
    let client = client();
    let stream = client
        .list_spaces()
        .search("stable-diffusion".to_string())
        .limit(3usize)
        .send()
        .expect("list_spaces builder");
    futures_util::pin_mut!(stream);

    let mut count = 0usize;
    while let Some(item) = stream.next().await {
        let space = item.expect("space item");
        assert!(!space.id.is_empty(), "space id was empty");
        count += 1;
    }

    assert!(count >= 1, "expected at least one space result, got {count}");
}
