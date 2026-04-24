//! Diff streaming: fetch and parse raw diffs as a stream of typed entries.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example diff

use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::commits::RepoGetRawDiffParams;

const GIT_EMPTY_TREE_HASH: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let repo = client.model("openai-community", "gpt2");

    let compare = format!("{GIT_EMPTY_TREE_HASH}..main");
    let mut diff_stream = repo
        .get_raw_diff_stream(&RepoGetRawDiffParams::builder().compare(&compare).build())
        .await?;

    let mut count = 0;
    println!("Streaming all files in main (first 10):");
    while let Some(result) = diff_stream.next().await {
        match result {
            Ok(d) => println!("  {:?} {} ({} bytes, binary={})", d.status, d.file_path, d.new_file_size, d.is_binary),
            Err(e) => eprintln!("  parse error: {e}"),
        }
        count += 1;
        if count >= 10 {
            break;
        }
    }

    Ok(())
}
