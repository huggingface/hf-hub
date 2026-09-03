//! Regression test for CWE-770: `download_file_to_bytes` must not pre-allocate its destination
//! buffer from the server-declared `Content-Length` header.
//!
//! The header is attacker-controlled on any untrusted endpoint (malicious mirror, `HF_ENDPOINT`
//! override, MITM). A hostile server advertising a huge `Content-Length` while sending only a few
//! bytes used to trigger `BytesMut::with_capacity(content_length)` — an unbounded *upfront*
//! allocation that aborts the process on allocation failure or exhausts memory (DoS), before a
//! single body byte is read.
//!
//! The test points an `HFRepository` at a local mock HTTP server that returns
//! `Content-Length: 134217728` (128 MiB) with a 3-byte body, then asserts that the largest
//! *single* heap allocation made while collecting the response stays bounded — i.e. the buffer
//! grows from the bytes that actually arrive, not from the declared size.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use hf_hub::HFClient;
use hf_hub::repository::{HFRepository, RepoTypeModel};

// --- counting global allocator (test-only) ---------------------------------------------
// Tracks the largest single allocation request observed in this process. Installed as the global
// allocator for the test binary only. NOTE: this is process-global, so the security assertions run
// serially inside this single test fn (no other `#[test]` in this file).
struct CountingAllocator;
static MAX_SINGLE_ALLOC: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let _ = MAX_SINGLE_ALLOC.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| Some(cur.max(size)));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

// A malicious server advertises 128 MiB but ships only 3 bytes.
const DECLARED_CONTENT_LENGTH: usize = 128 * 1024 * 1024;
// We tolerate single allocations up to this (covers tokio/reqwest connection buffers etc.); the
// vulnerability is a *single* allocation on the order of DECLARED_CONTENT_LENGTH.
const MAX_ACCEPTABLE_SINGLE_ALLOC: usize = 32 * 1024 * 1024;

#[tokio::test(flavor = "multi_thread")]
async fn download_file_to_bytes_does_not_trust_content_length() {
    // Mock server: 200 OK with a lying Content-Length and a tiny body.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let body = b"pwn";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
            DECLARED_CONTENT_LENGTH
        );
        use tokio::io::AsyncWriteExt;
        stream.write_all(resp.as_bytes()).await.unwrap();
        stream.write_all(body).await.unwrap();
        // Briefly hold the connection so the client observes the headers + body before EOF.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let _ = stream.shutdown().await;
    });

    let client = HFClient::builder()
        .endpoint(format!("http://{}", addr))
        .token("hf_test")
        .cache_enabled(false)
        .build()
        .expect("offline client construction");
    let repo: HFRepository<RepoTypeModel> = client.model("user", "repo");

    // Reset the counter right before the call so only this download's allocations are measured.
    MAX_SINGLE_ALLOC.store(0, Ordering::Relaxed);

    // We intentionally do not `.expect()` success: a short body may make the body read error in
    // *both* patched and unpatched code. The security property we assert is the *allocation*, which
    // the lying header would have forced before any body byte is read.
    let _ = repo
        .download_file_to_bytes()
        .filename("evil.bin".to_string())
        .send()
        .await;

    let max_alloc = MAX_SINGLE_ALLOC.load(Ordering::Relaxed);
    println!("largest single allocation during download: {max_alloc} bytes");
    assert!(
        max_alloc <= MAX_ACCEPTABLE_SINGLE_ALLOC,
        "download made a {max_alloc}-byte allocation (declared Content-Length was {DECLARED_CONTENT_LENGTH}); \
         the destination buffer must not be pre-sized from the untrusted header (CWE-770)"
    );
}
