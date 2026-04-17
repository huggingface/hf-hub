use criterion::{Criterion, criterion_group, criterion_main};
use hf_hub::types::{RepoDownloadFileParams, RepoGetFileMetadataParams, RepoInfoParams};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Download benchmarks
// ---------------------------------------------------------------------------

fn bench_download_small_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("download");
    group.sample_size(10);

    group.bench_function("small file (cold cache)", |b| {
        b.iter_with_setup(
            || {
                let tmp = TempDir::new().unwrap();
                let client = hf_hub::HFClientBuilder::new()
                    .cache_dir(tmp.path().to_path_buf())
                    .build_sync()
                    .unwrap();
                (tmp, client)
            },
            |(_tmp, client)| {
                client
                    .model("julien-c", "dummy-unknown")
                    .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
                    .unwrap();
            },
        );
    });

    group.finish();
}

fn bench_download_with_revision(c: &mut Criterion) {
    let mut group = c.benchmark_group("download");
    group.sample_size(10);

    group.bench_function("with revision (cold cache)", |b| {
        b.iter_with_setup(
            || {
                let tmp = TempDir::new().unwrap();
                let client = hf_hub::HFClientBuilder::new()
                    .cache_dir(tmp.path().to_path_buf())
                    .build_sync()
                    .unwrap();
                (tmp, client)
            },
            |(_tmp, client)| {
                client
                    .model("julien-c", "dummy-unknown")
                    .download_file(
                        &RepoDownloadFileParams::builder()
                            .revision("main")
                            .filename("config.json")
                            .build(),
                    )
                    .unwrap();
            },
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Cache hit benchmarks
// ---------------------------------------------------------------------------

fn bench_get_warm_cache(c: &mut Criterion) {
    // Pre-populate cache (kept alive for the whole benchmark)
    let tmp = TempDir::new().unwrap();
    let client = hf_hub::HFClientBuilder::new()
        .cache_dir(tmp.path().to_path_buf())
        .build_sync()
        .unwrap();
    client
        .model("julien-c", "dummy-unknown")
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .unwrap();

    let mut group = c.benchmark_group("cache");

    group.bench_function("get (warm cache, via Api)", |b| {
        b.iter(|| {
            client
                .model("julien-c", "dummy-unknown")
                .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
                .unwrap();
        });
    });

    group.finish();
}

fn bench_cache_repo_get(c: &mut Criterion) {
    // Pre-populate cache
    let tmp = TempDir::new().unwrap();
    let client = hf_hub::HFClientBuilder::new()
        .cache_dir(tmp.path().to_path_buf())
        .build_sync()
        .unwrap();
    client
        .model("julien-c", "dummy-unknown")
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .unwrap();

    let mut group = c.benchmark_group("cache");

    group.bench_function("CacheRepo::get (warm cache, no network)", |b| {
        b.iter(|| {
            client
                .model("julien-c", "dummy-unknown")
                .download_file(
                    &RepoDownloadFileParams::builder()
                        .local_files_only(true)
                        .filename("config.json")
                        .build(),
                )
                .unwrap();
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Info / metadata benchmarks
// ---------------------------------------------------------------------------

fn bench_info(c: &mut Criterion) {
    let client = hf_hub::HFClientSync::new().unwrap();

    let mut group = c.benchmark_group("api");
    group.sample_size(10);

    group.bench_function("info", |b| {
        b.iter(|| {
            client
                .model("julien-c", "dummy-unknown")
                .info(&RepoInfoParams::default())
                .unwrap();
        });
    });

    group.finish();
}

fn bench_metadata(c: &mut Criterion) {
    let client = hf_hub::HFClientSync::new().unwrap();
    let repo = client.model("julien-c", "dummy-unknown");

    let mut group = c.benchmark_group("api");
    group.sample_size(10);

    group.bench_function("metadata", |b| {
        b.iter(|| {
            repo.get_file_metadata(&RepoGetFileMetadataParams::builder().filepath("config.json").build())
                .unwrap();
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(download, bench_download_small_file, bench_download_with_revision);
criterion_group!(cache, bench_get_warm_cache, bench_cache_repo_get);
criterion_group!(api, bench_info, bench_metadata);

criterion_main!(download, cache, api);
