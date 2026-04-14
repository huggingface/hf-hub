use criterion::{criterion_group, criterion_main, Criterion};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bench_cache_dir() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("hf_hub_bench_{}", std::process::id()));
    std::fs::create_dir_all(&path).ok();
    path
}

fn fresh_cache_dir() -> PathBuf {
    let mut path = bench_cache_dir();
    let suffix: u64 = rand::random();
    path.push(format!("run_{suffix}"));
    std::fs::create_dir_all(&path).ok();
    path
}

// ---------------------------------------------------------------------------
// Download benchmarks
// ---------------------------------------------------------------------------

fn bench_download_small_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("download");
    group.sample_size(10);

    group.bench_function("small file (cold cache)", |b| {
        b.iter_with_setup(
            fresh_cache_dir,
            |cache_dir| {
                let api = ApiBuilder::new()
                    .with_progress(false)
                    .with_cache_dir(cache_dir)
                    .build()
                    .unwrap();
                api.model("julien-c/dummy-unknown".to_string())
                    .download("config.json")
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
            fresh_cache_dir,
            |cache_dir| {
                let api = ApiBuilder::new()
                    .with_progress(false)
                    .with_cache_dir(cache_dir)
                    .build()
                    .unwrap();
                let repo = Repo::with_revision(
                    "julien-c/dummy-unknown".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                );
                api.repo(repo).download("config.json").unwrap();
            },
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Cache hit benchmarks
// ---------------------------------------------------------------------------

fn bench_get_warm_cache(c: &mut Criterion) {
    // Pre-populate cache
    let cache_dir = fresh_cache_dir();
    let api = ApiBuilder::new()
        .with_progress(false)
        .with_cache_dir(cache_dir.clone())
        .build()
        .unwrap();
    api.model("julien-c/dummy-unknown".to_string())
        .download("config.json")
        .unwrap();

    let mut group = c.benchmark_group("cache");

    group.bench_function("get (warm cache, via Api)", |b| {
        b.iter(|| {
            let api = ApiBuilder::new()
                .with_progress(false)
                .with_cache_dir(cache_dir.clone())
                .build()
                .unwrap();
            api.model("julien-c/dummy-unknown".to_string())
                .get("config.json")
                .unwrap();
        });
    });

    group.finish();
}

fn bench_cache_repo_get(c: &mut Criterion) {
    // Pre-populate cache
    let cache_dir = fresh_cache_dir();
    let api = ApiBuilder::new()
        .with_progress(false)
        .with_cache_dir(cache_dir.clone())
        .build()
        .unwrap();
    api.model("julien-c/dummy-unknown".to_string())
        .download("config.json")
        .unwrap();

    let cache = Cache::new(cache_dir);
    let repo = cache.repo(Repo::model("julien-c/dummy-unknown".to_string()));

    let mut group = c.benchmark_group("cache");

    group.bench_function("CacheRepo::get (warm cache, no network)", |b| {
        b.iter(|| {
            repo.get("config.json").unwrap();
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Info / metadata benchmarks
// ---------------------------------------------------------------------------

fn bench_info(c: &mut Criterion) {
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("api");
    group.sample_size(10);

    group.bench_function("info", |b| {
        b.iter(|| {
            api.model("julien-c/dummy-unknown".to_string())
                .info()
                .unwrap();
        });
    });

    group.finish();
}

fn bench_metadata(c: &mut Criterion) {
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .unwrap();
    let repo = api.model("julien-c/dummy-unknown".to_string());
    let url = repo.url("config.json");

    let mut group = c.benchmark_group("api");
    group.sample_size(10);

    group.bench_function("metadata", |b| {
        b.iter(|| {
            api.metadata(&url).unwrap();
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
