window.BENCHMARK_DATA = {
  "lastUpdate": 1776256952955,
  "repoUrl": "https://github.com/huggingface/hf-hub",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "McPatate@users.noreply.github.com",
            "name": "Luc Georges",
            "username": "McPatate"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dfc858e31cfad4ec64aed01ed78b0d19be7d74fb",
          "message": "feat: add download and API benchmarks for sync (#149)\n\n* feat: add download and API benchmarks for sync\n\n* fix: use `CARGO_PKG_VERSION` in test rather than harcode version\n\n* feat: post comment in PR with bench results\n\n* refactor: use `tempfile` for benchmarks\n\n* refactor: bump dev-deps and rand to latest",
          "timestamp": "2026-04-15T14:40:02+02:00",
          "tree_id": "79a1efaf34c56aa5b3e104a272a0210f3aca8e02",
          "url": "https://github.com/huggingface/hf-hub/commit/dfc858e31cfad4ec64aed01ed78b0d19be7d74fb"
        },
        "date": 1776256952624,
        "tool": "cargo",
        "benches": [
          {
            "name": "download/small file (cold cache)",
            "value": 126726105,
            "range": "± 19677761",
            "unit": "ns/iter"
          },
          {
            "name": "download/with revision (cold cache)",
            "value": 137841269,
            "range": "± 8414938",
            "unit": "ns/iter"
          },
          {
            "name": "cache/get (warm cache, via Api)",
            "value": 16348,
            "range": "± 343",
            "unit": "ns/iter"
          },
          {
            "name": "cache/CacheRepo::get (warm cache, no network)",
            "value": 9110,
            "range": "± 36",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}