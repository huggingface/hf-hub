//! Cache manager example
//!
//! # Usage
//!
//! Scan the cache
//! ```text
//! cargo run --example cache-manager --features="cache-manager"
//! ```
//!
//! Get delete strategy for a specific revision
//! ```text
//! # The first (and only arg) is the revision ID
//! cargo run --example cache-manager --features="cache-manager" -- e8c3b32edf5434bc2275fc9bab85f82640a19130
//! ```
//!
//! Display table output
//! ```text
//! cargo run --example cache-manager --features="cache-manager,cache-manager-display"
//! ```

use hf_hub::cache_manager::HFCacheInfo;

fn main() {
    let revision = std::env::args().nth(1);

    let cache_info = HFCacheInfo::scan_cache_dir(None).unwrap();

    #[cfg(not(feature = "cache-manager-display"))]
    {
        dbg!(&cache_info);
    }

    if let Some(revision) = revision {
        let revisions = [revision.as_str()];
        let delete_strat = cache_info.delete_revisions(&revisions);
        dbg!(&delete_strat);
    }

    #[cfg(feature = "cache-manager-display")]
    {
        // let table = cache_info.export_as_table();
        let table = cache_info.export_as_table_comfy();
        print!("{table}");
    }
}
