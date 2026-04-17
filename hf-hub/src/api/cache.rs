use crate::client::HFClient;
use crate::error::HFResult;
use crate::types::cache::HFCacheInfo;

impl HFClient {
    /// Scan the configured cache directory and return a summary of all cached
    /// repositories, revisions, and files.
    pub async fn scan_cache(&self) -> HFResult<HFCacheInfo> {
        crate::cache::scan_cache_dir(self.cache_dir()).await
    }
}

sync_api! {
    impl HFClient -> HFClientSync {
        fn scan_cache(&self) -> HFResult<HFCacheInfo>;
    }
}
