use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures::FutureExt;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::repository::{HFRepository, RepoType, RepoTypeDataset, RepoTypeKernel, RepoTypeModel, RepoTypeSpace};

/// A dedicated background thread that owns a single-threaded tokio runtime,
/// modeled on `reqwest::blocking`.
///
/// The runtime is created on — and never leaves — the background thread, so
/// no caller thread ever calls `Runtime::block_on` or drops the runtime.
/// Both operations panic when performed from inside another tokio runtime,
/// which is exactly the situation a blocking API embedded in an async
/// application ends up in. Instead, sync wrapper methods hand a
/// `'static + Send` future over via [`RuntimeThread::run_future`] and park
/// on a std channel until the result comes back, which merely blocks the
/// calling thread.
///
/// Shutdown: the background thread sits in
/// `runtime.block_on(shutdown_rx)`. When the last handle sharing this
/// `RuntimeThread` drops, the `_shutdown` sender drops with it, the
/// receiver resolves, `block_on` returns, and the runtime is dropped on its
/// own thread. The thread is detached rather than joined so that dropping
/// the last handle never blocks the caller (reqwest-style).
pub(crate) struct RuntimeThread {
    handle: tokio::runtime::Handle,
    _shutdown: tokio::sync::oneshot::Sender<()>,
}

impl RuntimeThread {
    /// Spawns the background thread and waits for its runtime to come up.
    fn spawn() -> HFResult<Arc<Self>> {
        let (handle_tx, handle_rx) = std::sync::mpsc::channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        std::thread::Builder::new()
            .name("hf-hub-blocking-runtime".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    Ok(runtime) => runtime,
                    Err(e) => {
                        let _ = handle_tx.send(Err(e));
                        return;
                    },
                };
                let _ = handle_tx.send(Ok(runtime.handle().clone()));
                runtime.block_on(async {
                    let _ = shutdown_rx.await;
                });
            })
            .map_err(|e| HFError::Other(format!("Failed to spawn tokio runtime thread: {e}")))?;
        let handle = handle_rx
            .recv()
            .map_err(|_| HFError::Other("tokio runtime thread exited before initializing".to_string()))?
            .map_err(|e| HFError::Other(format!("Failed to create tokio runtime: {e}")))?;
        Ok(Arc::new(Self {
            handle,
            _shutdown: shutdown_tx,
        }))
    }

    /// Runs `future` on the background runtime and blocks the calling thread
    /// until it completes.
    ///
    /// Safe to call from inside another tokio runtime: the future executes on
    /// the dedicated thread while the caller parks on a std channel, so no
    /// runtime is ever entered re-entrantly. If the future panics, the panic
    /// is resumed on the calling thread, matching the semantics the previous
    /// `Runtime::block_on`-based implementation had for panicking tasks.
    pub(crate) fn run_future<F, T>(&self, future: F) -> HFResult<T>
    where
        F: Future<Output = HFResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
        self.handle.spawn(async move {
            let result = AssertUnwindSafe(future).catch_unwind().await;
            let _ = result_tx.send(result);
        });
        match result_rx.recv() {
            Ok(Ok(result)) => result,
            Ok(Err(panic)) => std::panic::resume_unwind(panic),
            Err(_) => Err(HFError::Other("hf-hub blocking runtime thread stopped unexpectedly".to_string())),
        }
    }
}

impl std::fmt::Debug for RuntimeThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeThread").finish()
    }
}

/// Synchronous/blocking counterpart to [`HFClient`].
///
/// Wraps an [`HFClient`] together with a dedicated background thread that
/// owns a single-threaded tokio runtime, so the async API can be used from
/// synchronous code. Because the runtime lives on its own thread (the
/// `reqwest::blocking` model), the blocking methods are also safe to call
/// from inside an async runtime: the call blocks the current thread while
/// the work runs on the background thread, instead of panicking on tokio's
/// nested-runtime guard.
///
/// Xet uploads and downloads do not run on this runtime: hf-xet requires a
/// multi-threaded runtime with the IO and time drivers enabled, so the
/// single-threaded runtime here does not meet its requirements. When a Xet
/// transfer is triggered through any blocking handle, hf-xet spins up its
/// own multi-threaded thread pool to back the `XetSession`, separate from
/// the runtime on the background thread owned by `HFClientSync`.
///
/// See [`HFClient`] for configuration and API semantics.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
#[derive(Clone)]
pub struct HFClientSync {
    pub(crate) inner: HFClient,
    pub(crate) runtime: Arc<RuntimeThread>,
}

/// Synchronous/blocking counterpart to [`HFRepository`], parameterized by the repo kind via `T`.
///
/// Wraps an [`HFRepository<T>`] and blocks on the corresponding async methods.
///
/// See [`HFRepository`] for method semantics.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub struct HFRepositorySync<T: RepoType> {
    pub(crate) inner: Arc<HFRepository<T>>,
    pub(crate) runtime: Arc<RuntimeThread>,
}

impl<T: RepoType> Clone for HFRepositorySync<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            runtime: Arc::clone(&self.runtime),
        }
    }
}

/// Synchronous/blocking counterpart to [`crate::buckets::HFBucket`].
///
/// Wraps an [`crate::buckets::HFBucket`] and blocks on the corresponding async
/// methods.
///
/// See [`crate::buckets::HFBucket`] for method semantics.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
#[derive(Clone)]
pub struct HFBucketSync {
    pub(crate) inner: Arc<crate::buckets::HFBucket>,
    pub(crate) runtime: Arc<RuntimeThread>,
}

impl std::fmt::Debug for HFClientSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFClientSync").finish()
    }
}

impl<T: RepoType> std::fmt::Debug for HFRepositorySync<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFRepositorySync").field("inner", &self.inner).finish()
    }
}

impl std::fmt::Debug for HFBucketSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFBucketSync").field("inner", &self.inner).finish()
    }
}

impl HFClientSync {
    /// Creates an `HFClientSync` using the default configuration from the environment.
    ///
    /// Reads the standard environment variables used by [`HFClient::new`].
    ///
    /// # Errors
    ///
    /// Returns an error if the tokio runtime cannot be created or if `HFClient::new` fails.
    pub fn new() -> HFResult<Self> {
        Ok(Self {
            inner: HFClient::new()?,
            runtime: RuntimeThread::spawn()?,
        })
    }

    /// Creates an `HFClientSync` from an existing [`HFClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if the tokio runtime cannot be created.
    pub fn from_inner(inner: HFClient) -> HFResult<Self> {
        Ok(Self {
            inner,
            runtime: RuntimeThread::spawn()?,
        })
    }

    /// Creates a blocking handle for any repo kind via a turbofished generic.
    ///
    /// See [`HFClient::repository`].
    pub fn repository<T: RepoType>(
        &self,
        repo_type: T,
        owner: impl Into<String>,
        name: impl Into<String>,
    ) -> HFRepositorySync<T> {
        HFRepositorySync::new(self.clone(), repo_type, owner, name)
    }

    /// Creates a blocking handle for a model repository.
    ///
    /// See [`HFClient::model`].
    pub fn model(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeModel> {
        self.repository(RepoTypeModel, owner, name)
    }

    /// Creates a blocking handle for a dataset repository.
    ///
    /// See [`HFClient::dataset`].
    pub fn dataset(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeDataset> {
        self.repository(RepoTypeDataset, owner, name)
    }

    /// Creates a blocking handle for a Space repository.
    ///
    /// See [`HFClient::space`].
    pub fn space(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeSpace> {
        self.repository(RepoTypeSpace, owner, name)
    }

    /// Creates a blocking handle for a kernel repository.
    ///
    /// See [`HFClient::kernel`].
    pub fn kernel(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeKernel> {
        self.repository(RepoTypeKernel, owner, name)
    }

    /// Creates a blocking handle for a bucket.
    ///
    /// See [`HFClient::bucket`].
    pub fn bucket(&self, owner: impl Into<String>, name: impl Into<String>) -> HFBucketSync {
        HFBucketSync::new(self.clone(), owner, name)
    }
}

impl<T: RepoType> HFRepositorySync<T> {
    /// Creates a blocking repository handle.
    ///
    /// See [`HFRepository::new`].
    pub fn new(client: HFClientSync, repo_type: T, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(HFRepository::new(client.inner.clone(), repo_type, owner, name)),
            runtime: client.runtime.clone(),
        }
    }

    /// The repository owner (user or organization name).
    pub fn owner(&self) -> &str {
        self.inner.owner()
    }

    /// The repository name (without the owner prefix).
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// The full `"owner/name"` identifier used in Hub API calls.
    ///
    /// If no owner is set, returns just the name (for repos using short-form IDs like `"gpt2"`).
    pub fn repo_path(&self) -> String {
        self.inner.repo_path()
    }

    /// The marker for this handle's repo kind. Call
    /// [`RepoType::singular`] / [`RepoType::plural`] / [`RepoType::url_prefix`] on it
    /// to get the corresponding string.
    pub fn repo_type(&self) -> &T {
        self.inner.repo_type()
    }
}

impl HFBucketSync {
    /// Creates a blocking bucket handle.
    ///
    /// See [`crate::buckets::HFBucket::new`].
    pub fn new(client: HFClientSync, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(crate::buckets::HFBucket::new(client.inner.clone(), owner, name)),
            runtime: client.runtime.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hfapisync_creation() {
        let client = HFClientSync::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_hfapisync_from_api() {
        let async_client = HFClient::builder().build().unwrap();
        let client = HFClientSync::from_inner(async_client);
        assert!(client.is_ok());
    }

    #[test]
    fn test_sync_repo_constructors() {
        let client = HFClientSync::from_inner(HFClient::builder().build().unwrap()).unwrap();
        let repo = client.model("openai-community", "gpt2");
        let space = client.space("huggingface", "transformers-benchmarks");

        assert_eq!(repo.owner(), "openai-community");
        assert_eq!(repo.name(), "gpt2");
        assert_eq!(repo.repo_type().singular(), "model");
        assert_eq!(space.repo_type().singular(), "space");
    }

    /// Endpoint that refuses connections: bind an ephemeral port, then free it.
    fn unreachable_endpoint() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        format!("http://127.0.0.1:{port}")
    }

    fn client_with_unreachable_endpoint() -> HFClientSync {
        let inner = HFClient::builder()
            .endpoint(unreachable_endpoint())
            .retry_max_attempts(0)
            .build()
            .unwrap();
        HFClientSync::from_inner(inner).unwrap()
    }

    /// Regression test: with the runtime owned by the caller-side handle, this
    /// panicked on tokio's nested-runtime guard ("Cannot start a runtime from
    /// within a runtime"). The connection error is expected; the point is
    /// completing without a panic.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sync_call_inside_multi_thread_runtime_does_not_panic() {
        let client = client_with_unreachable_endpoint();
        let result = client.model("openai-community", "gpt2").info().send();
        assert!(result.is_err());
    }

    /// Same regression as the multi-thread flavor. The call blocks the only
    /// runtime thread while the work runs on the dedicated background thread,
    /// so it still completes.
    #[tokio::test]
    async fn sync_call_inside_current_thread_runtime_does_not_panic() {
        let client = client_with_unreachable_endpoint();
        let result = client.model("openai-community", "gpt2").info().send();
        assert!(result.is_err());
    }

    /// Regression test: dropping the handle (and with it the old caller-side
    /// `Runtime`) from async context also panicked before the runtime moved
    /// to its own thread.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn construct_and_drop_inside_async_context_does_not_panic() {
        let client = HFClientSync::from_inner(HFClient::builder().build().unwrap()).unwrap();
        drop(client);
    }

    #[test]
    fn clones_share_runtime_thread_and_survive_staggered_drops() {
        let client = client_with_unreachable_endpoint();
        let clone = client.clone();
        let repo = client.model("openai-community", "gpt2");
        assert!(Arc::ptr_eq(&client.runtime, &clone.runtime));
        assert!(Arc::ptr_eq(&client.runtime, &repo.runtime));

        drop(client);
        drop(clone);
        let result = repo.exists().send();
        assert!(result.is_err());
    }

    #[test]
    fn run_future_resumes_task_panics_on_the_caller() {
        let runtime = RuntimeThread::spawn().unwrap();
        let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
            runtime.run_future(async {
                panic!("boom");
                #[allow(unreachable_code)]
                Ok::<(), HFError>(())
            })
        }));
        let panic = caught.unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"boom"));
    }
}
