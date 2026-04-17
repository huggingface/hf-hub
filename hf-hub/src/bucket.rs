use std::fmt;

use crate::client::HFClient;

/// A handle for a single bucket on the Hugging Face Hub.
///
/// `HFBucket` is created via [`HFClient::bucket`] and binds together the client,
/// owner (namespace), and bucket name. All bucket-scoped API operations are methods
/// on this type.
///
/// Cheap to clone — the inner [`HFClient`] is `Arc`-backed.
///
/// # Example
///
/// ```rust,no_run
/// # use hf_hub::HFClient;
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::builder().build()?;
/// let bucket = client.bucket("my-org", "my-bucket");
/// assert_eq!(bucket.bucket_id(), "my-org/my-bucket");
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct HFBucket {
    pub(crate) hf_client: HFClient,
    owner: String,
    name: String,
}

impl fmt::Debug for HFBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFBucket")
            .field("owner", &self.owner)
            .field("name", &self.name)
            .finish()
    }
}

impl HFBucket {
    /// Construct a new bucket handle. Prefer [`HFClient::bucket`] in most cases.
    pub fn new(client: HFClient, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            hf_client: client,
            owner: owner.into(),
            name: name.into(),
        }
    }

    /// Return a reference to the underlying [`HFClient`].
    pub fn client(&self) -> &HFClient {
        &self.hf_client
    }

    /// The bucket owner (user or organization namespace).
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The bucket name (without owner prefix).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full `"owner/name"` bucket identifier used in Hub API calls.
    pub fn bucket_id(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::HFBucket;

    #[test]
    fn test_bucket_accessors() {
        let client = crate::HFClient::builder().build().unwrap();
        let bucket = HFBucket::new(client, "my-org", "my-bucket");

        assert_eq!(bucket.owner(), "my-org");
        assert_eq!(bucket.name(), "my-bucket");
        assert_eq!(bucket.bucket_id(), "my-org/my-bucket");
    }
}
