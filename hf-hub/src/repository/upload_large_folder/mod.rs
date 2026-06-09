//! `HFRepository::upload_large_folder`: resumable, xet-optimized upload of a
//! large local folder as a sequence of adaptively-batched commits.

pub mod local_folder;
pub mod pipeline;
