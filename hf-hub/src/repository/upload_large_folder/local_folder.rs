//! Byte-compatible reimplementation of Python `huggingface_hub`'s local upload
//! cache (`.cache/huggingface/upload/<path>.metadata`), enabling resumable
//! large-folder uploads interoperable with the Python tool.

use std::path::PathBuf;
