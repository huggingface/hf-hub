use walkdir::DirEntry;

pub const IGNORED_NAMES: &[&str] = &[".git", ".DS_Store", "Thumbs.db"];

pub fn is_ignored(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|s| IGNORED_NAMES.contains(&s))
}
