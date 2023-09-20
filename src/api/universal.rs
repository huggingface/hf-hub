use std::path::{Component, Path, PathBuf};

/// Common library for sync/tokio versions
pub struct Universal;


impl Universal {
    /// symlink_or_rename
    pub fn symlink_or_rename(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
        if dst.exists() {
            return Ok(());
        }

        let rel_src = Universal::make_relative(src, dst);
        #[cfg(target_os = "windows")]
        {
            if std::os::windows::fs::symlink_file(rel_src, dst).is_err() {
                std::fs::rename(src, dst)?;
            }
        }

        #[cfg(target_family = "unix")]
        std::os::unix::fs::symlink(rel_src, dst)?;

        Ok(())
    }

    /// make_relative
    pub fn make_relative(src: &Path, dst: &Path) -> PathBuf {
        let path = src;
        let base = dst;

        if path.is_absolute() != base.is_absolute() {
            panic!("This function is made to look at absolute paths only");
        }
        let mut ita = path.components();
        let mut itb = base.components();

        loop {
            match (ita.next(), itb.next()) {
                (Some(a), Some(b)) if a == b => (),
                (some_a, _) => {
                    // Ignoring b, because 1 component is the filename
                    // for which we don't need to go back up for relative
                    // filename to work.
                    let mut new_path = PathBuf::new();
                    for _ in itb {
                        new_path.push(Component::ParentDir);
                    }
                    if let Some(a) = some_a {
                        new_path.push(a);
                        for comp in ita {
                            new_path.push(comp);
                        }
                    }
                    return new_path;
                }
            }
        }
    }

}