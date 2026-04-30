fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }
    // Windows defaults the main-thread stack reserve to 1 MB. The download and upload paths in
    // hfrs combine deeply-nested async state machines with bon-builder type-states, which
    // overflows that limit at runtime. Bump to 8 MB to match the Linux/macOS defaults.
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let arg = if target_env == "gnu" {
        "-Wl,--stack,8388608"
    } else {
        "/STACK:8388608"
    };
    println!("cargo:rustc-link-arg-bin=hfrs={arg}");
}
