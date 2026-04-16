use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::{hf_home, resolve_cache_dir};

use crate::output::CommandResult;

#[derive(ClapArgs)]
#[command(about = "Print information about the environment")]
pub struct Args {}

fn env_or(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "not set".to_string())
}

fn env_set(name: &str) -> &'static str {
    if std::env::var(name).is_ok() { "set" } else { "not set" }
}

pub async fn execute(_args: Args) -> Result<CommandResult> {
    let mut lines = Vec::new();

    lines.push(format!("hfrs version: {}", env!("CARGO_PKG_VERSION")));
    lines.push(format!("Platform: {} {}", std::env::consts::OS, std::env::consts::ARCH));
    lines.push(String::new());

    // Authentication
    lines.push("Authentication:".to_string());
    lines.push(format!("  HF_TOKEN: {}", env_set("HF_TOKEN")));
    lines.push(format!("  HF_TOKEN_PATH: {}", env_or("HF_TOKEN_PATH")));
    lines.push(format!("  HF_HUB_DISABLE_IMPLICIT_TOKEN: {}", env_or("HF_HUB_DISABLE_IMPLICIT_TOKEN")));
    lines.push(String::new());

    // Endpoints
    lines.push("Endpoints:".to_string());
    lines.push(format!("  HF_ENDPOINT: {}", env_or("HF_ENDPOINT")));
    lines.push(format!(
        "  Active endpoint: {}",
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
    ));
    lines.push(String::new());

    // Paths
    lines.push("Paths:".to_string());
    lines.push(format!("  HF_HOME: {}", env_or("HF_HOME")));
    lines.push(format!("  HF_HUB_CACHE: {}", env_or("HF_HUB_CACHE")));
    lines.push(format!("  HUGGINGFACE_HUB_CACHE: {}", env_or("HUGGINGFACE_HUB_CACHE")));
    lines.push(format!("  XDG_CACHE_HOME: {}", env_or("XDG_CACHE_HOME")));
    lines.push(format!("  Resolved HF home: {}", hf_home().display()));
    lines.push(format!("  Resolved cache dir: {}", resolve_cache_dir().display()));
    lines.push(String::new());

    // Logging
    lines.push("Logging:".to_string());
    let log_level = if let Ok(level) = std::env::var("HF_LOG_LEVEL") {
        level
    } else if std::env::var("HF_DEBUG").is_ok() {
        "debug (via HF_DEBUG)".to_string()
    } else {
        "warn (default)".to_string()
    };
    lines.push(format!("  HF_LOG_LEVEL: {log_level}"));
    lines.push(format!("  HF_DEBUG: {}", env_set("HF_DEBUG")));
    lines.push(format!("  HF_XET_LOG_LEVEL: {}", env_or("HF_XET_LOG_LEVEL")));
    lines.push(String::new());

    // Display
    lines.push("Display:".to_string());
    lines.push(format!("  NO_COLOR: {}", env_set("NO_COLOR")));
    lines.push(format!("  CLICOLOR_FORCE: {}", env_or("CLICOLOR_FORCE")));
    lines.push(String::new());

    // Misc
    lines.push("Misc:".to_string());
    lines.push(format!("  HF_HUB_USER_AGENT_ORIGIN: {}", env_or("HF_HUB_USER_AGENT_ORIGIN")));

    Ok(CommandResult::Raw(lines.join("\n")))
}
