use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand, ValueEnum};
use hf_hub::types::RepoType;

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default())
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Red.on_default());

#[derive(Parser)]
#[command(name = "hfrs", about = "Hugging Face Hub CLI (Rust)", version, styles = STYLES)]
pub struct Cli {
    /// Authentication token (overrides HF_TOKEN env var and stored credentials)
    #[arg(long, env = "HF_TOKEN", global = true, hide_env_values = true)]
    pub token: Option<String>,

    /// API endpoint override
    #[arg(long, env = "HF_ENDPOINT", global = true)]
    pub endpoint: Option<String>,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Disable progress bars
    #[arg(long, global = true)]
    pub disable_progress_bars: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Manage authentication (login, logout, etc.)
    Auth(crate::commands::auth::Args),
    /// Interact with buckets on the Hub
    Buckets(crate::commands::buckets::Args),
    /// Manage local cache directory
    Cache(crate::commands::cache::Args),
    /// Interact with datasets on the Hub
    Datasets(crate::commands::datasets::Args),
    /// Download files from the Hub
    Download(crate::commands::download::Args),
    /// Interact with models on the Hub
    Models(crate::commands::models::Args),
    /// Manage repos on the Hub
    #[command(alias = "repo")]
    Repos(crate::commands::repos::Args),
    /// Interact with Spaces on the Hub
    Spaces(crate::commands::spaces::Args),
    /// Upload a file or folder to the Hub
    Upload(crate::commands::upload::Args),
    /// Print information about the environment
    Env(crate::commands::env::Args),
    /// Print the hfrs version
    Version(crate::commands::version::Args),
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
}

#[derive(Clone, ValueEnum)]
pub enum RepoTypeArg {
    Model,
    Dataset,
    Space,
}

impl From<RepoTypeArg> for RepoType {
    fn from(arg: RepoTypeArg) -> Self {
        match arg {
            RepoTypeArg::Model => RepoType::Model,
            RepoTypeArg::Dataset => RepoType::Dataset,
            RepoTypeArg::Space => RepoType::Space,
        }
    }
}
