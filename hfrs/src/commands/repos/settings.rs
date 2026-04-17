use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::types::{GatedApprovalMode, GatedNotificationsMode, RepoUpdateSettingsParams};

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;

/// Update repository settings
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g. username/my-model)
    pub repo_id: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Gating strategy (e.g. "auto", "manual", or "false" to disable)
    #[arg(long)]
    pub gated: Option<String>,

    /// Set private visibility
    #[arg(long)]
    pub private: Option<bool>,

    /// Repository description
    #[arg(long)]
    pub description: Option<String>,

    /// Disable discussions on the repository
    #[arg(long)]
    pub discussions_disabled: Option<bool>,

    /// Email for gated access notifications
    #[arg(long)]
    pub gated_notifications_email: Option<String>,

    /// Gated notifications mode ("bulk" or "real-time")
    #[arg(long)]
    pub gated_notifications_mode: Option<String>,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let repo_type: hf_hub::types::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(client, &args.repo_id, repo_type);

    let gated: Option<GatedApprovalMode> = args.gated.map(|g| g.parse()).transpose()?;
    let gated_notifications_mode: Option<GatedNotificationsMode> =
        args.gated_notifications_mode.map(|m| m.parse()).transpose()?;

    let params = RepoUpdateSettingsParams {
        private: args.private,
        gated,
        description: args.description,
        discussions_disabled: args.discussions_disabled,
        gated_notifications_email: args.gated_notifications_email,
        gated_notifications_mode,
    };
    repo.update_settings(&params).await?;
    Ok(CommandResult::Silent)
}
