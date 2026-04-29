use anyhow::{Result, bail};
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::repository::{GatedApprovalMode, GatedNotifications, GatedNotificationsMode};

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
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(client, &args.repo_id, repo_type);

    let gated: Option<GatedApprovalMode> = args.gated.map(|g| g.parse()).transpose()?;
    let gated_notifications_mode: Option<GatedNotificationsMode> =
        args.gated_notifications_mode.map(|m| m.parse()).transpose()?;

    let gated_notifications = match (gated_notifications_mode, args.gated_notifications_email) {
        (Some(mode), email) => Some(GatedNotifications { mode, email }),
        (None, Some(_)) => bail!("--gated-notifications-email requires --gated-notifications-mode"),
        (None, None) => None,
    };

    repo.update_settings()
        .maybe_private(args.private)
        .maybe_gated(gated)
        .maybe_description(args.description)
        .maybe_discussions_disabled(args.discussions_disabled)
        .maybe_gated_notifications(gated_notifications)
        .send()
        .await?;
    Ok(CommandResult::Silent)
}
