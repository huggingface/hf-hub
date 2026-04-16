//! User operations: authentication, user info, and social features.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p hf-hub --example users

use futures::StreamExt;
use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::Result<()> {
    let api = HFClient::new()?;

    api.auth_check().await?;
    println!("Token is valid");

    let me = api.whoami().await?;
    println!("Logged in as: {} (type: {:?}, pro: {:?})", me.username, me.user_type, me.is_pro);

    let user = api.get_user_overview("julien-c").await?;
    println!("\nUser overview: {} (fullname: {:?})", user.username, user.fullname);

    let org = api.get_organization_overview("huggingface").await?;
    println!("Org overview: {} (fullname: {:?})", org.name, org.fullname);

    let followers = api.list_user_followers("julien-c", None)?;
    futures::pin_mut!(followers);
    println!("\nFollowers of julien-c:");
    let mut count = 0;
    while let Some(Ok(user)) = followers.next().await {
        println!("  - {}", user.username);
        count += 1;
        if count >= 3 {
            break;
        }
    }

    let following = api.list_user_following("julien-c", None)?;
    futures::pin_mut!(following);
    println!("\njulien-c is following:");
    let mut count = 0;
    while let Some(Ok(user)) = following.next().await {
        println!("  - {}", user.username);
        count += 1;
        if count >= 3 {
            break;
        }
    }

    let members = api.list_organization_members("huggingface", None)?;
    futures::pin_mut!(members);
    println!("\nMembers of huggingface:");
    let mut count = 0;
    while let Some(Ok(member)) = members.next().await {
        println!("  - {}", member.username);
        count += 1;
        if count >= 3 {
            break;
        }
    }

    Ok(())
}
