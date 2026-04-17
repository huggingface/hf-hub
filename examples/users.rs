//! User operations: authentication, user info, and social features.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example users

use futures::StreamExt;
use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    client.auth_check().await?;
    println!("Token is valid");

    let me = client.whoami().await?;
    println!("Logged in as: {} (type: {:?}, pro: {:?})", me.username, me.user_type, me.is_pro);

    let user = client.get_user_overview("julien-c").await?;
    println!("\nUser overview: {} (fullname: {:?})", user.username, user.fullname);

    let org = client.get_organization_overview("huggingface").await?;
    println!("Org overview: {} (fullname: {:?})", org.name, org.fullname);

    let followers = client.list_user_followers("julien-c", None)?;
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

    let following = client.list_user_following("julien-c", None)?;
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

    let members = client.list_organization_members("huggingface", None)?;
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
