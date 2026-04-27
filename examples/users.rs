//! User operations: authentication, user info, and social features.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example users

use futures::StreamExt;
use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    client.auth_check().send().await?;
    println!("Token is valid");

    let me = client.whoami().send().await?;
    println!("Logged in as: {} (type: {:?}, pro: {:?})", me.username, me.user_type, me.is_pro);

    let user = client.get_user_overview().username("julien-c").send().await?;
    println!("\nUser overview: {} (fullname: {:?})", user.username, user.fullname);

    let org = client.get_organization_overview().organization("huggingface").send().await?;
    println!("Org overview: {} (fullname: {:?})", org.name, org.fullname);

    let followers = client.list_user_followers().username("julien-c").send()?;
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

    let following = client.list_user_following().username("julien-c").send()?;
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

    let members = client.list_organization_members().organization("huggingface").send()?;
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
