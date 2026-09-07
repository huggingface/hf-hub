//! User operations: authentication, user info, and social features.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example users

use futures::StreamExt;
use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let me = client.whoami().send().await?;
    println!("Logged in as: {} (type: {:?}, pro: {:?})", me.username, me.user_type, me.is_pro);

    let user = client.user_overview().username("julien-c").send().await?;
    println!("\nUser overview: {} (fullname: {:?})", user.username, user.fullname);

    let org = client.organization_overview().organization("huggingface").send().await?;
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

    let org_followers = client
        .list_organization_followers()
        .organization("huggingface")
        .limit(3_usize)
        .send()?;
    futures::pin_mut!(org_followers);
    println!("\nFollowers of huggingface:");
    while let Some(Ok(follower)) = org_followers.next().await {
        println!("  - {}", follower.username);
    }

    let likes = client.list_user_likes().username("julien-c").limit(3_usize).send()?;
    futures::pin_mut!(likes);
    println!("\njulien-c likes:");
    while let Some(Ok(like)) = likes.next().await {
        println!("  - {} ({})", like.repo.name, like.repo.repo_type);
    }

    let repositories = client.list_settings_repositories().limit(3_usize).send()?;
    futures::pin_mut!(repositories);
    println!("\nYour storage usage:");
    while let Some(Ok(entry)) = repositories.next().await {
        println!("  - {} ({}): {:?} bytes", entry.id, entry.visibility, entry.storage);
    }

    Ok(())
}
