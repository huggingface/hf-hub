//! Test-only helpers shared across unit test modules.

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use crate::HFClient;

/// Loopback HTTP server matching exact request lines, plus an [`HFClient`] pointed at it.
///
/// Routes are `("<METHOD> <path> HTTP/1.1", "<raw response>")`; anything else gets a 404.
/// `{endpoint}` in a response expands to the server's base URL, so routes can redirect to
/// each other. Abort the returned handle when the test is done.
pub(crate) async fn mock_hub(routes: &[(&str, &str)]) -> (HFClient, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let routes: Vec<_> = routes
        .iter()
        .map(|(request, response)| (request.to_string(), response.replace("{endpoint}", &endpoint)))
        .collect();
    let server = tokio::spawn(async move {
        loop {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = BufReader::new(socket);
            let mut request = String::new();
            // A peer hanging up mid-request is not a test failure; drop the connection
            // rather than panicking this detached task.
            if socket.read_line(&mut request).await.is_err() {
                continue;
            }
            loop {
                let mut header = String::new();
                match socket.read_line(&mut header).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) if header == "\r\n" => break,
                    Ok(_) => {},
                }
            }
            let response = routes
                .iter()
                .find(|(expected, _)| request.trim_end() == expected)
                .map(|(_, response)| response.as_str())
                .unwrap_or("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            let _ = socket.get_mut().write_all(response.as_bytes()).await;
        }
    });
    let client = HFClient::builder()
        .endpoint(endpoint)
        .token("test-token")
        .retry_max_attempts(0)
        .build()
        .unwrap();
    (client, server)
}
