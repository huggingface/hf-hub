//! Crate-internal helpers.

use std::future::Future;

#[cfg(not(target_family = "wasm"))]
use futures::future::BoxFuture;
#[cfg(target_family = "wasm")]
use futures::future::LocalBoxFuture;

/// Box a heavyweight async-method body behind a type-erased pointer.
///
/// The download/upload/sync builders compile to state machines tens of kilobytes large. Awaiting
/// them inline inside the bon-generated `send()` future makes every consumer async fn embed that
/// state machine, growing the future and deepening its type until downstream crates hit rustc's
/// `queries overflow the depth limit!` error — even crates that never name hf-hub types. Boxing
/// here keeps the `send()` future small and type-erased so consumer future nesting stays shallow.
/// `hf-hub/tests/future_size.rs` guards against regressions.
///
/// Native futures are `Send`; the browser `reqwest` backend on `wasm32-unknown-unknown` produces
/// `!Send` futures, so the `Send` bound is dropped there.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn boxed_future<'a, F>(future: F) -> BoxFuture<'a, F::Output>
where
    F: Future + Send + 'a,
{
    Box::pin(future)
}

#[cfg(target_family = "wasm")]
pub(crate) fn boxed_future<'a, F>(future: F) -> LocalBoxFuture<'a, F::Output>
where
    F: Future + 'a,
{
    Box::pin(future)
}
