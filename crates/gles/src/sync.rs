//! `no_std` synchronization primitives for the workspace.
//!
//! Thin re-export of the `spin` crate: `spin::Mutex`/`spin::RwLock` are
//! themselves `lock_api` facades over maintained raw spin locks, giving the
//! same API shape the crate has always used (`Mutex::new` const-statics,
//! `.lock()` guards). Critical sections in these crates are short (state
//! updates, queue submits), so spinning is acceptable; a futex- or
//! event-based lock can be swapped in per-platform later without touching
//! call sites. History: this module hand-rolled `RawSpinMutex`/
//! `RawSpinRwLock` on top of `lock_api` before `spin` was adopted.

pub use spin::{Mutex, RwLock};
