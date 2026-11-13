#![cfg_attr(not(feature = "std"), no_std)]

// NOTE: volt-ir's `dialect` module requires std (`pliron`). Core IR modules are no_std + alloc.
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod attribute;
pub mod bitcode;
pub mod builder;
pub mod critical_edge;
#[cfg(feature = "std")]
pub mod dialect;
pub mod entity;
pub mod expand;
pub mod function;
pub mod legalize;
pub mod low_float;
pub mod nvfp4;
pub mod parser;
pub mod reachable;
pub mod softfp;
pub mod types;
pub mod verify;
