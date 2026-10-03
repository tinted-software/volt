extern crate alloc;

pub mod attribute;
pub mod builder;
pub mod critical_edge;
pub mod entity;
pub mod expand;
pub mod function;
pub mod legalize;
pub mod low_float;
pub mod nvfp4;
pub mod reachable;
pub mod softfp;
pub mod types;
pub mod verify;

// Parser (text) and bitcode (text+binary round-trip) are DEFERRED:
// porting them is ~5k lines of format code; the aarch64 isel path builds IR
// via `builder` and does not need them. They will land as `parser`/`bitcode`
// modules in a follow-up.
