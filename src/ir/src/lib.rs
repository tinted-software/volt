// NOTE: volt-ir requires std: pliron (and its dependencies) need std to
// build. no_std support is deferred until pliron can build without std.
extern crate alloc;

pub mod attribute;
pub mod bitcode;
pub mod builder;
pub mod critical_edge;
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
