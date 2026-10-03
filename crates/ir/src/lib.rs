#![no_std]

extern crate alloc;

pub mod attribute;
pub mod bitcode;
pub mod builder;
pub mod critical_edge;
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
