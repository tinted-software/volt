//! The `volt` dialect: volt IR modeled with pliron.
//!
//! [`types`] defines the volt type system as pliron types, [`attrs`]
//! defines operation payload attributes, [`ops`] defines every volt
//! operation, and [`func`] provides the function-level container and
//! builder used by passes, targets, and the emulator.

pub mod attrs;
pub mod func;
pub mod ops;
pub mod types;

/// The volt dialect name, used as the `volt.` prefix on ops, types,
/// and attributes.
pub const DIALECT_NAME: &str = "volt";
