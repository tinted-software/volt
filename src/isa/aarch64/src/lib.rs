//! The AArch64 (A64) instruction set: what an instruction word *is*.
//!
//! This crate owns architectural knowledge and nothing else. It does not know
//! about guest state, memory, block translation, register allocation or
//! executable memory; consumers (disassembly, lifting, emulation) decode a word
//! once here and work from the result.
//!
//! * [`decode`] turns an instruction word into a structured
//!   [`decode::Instruction`]. It is the only decoder: [`fp`], [`simd`], [`pauth`]
//!   and [`gxf`] hold the instruction families it delegates to.
//! * [`format`] prints a decoded instruction as assembly text (`objdump` spelling).
//! * [`flow`] answers the control-flow question for analysis.
//! * [`simd_struct`] describes SIMD structure loads and stores.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod decode;
pub mod flow;
pub mod format;
pub mod fp;
pub mod gxf;
pub mod pauth;
pub mod simd;
pub mod simd_struct;
