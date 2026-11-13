//! Pure-Rust GPU machine-code generation from pliron's LLVM dialect.
//!
//! `pliron-llvm` defines the input dialect; this crate does not link LLVM.
//! A target supplies the hardware ABI, pseudo-instruction expansion, machine
//! encoding, and loader-visible register configuration through [`Target`].

#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use pliron::context::{Context, Ptr};
use pliron::operation::Operation;

pub mod instruction_selection;
pub mod machine;
pub mod register_allocation;
pub mod targets;

use instruction_selection::SelectionError;
use register_allocation::RegisterAllocationError;
use targets::{Signature, Summary, Target};

#[derive(Debug)]
pub enum Error<E> {
    UnknownProcessor,
    InstructionSelection(SelectionError),
    RegisterAllocation(RegisterAllocationError),
    TargetExpansion(E),
    Encoding(E),
}

impl<E: fmt::Display> fmt::Display for Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownProcessor => f.write_str("unknown GPU processor"),
            Self::InstructionSelection(e) => write!(f, "instruction selection: {e:?}"),
            Self::RegisterAllocation(e) => write!(f, "register allocation: {e:?}"),
            Self::TargetExpansion(e) => write!(f, "target expansion: {e}"),
            Self::Encoding(e) => write!(f, "encoding: {e}"),
        }
    }
}

/// Machine code and the registers the driver must program for its entry ABI.
pub struct CompiledShader {
    pub code: Vec<u8>,
    pub configuration: Vec<(u32, u32)>,
    /// Human-readable instruction listing, allocated only when requested.
    pub assembly: Option<String>,
}

/// Compile against a [`Target`] implementation.
pub fn compile_for_target<T: Target>(
    context: &Context,
    module: Ptr<Operation>,
    signature: &Signature,
    target: &T,
    include_listing: bool,
) -> Result<CompiledShader, Error<T::Error>> {
    let mut function =
        instruction_selection::select_instructions(context, module, signature, target)
            .map_err(Error::InstructionSelection)?;
    target
        .expand_pseudo_instructions(&mut function)
        .map_err(Error::TargetExpansion)?;
    let allocation = register_allocation::allocate_registers(&function, target)
        .map_err(Error::RegisterAllocation)?;
    let summary = Summary::from_function(&function, &allocation);
    let code = target
        .encode_function(&function, &allocation)
        .map_err(Error::Encoding)?;
    let assembly = if include_listing {
        let mut listing = String::new();
        for instruction_index in 0..function.instructions.len() {
            if instruction_index != 0 {
                listing.push('\n');
            }
            listing.push_str(&target.format_instruction(&function, &allocation, instruction_index));
        }
        Some(listing)
    } else {
        None
    };
    Ok(CompiledShader {
        code,
        configuration: target.configuration(signature.stage, &summary),
        assembly,
    })
}
