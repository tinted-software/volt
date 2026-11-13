//! AMDGPU GFX10.3 (RDNA2) target backend for [`volt_codegen`].
#![no_std]

extern crate alloc;

pub mod gfx10;

pub use gfx10::{Gfx10, Gfx10Error, for_processor, gfx103_processor};

use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use volt_codegen::targets::Signature;
use volt_codegen::{CompiledShader, Error};

pub fn compile(
    context: &Context,
    module: Ptr<Operation>,
    signature: &Signature,
    processor: &str,
) -> Result<CompiledShader, Error<Gfx10Error>> {
    compile_with_listing(context, module, signature, processor, false)
}

pub fn compile_with_listing(
    context: &Context,
    module: Ptr<Operation>,
    signature: &Signature,
    processor: &str,
    include_listing: bool,
) -> Result<CompiledShader, Error<Gfx10Error>> {
    let target = for_processor(processor).ok_or(Error::UnknownProcessor)?;
    volt_codegen::compile_for_target(context, module, signature, &target, include_listing)
}
