//! Pliron LLVM-dialect validation and conversion boundary.
//!
//! Architecture frontends construct a `ModuleOp` containing LLVM dialect IR.
//! This crate verifies, lowers it to LLVM IR, and owns native JIT lifetime.

use pliron::{
    builtin::ops::ModuleOp, context::Context, op::Op, operation::verify_operation, result::Result,
};
use pliron_llvm::{
    from_llvm_ir,
    llvm_sys::{
        core::{LLVMContext, LLVMModule},
        lljit::{JitSymbol, SimpleJIT},
    },
    to_llvm_ir,
};

/// Backend failures after Pliron verification.
#[derive(Debug)]
pub enum BackendError {
    /// Invalid LLVM dialect/module IR.
    Pliron(pliron::result::Error),
    /// Invalid LLVM IR or failed native JIT setup.
    Llvm(String),
}

impl core::fmt::Display for BackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Pliron(error) => write!(f, "Pliron module error: {error}"),
            Self::Llvm(error) => write!(f, "LLVM backend error: {error}"),
        }
    }
}

impl core::error::Error for BackendError {}

/// Verify Pliron IR and lower its LLVM dialect operations to an LLVM module.
pub fn lower_module(ctx: &Context, llvm_ctx: &LLVMContext, module: ModuleOp) -> Result<LLVMModule> {
    verify_operation(module.get_operation(), ctx)?;
    to_llvm_ir::convert_module(ctx, llvm_ctx, module)
}

/// Native LLVM JIT holding translated executable code alive.
pub struct JitEngine {
    jit: SimpleJIT,
}

impl JitEngine {
    /// Compile a verified Pliron LLVM-dialect module for the current host.
    pub fn compile(ctx: &Context, module: ModuleOp) -> core::result::Result<Self, BackendError> {
        let llvm_ctx = LLVMContext::default();
        let llvm_module = lower_module(ctx, &llvm_ctx, module).map_err(BackendError::Pliron)?;
        llvm_module.verify().map_err(BackendError::Llvm)?;
        let jit = SimpleJIT::new(llvm_ctx, llvm_module).map_err(BackendError::Llvm)?;
        Ok(Self { jit })
    }

    /// Parse LLVM IR into the Pliron LLVM dialect, then lower and JIT it.
    /// Architecture-specific frontend crates use this path for their generated
    /// function modules.
    pub fn compile_llvm_ir(ir: &str) -> core::result::Result<Self, BackendError> {
        let llvm_ctx = LLVMContext::default();
        let parsed = LLVMModule::from_ir_in_str(&llvm_ctx, ir, Some("mirage-jit"))
            .map_err(BackendError::Llvm)?;
        let mut ctx = Context::new();
        let module =
            from_llvm_ir::convert_module(&mut ctx, &parsed).map_err(BackendError::Pliron)?;
        Self::compile(&ctx, module)
    }

    /// Look up a compiled symbol with the caller-specified ABI.
    ///
    /// # Safety
    /// `F` must exactly match the function's generated ABI and signature.
    pub unsafe fn lookup<F: Copy>(
        &self,
        name: &str,
    ) -> core::result::Result<JitSymbol<'_, F>, String> {
        unsafe { self.jit.lookup_symbol(name) }
    }
}

#[cfg(test)]
mod tests {
    use pliron::context::Context;
    use pliron_llvm::{
        from_llvm_ir,
        llvm_sys::core::{LLVMContext, LLVMModule},
    };

    use super::JitEngine;

    #[test]
    fn pliron_round_trip_executes_through_native_jit() {
        let llvm_ctx = LLVMContext::default();
        let original = LLVMModule::from_ir_in_str(
            &llvm_ctx,
            "define i64 @add(i64 %a, i64 %b) { entry: %sum = add i64 %a, %b ret i64 %sum }",
            Some("jit-smoke"),
        )
        .expect("valid LLVM test module");
        let mut ctx = Context::new();
        let module = from_llvm_ir::convert_module(&mut ctx, &original)
            .expect("convert LLVM module to Pliron");
        let jit = JitEngine::compile(&ctx, module).expect("compile Pliron module");
        let add =
            unsafe { jit.lookup::<fn(i64, i64) -> i64>("add") }.expect("lookup compiled function");
        assert_eq!(add(19, 23), 42);
    }
}
