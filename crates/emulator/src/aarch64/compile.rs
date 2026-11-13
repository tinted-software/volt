pub use volt_emulator_core::aarch64::compile::{Error, translate_block};
use volt_host::{HostExecImage, HostExecMemory};

pub type Block = volt_emulator_core::aarch64::compile::Block<HostExecImage>;

pub fn compile(guest_pc: u64, bytes: &[u8]) -> Result<Block, Error> {
    compile_with(guest_pc, bytes, false)
}

pub fn compile_with(guest_pc: u64, bytes: &[u8], inline_memory: bool) -> Result<Block, Error> {
    volt_emulator_core::aarch64::compile::compile_with_exec(
        &HostExecMemory,
        guest_pc,
        bytes,
        inline_memory,
    )
}
