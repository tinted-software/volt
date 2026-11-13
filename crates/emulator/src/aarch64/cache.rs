use alloc::sync::Arc;
use volt_host::HostExecMemory;

pub use volt_emulator_core::aarch64::cache::{AheadCompiler, CodeReader, Fetch};

pub type Cache = volt_emulator_core::aarch64::cache::Cache<HostExecMemory>;
pub type SharedBlocks = volt_emulator_core::aarch64::cache::SharedBlocks<HostExecMemory>;

pub fn new_shared(inline_memory: bool) -> SharedBlocks {
    SharedBlocks::new(HostExecMemory, inline_memory)
}

pub fn enable_ahead(shared: &Arc<SharedBlocks>, reader: Arc<dyn CodeReader>, threads: usize) {
    if threads > 0 {
        let ahead = super::ahead::Ahead::start(shared, reader, threads);
        shared.attach_ahead(ahead);
    }
}
