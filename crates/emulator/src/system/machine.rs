use crate::memory::GuestMemory;
use alloc::sync::Arc;
use volt_host::HostExecMemory;

pub use volt_emulator_core::system::machine::{DeviceIo, Error, Exit, TraceEntry};

pub struct Machine<M>(volt_emulator_core::system::machine::Machine<M, HostExecMemory>);

impl<M> core::ops::Deref for Machine<M> {
    type Target = volt_emulator_core::system::machine::Machine<M, HostExecMemory>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<M> core::ops::DerefMut for Machine<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<M> Machine<M> {
    pub fn into_inner(self) -> volt_emulator_core::system::machine::Machine<M, HostExecMemory> {
        self.0
    }
}

impl<M: GuestMemory> Machine<M> {
    pub fn new(memory: M) -> Self {
        let inline_memory = !std::env::var_os("VOLT_INLINE_MEMORY").is_some_and(|v| v == "0");
        let m = volt_emulator_core::system::machine::Machine::new_with(
            memory,
            HostExecMemory,
            inline_memory,
        );
        if let Some(reader) = m.memory.code_reader() {
            crate::aarch64::cache::enable_ahead(
                m.cache.shared(),
                reader,
                crate::aarch64::ahead::default_threads(),
            );
        }
        Self(m)
    }

    pub fn with_shared_blocks(
        memory: M,
        cpu_id: u32,
        blocks: Arc<crate::aarch64::cache::SharedBlocks>,
    ) -> Self {
        Self(
            volt_emulator_core::system::machine::Machine::with_shared_blocks(
                memory, cpu_id, blocks,
            ),
        )
    }

    pub fn with_cpu_id(memory: M, cpu_id: u32) -> Self {
        let mut m = Self::new(memory);
        m.0.cpu_id = cpu_id;
        m
    }
}
