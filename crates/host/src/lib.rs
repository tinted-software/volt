//! Host OS implementations of [`volt_platform`] traits (`std`, Unix).

extern crate alloc;

pub mod block;
pub mod clock;
pub mod console;
pub mod exec;
pub mod ram;
pub mod surface;

pub use block::FileBackend;
pub use clock::StdClock;
pub use console::StdoutConsole;
pub use exec::{
    ExecError, HostExecImage, HostExecMemory, instruction_barrier, map_code, resolve_host_symbol,
    sync_icache,
};
pub use ram::MmapRam;
pub use surface::{DumpSurface, WindowSurface};
