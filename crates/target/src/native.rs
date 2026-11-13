//! Native host compilation and W^X executable memory, backed by [`volt_host`].

use alloc::boxed::Box;
use core::fmt;
use volt_host::{
    ExecError, HostExecImage, instruction_barrier as host_barrier, map_code as host_map_code,
};
use volt_ir::function::Function;

#[cfg(any(
    all(target_arch = "x86_64", feature = "x86_64"),
    all(target_arch = "aarch64", feature = "aarch64")
))]
mod module;
#[cfg(all(
    feature = "aarch64",
    target_os = "linux",
    any(
        all(target_arch = "x86_64", feature = "x86_64"),
        target_arch = "aarch64"
    )
))]
pub use module::map_linked_aarch64;
#[cfg(any(
    all(target_arch = "x86_64", feature = "x86_64"),
    all(target_arch = "aarch64", feature = "aarch64")
))]
pub use module::{
    DataKind, DataReloc, Error as ModuleError, JittedModule, ModuleData, ModuleFunction,
    jit_module, jit_module_data,
};

pub type JittedFunction = HostExecImage;

#[derive(Debug)]
pub enum Error {
    Compile(Box<dyn core::error::Error + Send + Sync>),
    EmptyCode,
    Mapping(std::io::Error),
    Protection(std::io::Error),
    UnsupportedHost,
}

impl From<ExecError> for Error {
    fn from(value: ExecError) -> Self {
        match value {
            ExecError::EmptyCode => Self::EmptyCode,
            ExecError::Mapping(e) => Self::Mapping(e),
            ExecError::Protection(e) => Self::Protection(e),
            ExecError::UnsupportedHost => Self::UnsupportedHost,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "native compilation: {e}"),
            Self::EmptyCode => f.write_str("cannot map empty machine code"),
            Self::Mapping(e) => write!(f, "executable allocation: {e}"),
            Self::Protection(e) => write!(f, "executable protection: {e}"),
            Self::UnsupportedHost => {
                f.write_str("native JIT requires Linux or macOS x86_64/AArch64")
            }
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Compile(e) => Some(e.as_ref()),
            Self::Mapping(e) | Self::Protection(e) => Some(e),
            _ => None,
        }
    }
}

/// Copy generated code into writable pages, protect R+X, synchronize the host
/// instruction cache, and transfer ownership to an executable image.
pub fn map_code(code: impl AsRef<[u8]>) -> Result<JittedFunction, Error> {
    Ok(host_map_code(code)?)
}

/// Compile a Rust SSA function for the architecture executing this process.
pub fn compile(function: &Function) -> Result<JittedFunction, Error> {
    #[cfg(all(target_arch = "x86_64", feature = "x86_64"))]
    {
        let code = crate::x86_64::compile(function).map_err(|e| Error::Compile(Box::new(e)))?;
        map_code(code)
    }
    #[cfg(all(target_arch = "aarch64", feature = "aarch64"))]
    {
        let words = crate::aarch64::isel::compile_with_symbols(
            function,
            Some(&volt_host::resolve_host_symbol),
        )
        .map_err(|e| Error::Compile(Box::new(e)))?;
        let bytes =
            unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
        map_code(bytes)
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", feature = "x86_64"),
        all(target_arch = "aarch64", feature = "aarch64")
    )))]
    {
        let _ = function;
        Err(Error::UnsupportedHost)
    }
}

/// Like [`compile`], consuming `function` so the backend need not copy it.
pub fn compile_owned(function: Function) -> Result<JittedFunction, Error> {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64"))]
    {
        let words = crate::aarch64::isel::compile_owned_with_symbols(
            function,
            Some(&volt_host::resolve_host_symbol),
        )
        .map_err(|e| Error::Compile(Box::new(e)))?;
        let bytes =
            unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
        map_code(bytes)
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "aarch64")))]
    {
        compile(&function)
    }
}

/// Make instructions another thread wrote (and synchronized with
/// [`map_code`]) visible to instruction fetch on the calling thread.
pub fn instruction_barrier() {
    host_barrier();
}
