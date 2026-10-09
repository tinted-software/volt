//! A translation unit: named functions plus named data globals.
//!
//! [`Module`] is the single description of a unit of code and data shared by
//! every consumer (object writers, in-process linkers, JITs). Symbols that are
//! referenced but not defined here (external functions or data) are not an
//! error at this level; the linker or host resolves them.

use alloc::string::{String, ToString as _};
use alloc::vec::Vec;
use core::fmt;

use crate::function::Function;
use crate::verify::{self, Diagnostic, Profile};

/// Size in bytes of a data relocation slot: a pointer-sized absolute address.
pub const RELOC_SLOT_SIZE: u64 = 8;

/// The section a data global lands in: read-only, writable, or zero-initialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataKind {
    Rodata,
    Data,
    Bss,
}

/// An internal relocation within a global. At byte `offset` within the global,
/// the linker writes the pointer-sized absolute runtime address of `symbol`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataReloc {
    pub offset: usize,
    pub symbol: String,
}

/// A named data global. For [`DataKind::Bss`], `bytes` is empty and `size`
/// gives the zero-initialized length; otherwise `size == bytes.len()`.
///
/// `align` is the required alignment in bytes: `0` selects the natural
/// alignment ([`natural_align`] of `size`), otherwise it must be a power of two.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Global {
    pub name: String,
    pub kind: DataKind,
    pub bytes: Vec<u8>,
    pub size: u64,
    pub align: u32,
    pub relocs: Vec<DataReloc>,
}

impl Global {
    /// The alignment this global is laid out with: `align`, or the natural
    /// alignment of `size` when `align` is `0`.
    pub fn effective_align(&self) -> usize {
        match self.align {
            0 => natural_align(self.size),
            a => a as usize,
        }
    }
}

/// A data object's natural alignment: the next power of two up to `size`,
/// capped at 8. Covers scalars and pointers up to a 64-bit word.
pub fn natural_align(size: u64) -> usize {
    size.max(1).checked_next_power_of_two().unwrap_or(8).min(8) as usize
}

/// A named function definition.
#[derive(Clone, Debug)]
pub struct FunctionDef {
    pub name: String,
    pub function: Function,
}

/// Why [`Module::verify`] rejected a module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModuleError {
    /// Two definitions (function or global) share a name.
    DuplicateName(String),
    /// A non-bss global's `size` differs from its `bytes.len()`.
    SizeMismatch { name: String, size: u64, len: usize },
    /// A bss global carries bytes or relocations.
    BssInitialized(String),
    /// A relocation slot does not fit within its global.
    RelocOutOfRange { name: String, offset: usize },
    /// A global's `align` is neither `0` nor a power of two.
    InvalidAlign { name: String, align: u32 },
    /// A function failed IR verification.
    InvalidFunction {
        name: String,
        diagnostics: Vec<Diagnostic>,
    },
}

impl fmt::Display for ModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "duplicate module symbol {name}"),
            Self::SizeMismatch { name, size, len } => {
                write!(
                    f,
                    "global {name} has size {size} but {len} initializer bytes"
                )
            }
            Self::BssInitialized(name) => {
                write!(f, "bss global {name} has initializer bytes or relocations")
            }
            Self::RelocOutOfRange { name, offset } => {
                write!(
                    f,
                    "relocation at offset {offset} does not fit in global {name}"
                )
            }
            Self::InvalidAlign { name, align } => {
                write!(f, "global {name} has invalid alignment {align}")
            }
            Self::InvalidFunction { name, diagnostics } => {
                write!(f, "function {name} failed verification")?;
                if let Some(first) = diagnostics.first() {
                    write!(f, ": {first:?}")?;
                    if diagnostics.len() > 1 {
                        write!(f, " (and {} more)", diagnostics.len() - 1)?;
                    }
                }
                Ok(())
            }
        }
    }
}

impl core::error::Error for ModuleError {}

/// A set of named functions and data globals. Function order is preserved;
/// linkers place the first function at the start of the code image.
#[derive(Clone, Debug, Default)]
pub struct Module {
    pub functions: Vec<FunctionDef>,
    pub globals: Vec<Global>,
}

impl Module {
    pub fn new() -> Module {
        Module::default()
    }

    pub fn add_function(&mut self, name: &str, function: Function) {
        self.functions.push(FunctionDef {
            name: name.to_string(),
            function,
        });
    }

    /// Add a named read-only data blob (a global constant).
    pub fn add_data(&mut self, name: &str, bytes: Vec<u8>) {
        self.add_data_relocs(name, bytes, Vec::new());
    }

    /// Add a named read-only data blob carrying internal relocations.
    pub fn add_data_relocs(&mut self, name: &str, bytes: Vec<u8>, relocs: Vec<DataReloc>) {
        self.add_initialized(name, DataKind::Rodata, bytes, relocs);
    }

    /// Add a named writable data global.
    pub fn add_writable(&mut self, name: &str, bytes: Vec<u8>) {
        self.add_writable_relocs(name, bytes, Vec::new());
    }

    /// Add a named writable data global carrying internal relocations.
    pub fn add_writable_relocs(&mut self, name: &str, bytes: Vec<u8>, relocs: Vec<DataReloc>) {
        self.add_initialized(name, DataKind::Data, bytes, relocs);
    }

    /// Add a named zero-initialized data global of `size` bytes.
    pub fn add_bss(&mut self, name: &str, size: u64) {
        self.globals.push(Global {
            name: name.to_string(),
            kind: DataKind::Bss,
            bytes: Vec::new(),
            size,
            align: 0,
            relocs: Vec::new(),
        });
    }

    fn add_initialized(
        &mut self,
        name: &str,
        kind: DataKind,
        bytes: Vec<u8>,
        relocs: Vec<DataReloc>,
    ) {
        self.globals.push(Global {
            name: name.to_string(),
            kind,
            size: bytes.len() as u64,
            bytes,
            align: 0,
            relocs,
        });
    }

    /// The function defined under `name`.
    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions
            .iter()
            .find(|f| f.name == name)
            .map(|f| &f.function)
    }

    /// The global defined under `name`.
    pub fn global(&self, name: &str) -> Option<&Global> {
        self.globals.iter().find(|g| g.name == name)
    }

    /// Check module-level well-formedness and verify every function.
    ///
    /// Undefined symbols (calls, `global_addr`, data relocations naming
    /// something not defined here) are allowed: they are external.
    pub fn verify(&self) -> Result<(), ModuleError> {
        let mut names: Vec<&str> = self
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .chain(self.globals.iter().map(|g| g.name.as_str()))
            .collect();
        names.sort_unstable();
        if let Some(pair) = names.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(ModuleError::DuplicateName(pair[0].to_string()));
        }
        for g in &self.globals {
            self.verify_global(g)?;
        }
        for f in &self.functions {
            let diagnostics = verify::verify(&f.function, Profile::High);
            if !diagnostics.ok() {
                return Err(ModuleError::InvalidFunction {
                    name: f.name.clone(),
                    diagnostics: diagnostics.items().to_vec(),
                });
            }
        }
        Ok(())
    }

    fn verify_global(&self, g: &Global) -> Result<(), ModuleError> {
        if g.align != 0 && !g.align.is_power_of_two() {
            return Err(ModuleError::InvalidAlign {
                name: g.name.clone(),
                align: g.align,
            });
        }
        if g.kind == DataKind::Bss {
            if !g.bytes.is_empty() || !g.relocs.is_empty() {
                return Err(ModuleError::BssInitialized(g.name.clone()));
            }
            return Ok(());
        }
        if g.size != g.bytes.len() as u64 {
            return Err(ModuleError::SizeMismatch {
                name: g.name.clone(),
                size: g.size,
                len: g.bytes.len(),
            });
        }
        for r in &g.relocs {
            let fits = (r.offset as u64)
                .checked_add(RELOC_SLOT_SIZE)
                .is_some_and(|end| end <= g.size);
            if !fits {
                return Err(ModuleError::RelocOutOfRange {
                    name: g.name.clone(),
                    offset: r.offset,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::function::{MemFlags, Ret, Terminator};

    fn constant(value: i64) -> Function {
        let mut f = Function::new();
        let ty = f.types.parse_type("i64").unwrap();
        let block = f.append_block();
        let v = f.append_inst(block, ty, crate::function::Opcode::Iconst(value));
        f.set_terminator(block, Terminator::Ret(Ret::one(v)));
        f
    }

    /// A function the verifier rejects: a load with a non-power-of-two `align`.
    fn malformed() -> Function {
        let mut f = Function::new();
        let ty = f.types.parse_type("i32").unwrap();
        let ptr = f.types.ptr_global();
        let entry = f.append_block();
        let p = f.append_block_param(entry, ptr);
        let mem = MemFlags {
            align: 3,
            ..MemFlags::new()
        };
        let v = f.append_load(entry, ty, p, mem);
        f.set_terminator(entry, Terminator::Ret(Ret::one(v)));
        f
    }

    #[test]
    fn natural_alignment_covers_scalars() {
        assert_eq!(natural_align(0), 1);
        assert_eq!(natural_align(1), 1);
        assert_eq!(natural_align(3), 4);
        assert_eq!(natural_align(8), 8);
        assert_eq!(natural_align(100), 8);
        assert_eq!(natural_align(u64::MAX), 8);
    }

    #[test]
    fn builders_fill_kind_size_and_lookup() {
        let mut m = Module::new();
        m.add_function("f", constant(1));
        m.add_data("ro", vec![1, 2, 3]);
        m.add_writable_relocs(
            "rw",
            vec![0; 8],
            vec![DataReloc {
                offset: 0,
                symbol: "f".to_string(),
            }],
        );
        m.add_bss("zero", 64);
        assert!(m.function("f").is_some());
        assert!(m.function("ro").is_none());
        let ro = m.global("ro").unwrap();
        assert_eq!((ro.kind, ro.size, ro.align), (DataKind::Rodata, 3, 0));
        assert_eq!(ro.effective_align(), 4);
        let rw = m.global("rw").unwrap();
        assert_eq!((rw.kind, rw.size, rw.relocs.len()), (DataKind::Data, 8, 1));
        let zero = m.global("zero").unwrap();
        assert_eq!(
            (zero.kind, zero.size, zero.bytes.len()),
            (DataKind::Bss, 64, 0)
        );
        assert_eq!(zero.effective_align(), 8);
        assert!(m.global("f").is_none());
        assert_eq!(m.verify(), Ok(()));
    }

    #[test]
    fn explicit_alignment_overrides_natural() {
        let mut m = Module::new();
        m.add_bss("big", 64);
        m.globals[0].align = 64;
        assert_eq!(m.globals[0].effective_align(), 64);
        assert_eq!(m.verify(), Ok(()));
    }

    #[test]
    fn undefined_symbols_are_external_not_errors() {
        let mut m = Module::new();
        m.add_data_relocs(
            "table",
            vec![0; 16],
            vec![DataReloc {
                offset: 8,
                symbol: "elsewhere".to_string(),
            }],
        );
        assert_eq!(m.verify(), Ok(()));
    }

    #[test]
    fn rejects_duplicate_names_across_functions_and_globals() {
        let mut m = Module::new();
        m.add_function("x", constant(1));
        m.add_data("x", vec![0]);
        assert_eq!(m.verify(), Err(ModuleError::DuplicateName("x".to_string())));

        let mut m = Module::new();
        m.add_function("f", constant(1));
        m.add_function("f", constant(2));
        assert_eq!(m.verify(), Err(ModuleError::DuplicateName("f".to_string())));

        let mut m = Module::new();
        m.add_bss("b", 8);
        m.add_writable("b", vec![0]);
        assert_eq!(m.verify(), Err(ModuleError::DuplicateName("b".to_string())));
    }

    #[test]
    fn rejects_size_mismatch_and_initialized_bss() {
        let mut m = Module::new();
        m.add_data("d", vec![0; 4]);
        m.globals[0].size = 8;
        assert_eq!(
            m.verify(),
            Err(ModuleError::SizeMismatch {
                name: "d".to_string(),
                size: 8,
                len: 4
            })
        );

        let mut m = Module::new();
        m.add_bss("z", 8);
        m.globals[0].bytes = vec![0; 8];
        assert_eq!(
            m.verify(),
            Err(ModuleError::BssInitialized("z".to_string()))
        );

        let mut m = Module::new();
        m.add_bss("z", 16);
        m.globals[0].relocs.push(DataReloc {
            offset: 0,
            symbol: "f".to_string(),
        });
        assert_eq!(
            m.verify(),
            Err(ModuleError::BssInitialized("z".to_string()))
        );
    }

    #[test]
    fn relocation_slot_must_fit_in_the_global() {
        let reloc = |offset| DataReloc {
            offset,
            symbol: "s".to_string(),
        };
        let mut m = Module::new();
        m.add_data_relocs("fits", vec![0; 16], vec![reloc(0), reloc(8)]);
        assert_eq!(m.verify(), Ok(()));

        for offset in [9, 16, usize::MAX] {
            let mut m = Module::new();
            m.add_data_relocs("d", vec![0; 16], vec![reloc(offset)]);
            assert_eq!(
                m.verify(),
                Err(ModuleError::RelocOutOfRange {
                    name: "d".to_string(),
                    offset
                }),
                "offset {offset}"
            );
        }
        let mut m = Module::new();
        m.add_data_relocs("short", vec![0; 4], vec![reloc(0)]);
        assert!(matches!(
            m.verify(),
            Err(ModuleError::RelocOutOfRange { .. })
        ));
    }

    #[test]
    fn alignment_must_be_zero_or_a_power_of_two() {
        for align in [0, 1, 2, 16, 4096] {
            let mut m = Module::new();
            m.add_data("d", vec![0; 4]);
            m.globals[0].align = align;
            assert_eq!(m.verify(), Ok(()), "align {align}");
        }
        for align in [3, 6, 12] {
            let mut m = Module::new();
            m.add_data("d", vec![0; 4]);
            m.globals[0].align = align;
            assert_eq!(
                m.verify(),
                Err(ModuleError::InvalidAlign {
                    name: "d".to_string(),
                    align
                }),
                "align {align}"
            );
        }
    }

    #[test]
    fn function_diagnostics_surface_in_the_error() {
        let mut m = Module::new();
        m.add_function("ok", constant(1));
        m.add_function("bad", malformed());
        match m.verify() {
            Err(ModuleError::InvalidFunction { name, diagnostics }) => {
                assert_eq!(name, "bad");
                assert_eq!(diagnostics.len(), 1);
                assert!(matches!(diagnostics[0], Diagnostic::InvalidMemFlags(_)));
            }
            other => panic!("expected InvalidFunction, got {other:?}"),
        }
    }

    #[test]
    fn errors_display_and_implement_error() {
        fn is_error<E: core::error::Error>(_: &E) {}
        let e = ModuleError::DuplicateName("x".to_string());
        is_error(&e);
        assert_eq!(e.to_string(), "duplicate module symbol x");
        let mut m = Module::new();
        m.add_function("bad", malformed());
        let text = m.verify().unwrap_err().to_string();
        assert!(text.starts_with("function bad failed verification: InvalidMemFlags"));
    }
}
