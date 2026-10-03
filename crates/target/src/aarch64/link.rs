//! AArch64 static/JIT linker: concatenate pre-compiled functions, resolve
//! intra-module calls, lay out data globals, and carry `global_addr`
//! relocations forward.
//!
//! IR-free: [`Module`] takes pre-compiled [`Vec<u32>`] function bodies, not
//! `vulcan-ir` functions. Compiling IR to words is the (stubbed) `isel`
//! stage's job; linking starts after that.

use alloc::string::String;
use alloc::vec::Vec;

use super::encode;
use super::isel::{Reloc, RelocKind};

/// Linker failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// An isel-level lowering is not supported.
    Unsupported,
    /// A `call` relocation names a function not present in the module.
    UndefinedSymbol,
    /// A GOT-indirect relocation reached the in-process linker (resolved only
    /// by a real dynamic linker/loader).
    UnsupportedReloc,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Unsupported => write!(f, "unsupported construct"),
            Error::UndefinedSymbol => write!(f, "undefined symbol"),
            Error::UnsupportedReloc => write!(f, "unsupported relocation"),
        }
    }
}

impl core::error::Error for Error {}

/// The section a data global lands in: read-only, writable, or zero-initialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataKind {
    Rodata,
    Data,
    Bss,
}

/// An internal relocation within a data object. At byte offset `off` within
/// the object, the linker must write a pointer-sized absolute runtime address
/// of `symbol` once the module is mapped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataReloc {
    pub off: usize,
    pub symbol: String,
}

/// A named data global placed in the linked output. For `.bss`, `bytes` is
/// empty and `size` gives the zero-initialized length; otherwise
/// `size == bytes.len` as `u64`.
#[derive(Clone, Debug)]
pub struct Data {
    pub name: String,
    pub bytes: Vec<u8>,
    pub kind: DataKind,
    pub size: u64,
    pub relocs: Vec<DataReloc>,
}

/// A set of named functions and data globals to link together. The first added
/// function is the entry point at offset 0 of the linked image.
#[derive(Clone, Debug, Default)]
pub struct Module {
    pub functions: Vec<(String, Vec<u32>)>,
    pub data: Vec<Data>,
}

impl Module {
    pub fn new() -> Module {
        Module::default()
    }

    pub fn add_function(&mut self, name: &str, code: Vec<u32>) {
        self.functions.push((name.to_string(), code));
    }

    /// Add a named read-only data blob (a global constant) into `.rodata`.
    pub fn add_data(&mut self, name: &str, bytes: Vec<u8>) {
        let size = bytes.len() as u64;
        self.data.push(Data {
            name: name.to_string(),
            bytes,
            kind: DataKind::Rodata,
            size,
            relocs: Vec::new(),
        });
    }

    /// Add a named read-only data blob carrying internal relocations.
    pub fn add_data_relocs(&mut self, name: &str, bytes: Vec<u8>, relocs: Vec<DataReloc>) {
        let size = bytes.len() as u64;
        self.data.push(Data {
            name: name.to_string(),
            bytes,
            kind: DataKind::Rodata,
            size,
            relocs,
        });
    }

    /// Add a named writable data global into `.data`.
    pub fn add_writable(&mut self, name: &str, bytes: Vec<u8>) {
        let size = bytes.len() as u64;
        self.data.push(Data {
            name: name.to_string(),
            bytes,
            kind: DataKind::Data,
            size,
            relocs: Vec::new(),
        });
    }

    /// Add a named writable data global carrying internal relocations.
    pub fn add_writable_relocs(&mut self, name: &str, bytes: Vec<u8>, relocs: Vec<DataReloc>) {
        let size = bytes.len() as u64;
        self.data.push(Data {
            name: name.to_string(),
            bytes,
            kind: DataKind::Data,
            size,
            relocs,
        });
    }

    /// Add a named zero-initialized data global of `size` bytes into `.bss`.
    pub fn add_bss(&mut self, name: &str, size: u64) {
        self.data.push(Data {
            name: name.to_string(),
            bytes: Vec::new(),
            kind: DataKind::Bss,
            size,
            relocs: Vec::new(),
        });
    }
}

/// A function's byte offset within the linked image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub offset: usize,
}

/// A data global's placement within its section. `off` is the byte offset
/// within that section. `bytes` is empty for `.bss`.
#[derive(Clone, Debug)]
pub struct DataSym {
    pub name: String,
    pub kind: DataKind,
    pub off: usize,
    pub size: usize,
    pub bytes: Vec<u8>,
    pub relocs: Vec<DataReloc>,
}

/// A linked code image: the machine words, each function's byte offset, each
/// data global's per-section placement, and any still-unresolved
/// `global_addr` relocations (word indices into `code`).
#[derive(Clone, Debug, Default)]
pub struct Linked {
    pub code: Vec<u32>,
    pub symbols: Vec<Symbol>,
    pub data: Vec<DataSym>,
    pub relocs: Vec<Reloc>,
}

impl Linked {
    /// The byte offset of the function named `name`, or `None` if absent.
    pub fn address_of(&self, name: &str) -> Option<usize> {
        self.symbols
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.offset)
    }
}

/// A data object's natural alignment: the next power of two up to `size`,
/// capped at 8. Covers scalars and pointers up to a 64-bit word.
pub fn align_of_data(size: u64) -> usize {
    let size = size.max(1);
    let pow = size.next_power_of_two();
    pow.min(8) as usize
}

fn align_forward(off: usize, align: usize) -> usize {
    let mask = align - 1;
    (off + mask) & !mask
}

/// Patch a carried-forward `global_addr` relocation now that the symbol's
/// runtime address is known.
///
/// `code` is the linked word buffer; `r.offset` is a word index into it.
/// `site_addr` is the runtime address of the `adrp` instruction (base of the
/// page-relative delta); `target_addr` is the symbol's resolved address.
/// Edits the word in place, preserving every other bit.
///
/// `.call` relocations never reach here (resolved by `compile_module`);
/// GOT-indirect relocations are rejected there, so reaching one is a bug.
pub fn apply_global_reloc(code: &mut [u32], r: &Reloc, site_addr: usize, target_addr: usize) {
    let word = &mut code[r.offset];
    match r.kind {
        RelocKind::AdrpPg => {
            let site_page = site_addr & !0xFFFusize;
            let target_page = target_addr & !0xFFFusize;
            let delta_pages = (target_page as i64 - site_page as i64) / 4096;
            let u = delta_pages as u32;
            let immlo = u & 0x3;
            let immhi = (u >> 2) & 0x7FFFF;
            *word = (*word & !((0x3 << 29) | (0x7FFFF << 5))) | (immlo << 29) | (immhi << 5);
        }
        RelocKind::AddPgOff => {
            let lo12 = (target_addr & 0xFFF) as u32;
            *word = (*word & !(0xFFF << 10)) | (lo12 << 10);
        }
        RelocKind::Call => unreachable!("call relocations never carry forward"),
        RelocKind::GotPg | RelocKind::GotLo12 => {
            unreachable!("GOT relocations need a dynamic linker")
        }
    }
}

/// Concatenate the module's functions, resolve each intra-module `call`
/// relocation to a PC-relative `bl`, lay out data globals into per-section
/// `DataSym`s, and carry every `global_addr` relocation forward unresolved.
///
/// `relocs` are supplied alongside the module: `relocs[i]` holds the isel
/// relocations for `module.functions[i]`.
pub fn compile_module(module: &Module, relocs: &[Vec<Reloc>]) -> Result<Linked, Error> {
    let n = module.functions.len();
    let mut word_off = Vec::with_capacity(n);
    let mut total = 0usize;
    for (_, code) in module.functions.iter() {
        word_off.push(total);
        total += code.len();
    }
    let mut code: Vec<u32> = Vec::with_capacity(total);
    for (_, words) in module.functions.iter() {
        code.extend_from_slice(words);
    }
    let mut global_relocs: Vec<Reloc> = Vec::new();
    for (i, rs) in relocs.iter().enumerate() {
        for r in rs {
            let at = word_off[i] + r.offset;
            match r.kind {
                RelocKind::Call => {
                    let target_word = module
                        .functions
                        .iter()
                        .enumerate()
                        .find(|(_, (name, _))| *name == r.symbol)
                        .map(|(j, _)| word_off[j])
                        .ok_or(Error::UndefinedSymbol)?;
                    let byte_off = (target_word as i64 - at as i64) * 4;
                    if !(-(128 << 20)..=(128 << 20) - 4).contains(&byte_off) {
                        return Err(Error::UnsupportedReloc);
                    }
                    code[at] = encode::bl(byte_off as i32);
                }
                RelocKind::AdrpPg | RelocKind::AddPgOff => {
                    global_relocs.push(Reloc {
                        offset: at,
                        symbol: r.symbol.clone(),
                        kind: r.kind,
                    });
                }
                RelocKind::GotPg | RelocKind::GotLo12 => return Err(Error::UnsupportedReloc),
            }
        }
    }
    let symbols = module
        .functions
        .iter()
        .enumerate()
        .map(|(i, (name, _))| Symbol {
            name: name.clone(),
            offset: word_off[i] * 4,
        })
        .collect();
    let mut data_syms: Vec<DataSym> = Vec::with_capacity(module.data.len());
    let (mut rodata_off, mut data_off, mut bss_off) = (0usize, 0usize, 0usize);
    for d in module.data.iter() {
        let a = align_of_data(d.size);
        let size = d.size as usize;
        match d.kind {
            DataKind::Rodata => {
                rodata_off = align_forward(rodata_off, a);
                data_syms.push(DataSym {
                    name: d.name.clone(),
                    kind: d.kind,
                    off: rodata_off,
                    size,
                    bytes: d.bytes.clone(),
                    relocs: d.relocs.clone(),
                });
                rodata_off += size;
            }
            DataKind::Data => {
                data_off = align_forward(data_off, a);
                data_syms.push(DataSym {
                    name: d.name.clone(),
                    kind: d.kind,
                    off: data_off,
                    size,
                    bytes: d.bytes.clone(),
                    relocs: d.relocs.clone(),
                });
                data_off += size;
            }
            DataKind::Bss => {
                bss_off = align_forward(bss_off, a);
                data_syms.push(DataSym {
                    name: d.name.clone(),
                    kind: d.kind,
                    off: bss_off,
                    size,
                    bytes: Vec::new(),
                    relocs: Vec::new(),
                });
                bss_off += size;
            }
        }
    }
    Ok(Linked {
        code,
        symbols,
        data: data_syms,
        relocs: global_relocs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_alignment_covers_scalars() {
        assert_eq!(align_of_data(1), 1);
        assert_eq!(align_of_data(3), 4);
        assert_eq!(align_of_data(8), 8);
        assert_eq!(align_of_data(100), 8);
    }

    #[test]
    fn address_of_finds_entry() {
        let linked = Linked {
            code: vec![0; 4],
            symbols: vec![Symbol {
                name: "f".to_string(),
                offset: 8,
            }],
            data: Vec::new(),
            relocs: Vec::new(),
        };
        assert_eq!(linked.address_of("f"), Some(8));
        assert_eq!(linked.address_of("g"), None);
    }

    #[test]
    fn compile_module_resolves_call() {
        let mut m = Module::new();
        m.add_function("callee", vec![0xD65F_03C0]);
        m.add_function("caller", vec![encode::bl(0)]);
        let relocs = vec![Vec::new(), vec![Reloc::call(0, "callee")]];
        let linked = compile_module(&m, &relocs).unwrap();
        assert_eq!(linked.address_of("callee"), Some(0));
        assert_eq!(linked.address_of("caller"), Some(4));
        // caller word 1 branches back one word: byte offset -4.
        assert_eq!(linked.code[1], encode::bl(-4));
        assert!(linked.relocs.is_empty());
    }

    #[test]
    fn apply_global_reloc_round_trip() {
        let r = encode::Reg::from_u5(3);
        let mut code = vec![encode::adrp(r, 0), encode::add_imm64(r, r, 0)];
        let site = 0x1_0000_5000usize;
        let target = (site & !0xFFF) + 3 * 4096 + 0x123;
        apply_global_reloc(
            &mut code,
            &Reloc {
                offset: 0,
                symbol: "g".to_string(),
                kind: RelocKind::AdrpPg,
            },
            site,
            target,
        );
        apply_global_reloc(
            &mut code,
            &Reloc {
                offset: 1,
                symbol: "g".to_string(),
                kind: RelocKind::AddPgOff,
            },
            site,
            target,
        );
        let adrp = code[0];
        let immlo = (adrp >> 29) & 0x3;
        let immhi = (adrp >> 5) & 0x7FFFF;
        let u = (immhi << 2) | immlo;
        let delta = ((u << 11) as i32) >> 11; // sign-extend 21 bits
        let decoded_page = (site & !0xFFFusize).wrapping_add((delta as usize).wrapping_mul(4096));
        assert_eq!(decoded_page, target & !0xFFF);
        assert_eq!((code[1] >> 10) & 0xFFF, 0x123);
        assert_eq!(adrp & 0x1F, 3);
    }
}
