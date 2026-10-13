//! AArch64 static/JIT linker: compile a [`volt_ir::module::Module`],
//! concatenate its functions, resolve intra-module calls, lay out data
//! globals, and carry `global_addr` relocations forward.
//!
//! Each function goes through [`super::isel::compile_function`]; the
//! relocations it retains drive linking here and, for object emission or native
//! module mapping, the unresolved ones are carried in [`Linked`].

use alloc::string::String;
use alloc::vec::Vec;

use volt_ir::module::{DataKind, DataReloc, Module};

use super::encode;
use super::isel::{self, ModelCaps, Reloc, RelocKind};

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
    /// The global's requested alignment (`0` = natural), as in [`volt_ir::module::Global::align`].
    pub align: u32,
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

/// Compile every function of `module` with `caps`, concatenate the code,
/// resolve each intra-module `call` relocation to a PC-relative `bl`, lay out
/// data globals into per-section `DataSym`s, and carry every `global_addr`
/// relocation forward unresolved.
///
/// The first function sits at offset 0 of the image. Symbols the module does
/// not define are an error for calls ([`Error::UndefinedSymbol`]) but are left
/// unresolved for `global_addr` relocations and data relocations.
pub fn compile_module(module: &Module, caps: &ModelCaps) -> Result<Linked, Error> {
    let compiled = module
        .functions
        .iter()
        .map(|def| isel::compile_function(&def.function, caps).map_err(|_| Error::Unsupported))
        .collect::<Result<Vec<_>, _>>()?;
    let n = compiled.len();
    let mut word_off = Vec::with_capacity(n);
    let mut total = 0usize;
    for part in &compiled {
        word_off.push(total);
        total += part.code.len();
    }
    let mut code: Vec<u32> = Vec::with_capacity(total);
    for part in &compiled {
        code.extend_from_slice(&part.code);
    }
    let mut global_relocs: Vec<Reloc> = Vec::new();
    for (i, part) in compiled.iter().enumerate() {
        for r in &part.relocs {
            let at = word_off[i] + r.offset;
            match r.kind {
                RelocKind::Call => {
                    let target_word = module
                        .functions
                        .iter()
                        .position(|def| def.name == r.symbol)
                        .map(|j| word_off[j])
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
        .map(|(i, def)| Symbol {
            name: def.name.clone(),
            offset: word_off[i] * 4,
        })
        .collect();
    let mut data_syms: Vec<DataSym> = Vec::with_capacity(module.globals.len());
    let (mut rodata_off, mut data_off, mut bss_off) = (0usize, 0usize, 0usize);
    for g in module.globals.iter() {
        let a = g.effective_align();
        let size = g.size as usize;
        let (cursor, bytes, relocs) = match g.kind {
            DataKind::Rodata => (&mut rodata_off, g.bytes.clone(), g.relocs.clone()),
            DataKind::Data => (&mut data_off, g.bytes.clone(), g.relocs.clone()),
            DataKind::Bss => (&mut bss_off, Vec::new(), Vec::new()),
        };
        *cursor = align_forward(*cursor, a);
        data_syms.push(DataSym {
            name: g.name.clone(),
            kind: g.kind,
            off: *cursor,
            size,
            align: g.align,
            bytes,
            relocs,
        });
        *cursor += size;
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
    use alloc::string::ToString as _;
    use alloc::vec;
    use volt_ir::function::{Call, Function, GlobalAddr, Opcode, Ret, RetPiece, Terminator};

    use super::*;

    fn function_returning(value: i64) -> Function {
        let mut f = Function::new();
        let ty = f.types.parse_type("i64").unwrap();
        let block = f.append_block();
        let v = f.append_inst(block, ty, Opcode::Iconst(value));
        f.set_terminator(block, Terminator::Ret(Ret::one(v)));
        f
    }

    fn function_calling(callee: &str) -> Function {
        let mut f = Function::new();
        let ty = f.types.parse_type("i64").unwrap();
        let block = f.append_block();
        let symbol = f.intern_symbol(callee);
        let args = f.intern_values(&[]);
        let result = f.append_inst(
            block,
            ty,
            Opcode::Call(Call {
                symbol,
                args,
                is_variadic: false,
                num_fixed: 0,
                ret_dest: None,
                ret_regs: 0,
                ret_pieces: [RetPiece::default(); 4],
                sret: false,
            }),
        );
        f.set_terminator(block, Terminator::Ret(Ret::one(result)));
        f
    }

    fn function_taking_address_of(symbol: &str) -> Function {
        let mut f = Function::new();
        let ty = f.types.parse_type("ptr").unwrap();
        let block = f.append_block();
        let symbol = f.intern_symbol(symbol);
        let v = f.append_inst(
            block,
            ty,
            Opcode::GlobalAddr(GlobalAddr {
                symbol,
                via_got: false,
            }),
        );
        f.set_terminator(block, Terminator::Ret(Ret::one(v)));
        f
    }

    #[test]
    fn globals_are_laid_out_per_section_with_alignment() {
        let mut m = Module::new();
        m.add_data("ro1", vec![1, 2, 3]);
        m.add_data("ro2", vec![0; 8]);
        m.add_writable("w1", vec![9]);
        m.add_writable("w2", vec![0; 4]);
        m.add_bss("b1", 3);
        m.add_bss("b2", 64);
        m.globals[5].align = 64;
        let linked = compile_module(&m, &ModelCaps::default()).unwrap();
        let place = |name: &str| {
            let d = linked.data.iter().find(|d| d.name == name).unwrap();
            (d.kind, d.off, d.size)
        };
        assert_eq!(place("ro1"), (DataKind::Rodata, 0, 3));
        // `ro2` is 8 bytes: aligned to 8 after the 3-byte `ro1`.
        assert_eq!(place("ro2"), (DataKind::Rodata, 8, 8));
        assert_eq!(place("w1"), (DataKind::Data, 0, 1));
        assert_eq!(place("w2"), (DataKind::Data, 4, 4));
        assert_eq!(place("b1"), (DataKind::Bss, 0, 3));
        // An explicit alignment overrides the natural (<= 8) one.
        assert_eq!(place("b2"), (DataKind::Bss, 64, 64));
        assert!(
            linked
                .data
                .iter()
                .find(|d| d.name == "b2")
                .unwrap()
                .bytes
                .is_empty()
        );
    }

    #[test]
    fn global_addr_relocations_carry_forward_unresolved() {
        let mut m = Module::new();
        m.add_function("first", function_returning(0));
        m.add_function("addr", function_taking_address_of("table"));
        m.add_writable_relocs(
            "table",
            vec![0; 8],
            vec![DataReloc {
                offset: 0,
                symbol: "first".to_string(),
            }],
        );
        let linked = compile_module(&m, &ModelCaps::default()).unwrap();
        let start_word = linked.address_of("addr").unwrap() / 4;
        let kinds: Vec<_> = linked
            .relocs
            .iter()
            .map(|r| (r.kind, r.symbol.as_str()))
            .collect();
        assert!(kinds.contains(&(RelocKind::AdrpPg, "table")));
        assert!(kinds.contains(&(RelocKind::AddPgOff, "table")));
        // Carried-forward offsets are word indices into the whole image.
        assert!(linked.relocs.iter().all(|r| r.offset >= start_word));
        assert_eq!(linked.data[0].relocs[0].symbol, "first");
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
        m.add_function("callee", function_returning(42));
        m.add_function("caller", function_calling("callee"));
        let linked = compile_module(&m, &ModelCaps::default()).unwrap();
        let caller = linked.address_of("caller").unwrap();
        assert_eq!(linked.address_of("callee"), Some(0));
        assert!(caller > 0 && caller % 4 == 0);
        assert!(linked.relocs.is_empty());
        // The caller contains exactly one `bl`, which branches back to word 0.
        let bls: Vec<usize> = (caller / 4..linked.code.len())
            .filter(|&i| linked.code[i] >> 26 == 0b100101)
            .collect();
        assert_eq!(bls.len(), 1);
        let at = bls[0];
        assert_eq!(linked.code[at], encode::bl(-((at * 4) as i32)));
    }

    #[test]
    fn undefined_call_target_is_an_error() {
        let mut m = Module::new();
        m.add_function("caller", function_calling("nowhere"));
        assert_eq!(
            compile_module(&m, &ModelCaps::default()).unwrap_err(),
            Error::UndefinedSymbol
        );
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
