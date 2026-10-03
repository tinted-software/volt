//! AArch64 relocatable object vocabulary: symbols, sections, relocations.
//!
//! These types describe the ELF `R_AARCH64_*` relocations and symbol table
//! entries this backend emits. Actual ELF serialization lives elsewhere; this
//! module only owns the shared data vocabulary.
#![allow(
    clippy::std_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::alloc_instead_of_core
)]
use alloc::string::String;
use alloc::vec::Vec;

/// A symbol's binding. Locals must precede globals in the symbol table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Binding {
    Local,
    Global,
}

/// A symbol's type. `Func` is an entry point. `Object` is a data object.
/// `NoType` is unknown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymKind {
    NoType,
    Func,
    Object,
}

/// The allocatable output sections for a symbol. `Text` holds code, `Rodata`
/// holds read-only data, `Data` holds writable data, `Bss` holds
/// zero-initialized data (`Bss` uses memory but no file bytes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectionKind {
    Text,
    Rodata,
    Data,
    Bss,
}

/// One symbol table entry. A defined symbol lives in `section` at `value`. An
/// undefined symbol (`defined = false`) is external for the linker to resolve.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub value: u64,
    pub size: u64,
    pub binding: Binding,
    pub kind: SymKind,
    pub defined: bool,
    pub section: SectionKind,
}

impl Symbol {
    pub fn defined_func(name: &str, value: u64, size: u64) -> Symbol {
        Symbol {
            name: name.to_string(),
            value,
            size,
            binding: Binding::Global,
            kind: SymKind::Func,
            defined: true,
            section: SectionKind::Text,
        }
    }

    pub fn undefined(name: &str) -> Symbol {
        Symbol {
            name: name.to_string(),
            value: 0,
            size: 0,
            binding: Binding::Global,
            kind: SymKind::NoType,
            defined: false,
            section: SectionKind::Text,
        }
    }
}

/// The AArch64 relocation types this backend emits (architectural codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RelocType {
    /// `R_AARCH64_ADR_PREL_PG_HI21`: patches an `adrp` page-relative immediate
    /// (high half of a `global_addr` adrp/add pair).
    AdrPrelPgHi21 = 275,
    /// `R_AARCH64_ADD_ABS_LO12_NC`: patches an `add` 12-bit page-offset
    /// immediate (low half of a `global_addr` adrp/add pair).
    AddAbsLo12Nc = 277,
    /// `R_AARCH64_CALL26`: patches a `bl`/`b` 26-bit immediate (±128 MiB).
    Call26 = 283,
    /// `R_AARCH64_ADR_GOT_PAGE`: patches an `adrp` to a GOT entry's page.
    AdrGotPage = 311,
    /// `R_AARCH64_LD64_GOT_LO12_NC`: patches an `ldr` imm12 to a GOT entry's
    /// lo12 >> 3.
    Ld64GotLo12Nc = 312,
}

/// `R_AARCH64_ABS64`: a 64-bit absolute address (`S + A`) written into a data
/// section slot holding a pointer initializer.
pub const R_AARCH64_ABS64: u32 = 257;

/// A relocation applied to a `.text` byte offset against a symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Reloc {
    pub offset: u64,
    pub symbol: u32,
    pub reloc_type: RelocType,
    pub addend: i64,
}

impl Reloc {
    pub fn new(offset: u64, symbol: u32, reloc_type: RelocType) -> Reloc {
        Reloc {
            offset,
            symbol,
            reloc_type,
            addend: 0,
        }
    }
}

/// A relocation applied to a data section slot: at byte `offset` within
/// `section`, an `R_AARCH64_ABS64` relocation writes `symbol`'s runtime
/// address plus `addend`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataRelocEntry {
    pub section: SectionKind,
    pub offset: u64,
    pub symbol: u32,
    pub addend: i64,
}

/// Lower the vocabulary above to an AArch64 ELF relocatable object via the
/// `object` crate.
///
/// `code` holds little-endian machine words for `.text`; `rodata`/`data` hold
/// the initialized bytes for their sections; `bss_size` is the
/// zero-initialized tail of `.bss`. `relocs` apply to `.text` at byte
/// `offset`s; `data_relocs` apply to their named section at byte `offset`s.
/// Symbol cross-references use indices into `symbols`.
pub fn write_object(
    symbols: &[Symbol],
    code: &[u32],
    rodata: &[u8],
    data: &[u8],
    bss_size: u64,
    relocs: &[Reloc],
    data_relocs: &[DataRelocEntry],
) -> Vec<u8> {
    use object::write::{Object, StandardSection, SymbolSection};
    use object::{Architecture, BinaryFormat, Endianness};

    let mut obj = Object::new(BinaryFormat::Elf, Architecture::Aarch64, Endianness::Little);

    let text = obj.section_id(StandardSection::Text);
    let rodata_sec = obj.section_id(StandardSection::ReadOnlyDataWithRel);
    let data_sec = obj.section_id(StandardSection::Data);
    let bss_sec = obj.section_id(StandardSection::UninitializedData);
    let mut bytes = Vec::with_capacity(code.len() * 4);
    for w in code {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    if !bytes.is_empty() {
        obj.append_section_data(text, &bytes, 4);
    }
    if !rodata.is_empty() {
        obj.append_section_data(rodata_sec, rodata, 1);
    }
    if !data.is_empty() {
        obj.append_section_data(data_sec, data, 1);
    }
    if bss_size != 0 {
        obj.append_section_bss(bss_sec, bss_size, 8);
    }

    let section_of = |kind: SectionKind| match kind {
        SectionKind::Text => text,
        SectionKind::Rodata => rodata_sec,
        SectionKind::Data => data_sec,
        SectionKind::Bss => bss_sec,
    };

    let mut ids = Vec::with_capacity(symbols.len());
    for s in symbols {
        let kind = match s.kind {
            SymKind::Func => object::SymbolKind::Text,
            SymKind::Object => object::SymbolKind::Data,
            SymKind::NoType => object::SymbolKind::Unknown,
        };
        let scope = match s.binding {
            Binding::Local => object::SymbolScope::Compilation,
            Binding::Global => object::SymbolScope::Linkage,
        };
        let section = if s.defined {
            SymbolSection::Section(section_of(s.section))
        } else {
            SymbolSection::Undefined
        };
        ids.push(obj.add_symbol(object::write::Symbol {
            name: s.name.as_bytes().to_vec(),
            value: s.value,
            size: s.size,
            kind,
            scope,
            weak: false,
            section,
            flags: object::SymbolFlags::None,
        }));
    }

    for r in relocs {
        let target = ids[r.symbol as usize];
        let flags = reloc_flags(r.reloc_type);
        obj.add_relocation(
            text,
            object::write::Relocation {
                offset: r.offset,
                symbol: target,
                addend: r.addend,
                flags,
            },
        )
        .expect("text relocation section exists");
    }
    for r in data_relocs {
        let target = ids[r.symbol as usize];
        obj.add_relocation(
            section_of(r.section),
            object::write::Relocation {
                offset: r.offset,
                symbol: target,
                addend: r.addend,
                flags: object::RelocationFlags::Generic {
                    kind: object::RelocationKind::Absolute,
                    encoding: object::RelocationEncoding::Generic,
                    size: 64,
                },
            },
        )
        .expect("data relocation section exists");
    }

    let mut out = Vec::new();
    obj.emit(&mut out).expect("ELF emission succeeds");
    out
}

/// Map an architectural relocation code to `object` flags.
///
/// `CALL26` has a first-class generic mapping; the `adrp`/`add`/GOT pair
/// halves (275/277/311/312) have no generic encoding, so they pass through
/// as raw ELF `r_type` values.
fn reloc_flags(reloc_type: RelocType) -> object::RelocationFlags {
    match reloc_type {
        RelocType::Call26 => object::RelocationFlags::Generic {
            kind: object::RelocationKind::PltRelative,
            encoding: object::RelocationEncoding::AArch64Call,
            size: 26,
        },
        other => object::RelocationFlags::Elf {
            r_type: object::elf::RelocationType(other as u32),
        },
    }
}

/// Lower a [`super::link::Linked`] image through [`write_object`].
///
/// Function symbols become defined `Func` symbols in `.text` at their linked
/// byte offsets; data globals become defined `Object` symbols in their own
/// section; `global_addr` halves (word indices in `linked.relocs`) become
/// 275/277 text relocations at byte offsets (`* 4`); data-internal pointer
/// slots become `ABS64` data relocations. Names referenced but never defined
/// are emitted as undefined extern symbols.
pub fn write_module_object(linked: &super::link::Linked) -> Vec<u8> {
    use super::isel::RelocKind;
    use super::link::DataKind;
    use alloc::collections::BTreeMap;

    let mut symbols: Vec<Symbol> = Vec::new();
    let mut id_of_name: BTreeMap<&str, u32> = BTreeMap::new();
    for s in linked.symbols.iter() {
        let id = symbols.len() as u32;
        symbols.push(Symbol {
            name: s.name.clone(),
            value: s.offset as u64,
            size: 0,
            binding: Binding::Global,
            kind: SymKind::Func,
            defined: true,
            section: SectionKind::Text,
        });
        id_of_name.insert(s.name.as_str(), id);
    }
    let mut rodata_bytes: Vec<u8> = Vec::new();
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut bss_size: u64 = 0;
    for d in linked.data.iter() {
        let section = match d.kind {
            DataKind::Rodata => {
                rodata_bytes.extend_from_slice(&d.bytes);
                SectionKind::Rodata
            }
            DataKind::Data => {
                data_bytes.extend_from_slice(&d.bytes);
                SectionKind::Data
            }
            DataKind::Bss => {
                bss_size += d.size as u64;
                SectionKind::Bss
            }
        };
        let id = symbols.len() as u32;
        symbols.push(Symbol {
            name: d.name.clone(),
            value: d.off as u64,
            size: d.size as u64,
            binding: Binding::Global,
            kind: SymKind::Object,
            defined: true,
            section,
        });
        id_of_name.insert(d.name.as_str(), id);
    }
    // Externs referenced by relocations but never defined.
    let mut missing: Vec<&str> = Vec::new();
    for r in linked.relocs.iter() {
        if !id_of_name.contains_key(r.symbol.as_str()) && !missing.contains(&r.symbol.as_str()) {
            missing.push(r.symbol.as_str());
        }
    }
    for d in linked.data.iter() {
        for r in d.relocs.iter() {
            if !id_of_name.contains_key(r.symbol.as_str()) && !missing.contains(&r.symbol.as_str())
            {
                missing.push(r.symbol.as_str());
            }
        }
    }
    for name in missing {
        let id = symbols.len() as u32;
        symbols.push(Symbol::undefined(name));
        // `name` borrows from `linked`; outlives this function.
        id_of_name.insert(name, id);
    }

    let mut relocs: Vec<Reloc> = Vec::with_capacity(linked.relocs.len());
    for r in linked.relocs.iter() {
        let reloc_type = match r.kind {
            RelocKind::AdrpPg => RelocType::AdrPrelPgHi21,
            RelocKind::AddPgOff => RelocType::AddAbsLo12Nc,
            RelocKind::Call => RelocType::Call26,
            RelocKind::GotPg => RelocType::AdrGotPage,
            RelocKind::GotLo12 => RelocType::Ld64GotLo12Nc,
        };
        relocs.push(Reloc {
            offset: (r.offset as u64) * 4,
            symbol: id_of_name[r.symbol.as_str()],
            reloc_type,
            addend: 0,
        });
    }
    let mut data_relocs: Vec<DataRelocEntry> = Vec::new();
    for d in linked.data.iter() {
        let section = match d.kind {
            DataKind::Rodata => SectionKind::Rodata,
            DataKind::Data => SectionKind::Data,
            DataKind::Bss => SectionKind::Bss,
        };
        for r in d.relocs.iter() {
            data_relocs.push(DataRelocEntry {
                section,
                offset: (d.off + r.off) as u64,
                symbol: id_of_name[r.symbol.as_str()],
                addend: 0,
            });
        }
    }

    write_object(
        &symbols,
        &linked.code,
        &rodata_bytes,
        &data_bytes,
        bss_size,
        &relocs,
        &data_relocs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aarch64::link::{DataReloc, Linked};
    use object::read::{Object as _, ObjectSymbol as _};

    #[test]
    fn elf_round_trip_aarch64() {
        let code = vec![0xD65F_03C0u32, 0x1400_0000u32];
        let symbols = vec![
            Symbol::defined_func("callee", 0, 4),
            Symbol {
                name: "local_data".to_string(),
                value: 0,
                size: 8,
                binding: Binding::Local,
                kind: SymKind::Object,
                defined: true,
                section: SectionKind::Data,
            },
            Symbol::undefined("extern_fn"),
        ];
        let relocs = vec![
            Reloc {
                offset: 4,
                symbol: 2,
                reloc_type: RelocType::Call26,
                addend: 0,
            },
            Reloc {
                offset: 0,
                symbol: 1,
                reloc_type: RelocType::AdrPrelPgHi21,
                addend: 0,
            },
        ];
        let data_relocs = vec![DataRelocEntry {
            section: SectionKind::Data,
            offset: 0,
            symbol: 1,
            addend: 0,
        }];
        let bytes = write_object(&symbols, &code, &[], &[0u8; 8], 16, &relocs, &data_relocs);
        let obj = object::read::File::parse(&bytes[..]).expect("emitted ELF parses");
        assert_eq!(obj.architecture(), object::Architecture::Aarch64);
        assert_eq!(
            u16::from_le_bytes([bytes[18], bytes[19]]),
            183,
            "e_machine == EM_AARCH64"
        );
        assert_eq!(
            obj.endianness(),
            object::Endianness::Little,
            "emitted object is little-endian"
        );
        assert!(obj.sections().count() >= 4, "text+data+bss+symtab sections");
        // Section symbols + file symbol + 3 named symbols.
        assert!(obj.symbols().count() >= 3, "named symbols survive");
        let names: Vec<String> = obj
            .symbols()
            .filter_map(|s| s.name().ok().map(|n| n.to_string()))
            .collect();
        for want in ["callee", "local_data", "extern_fn"] {
            assert!(names.iter().any(|n| n == want), "symbol {want} present");
        }

        // Linked-image lowering: word-idx reloc offsets become byte offsets.
        let linked = Linked {
            code: vec![0xD65F_03C0, 0x9000_0000],
            symbols: vec![crate::aarch64::link::Symbol {
                name: "f".to_string(),
                offset: 0,
            }],
            data: Vec::new(),
            relocs: vec![crate::aarch64::isel::Reloc {
                offset: 1,
                symbol: "ext".to_string(),
                kind: crate::aarch64::isel::RelocKind::AdrpPg,
            }],
        };
        let bytes = write_module_object(&linked);
        let obj = object::read::File::parse(&bytes[..]).expect("linked ELF parses");
        assert_eq!(obj.architecture(), object::Architecture::Aarch64);
        assert!(obj.sections().count() >= 2);
        assert!(obj.symbols().count() >= 2, "defined + extern symbols");
        let _ = DataReloc {
            off: 0,
            symbol: String::new(),
        };
    }
}
