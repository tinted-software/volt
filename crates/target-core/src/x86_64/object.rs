// SPDX-License-Identifier: Apache-2.0
// Copyright LilithSemi (Vulcan). Modified for the Volt Rust port.
//! ELF64 ET_REL emission, with one section per function or data definition.
use super::{
    Error,
    isel::{self, RelocKind},
    link::{DataKind, Module},
};
use alloc::{format, vec::Vec};
use object::write::{Object, Relocation, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags, SymbolKind,
    SymbolScope,
};
pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_PC32: u32 = 2;
pub const R_X86_64_PLT32: u32 = 4;
pub const R_X86_64_GOTPCREL: u32 = 9;
pub fn write_module(module: &Module) -> Result<Vec<u8>, Error> {
    let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
    let mut symbols = Vec::new();
    let mut pending = Vec::new();
    for (name, f) in &module.functions {
        let compiled = isel::compile_object(f)?;
        let section = obj.add_section(
            Vec::new(),
            format!(".text.{name}").into_bytes(),
            SectionKind::Text,
        );
        obj.append_section_data(section, &compiled.code, 16);
        let id = obj.add_symbol(Symbol {
            name: name.as_bytes().to_vec(),
            value: 0,
            size: compiled.code.len() as u64,
            kind: SymbolKind::Text,
            scope: if f.is_local || name.starts_with('.') {
                SymbolScope::Compilation
            } else {
                SymbolScope::Linkage
            },
            weak: false,
            section: SymbolSection::Section(section),
            flags: SymbolFlags::None,
        });
        symbols.push((name.clone(), id));
        for r in compiled.relocs {
            pending.push((
                section,
                r.offset as u64,
                r.symbol,
                match r.kind {
                    RelocKind::Call => R_X86_64_PLT32,
                    RelocKind::Absolute => R_X86_64_64,
                    RelocKind::PcRelative => R_X86_64_PC32,
                    RelocKind::GotPcRelative => R_X86_64_GOTPCREL,
                },
                r.addend,
            ));
        }
    }
    for data in &module.data {
        let (class, kind) = match data.kind {
            DataKind::Rodata => ("rodata", SectionKind::ReadOnlyData),
            DataKind::Data => ("data", SectionKind::Data),
            DataKind::Bss => ("bss", SectionKind::UninitializedData),
        };
        let sec = obj.add_section(
            Vec::new(),
            format!(".{class}.{}", data.name.trim_start_matches('.')).into_bytes(),
            kind,
        );
        if data.kind == DataKind::Bss {
            obj.append_section_bss(sec, data.size, 8);
        } else {
            obj.append_section_data(sec, &data.bytes, 8);
        }
        let id = obj.add_symbol(Symbol {
            name: data.name.as_bytes().to_vec(),
            value: 0,
            size: data.size,
            kind: SymbolKind::Data,
            scope: if data.name.starts_with('.') {
                SymbolScope::Compilation
            } else {
                SymbolScope::Linkage
            },
            weak: false,
            section: SymbolSection::Section(sec),
            flags: SymbolFlags::None,
        });
        symbols.push((data.name.clone(), id));
        for r in &data.relocs {
            if r.off
                .checked_add(8)
                .is_none_or(|end| end > data.size as usize)
            {
                return Err(Error::InvalidFunction);
            }
            pending.push((sec, r.off as u64, r.symbol.clone(), R_X86_64_64, 0));
        }
    }
    for (section, offset, name, r_type, addend) in pending {
        let id = if let Some((_, id)) = symbols.iter().find(|(n, _)| *n == name) {
            *id
        } else {
            let id = obj.add_symbol(Symbol {
                name: name.as_bytes().to_vec(),
                value: 0,
                size: 0,
                kind: SymbolKind::Unknown,
                scope: SymbolScope::Unknown,
                weak: false,
                section: SymbolSection::Undefined,
                flags: SymbolFlags::None,
            });
            symbols.push((name, id));
            id
        };
        obj.add_relocation(
            section,
            Relocation {
                offset,
                symbol: id,
                addend,
                flags: RelocationFlags::Elf {
                    r_type: object::elf::RelocationType(r_type),
                },
            },
        )
        .map_err(|e| Error::Object(format!("{e}")))?;
    }
    obj.write().map_err(|e| Error::Object(format!("{e}")))
}
#[cfg(test)]
mod tests {
    use super::*;
    use object::{Object as _, ObjectSection as _, ObjectSymbol as _};
    use volt_ir::function::{Function, Opcode, Ret, Terminator};
    #[test]
    fn elf_function_section_and_symbol() {
        let mut f = Function::new();
        let t = f.types.parse_type("u64").unwrap();
        let b = f.append_block();
        let v = f.append_inst(b, t, Opcode::Iconst(42));
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        let mut module = Module::new();
        module.add_function("answer", f);
        let bytes = write_module(&module).unwrap();
        let elf = object::File::parse(bytes.as_slice()).unwrap();
        assert_eq!(elf.architecture(), Architecture::X86_64);
        assert!(
            elf.section_by_name(".text.answer")
                .unwrap()
                .data()
                .unwrap()
                .ends_with(&[0xc3])
        );
        assert!(elf.symbols().any(|s| s.name().unwrap() == "answer"));
    }
    #[test]
    fn elf_preserves_external_call_and_got_relocations() {
        use volt_ir::function::{Call, GlobalAddr, RetPiece};
        let mut f = Function::new();
        let t = f.types.parse_type("ptr").unwrap();
        let b = f.append_block();
        let data = f.intern_symbol("external_data");
        let external = f.intern_symbol("external_function");
        let ptr = f.append_inst(
            b,
            t,
            Opcode::GlobalAddr(GlobalAddr {
                symbol: data,
                via_got: true,
            }),
        );
        let args = f.intern_values(&[ptr]);
        let result = f.append_inst(
            b,
            t,
            Opcode::Call(Call {
                symbol: external,
                args,
                is_variadic: false,
                num_fixed: 0,
                ret_dest: None,
                ret_regs: 0,
                ret_pieces: [RetPiece::default(); 4],
                sret: false,
            }),
        );
        f.set_terminator(b, Terminator::Ret(Ret::one(result)));
        let mut module = Module::new();
        module.add_function("entry", f);
        let bytes = write_module(&module).unwrap();
        let elf = object::File::parse(bytes.as_slice()).unwrap();
        let section = elf.section_by_name(".text.entry").unwrap();
        let relocations: Vec<_> = section.relocations().map(|(_, r)| r).collect();
        assert_eq!(relocations.len(), 2);
        assert!(relocations.iter().any(|r| r.flags()
            == RelocationFlags::Elf {
                r_type: object::elf::RelocationType(R_X86_64_PLT32)
            }));
        assert!(relocations.iter().any(|r| r.flags()
            == RelocationFlags::Elf {
                r_type: object::elf::RelocationType(R_X86_64_GOTPCREL)
            }));
        assert!(relocations.iter().all(|r| r.addend() == -4));
        assert!(
            elf.symbols()
                .any(|s| s.name().unwrap() == "external_function" && s.is_undefined())
        );
    }
}
