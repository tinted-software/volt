// SPDX-License-Identifier: Apache-2.0
// Copyright LilithSemi (Vulcan). Modified for the Volt Rust port.
//! Compile and link native functions and data globals into one relocatable image.
use super::{
    Error,
    isel::{self, RelocKind},
};
use alloc::{string::String, vec::Vec};
use volt_ir::function::Function;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataKind {
    Rodata,
    Data,
    Bss,
}
#[derive(Clone, Debug)]
pub struct DataReloc {
    pub off: usize,
    pub symbol: String,
}
#[derive(Clone, Debug)]
pub struct Data {
    pub name: String,
    pub bytes: Vec<u8>,
    pub kind: DataKind,
    pub size: u64,
    pub relocs: Vec<DataReloc>,
}
#[derive(Clone, Debug, Default)]
pub struct Module {
    pub functions: Vec<(String, Function)>,
    pub data: Vec<Data>,
}
impl Module {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_function(&mut self, name: &str, function: Function) {
        self.functions.push((name.into(), function));
    }
    pub fn add_data(&mut self, name: &str, bytes: Vec<u8>) {
        self.add_data_relocs(name, bytes, Vec::new());
    }
    pub fn add_data_relocs(&mut self, name: &str, bytes: Vec<u8>, relocs: Vec<DataReloc>) {
        let size = bytes.len() as u64;
        self.data.push(Data {
            name: name.into(),
            bytes,
            kind: DataKind::Rodata,
            size,
            relocs,
        });
    }
    pub fn add_writable(&mut self, name: &str, bytes: Vec<u8>) {
        self.add_writable_relocs(name, bytes, Vec::new());
    }
    pub fn add_writable_relocs(&mut self, name: &str, bytes: Vec<u8>, relocs: Vec<DataReloc>) {
        let size = bytes.len() as u64;
        self.data.push(Data {
            name: name.into(),
            bytes,
            kind: DataKind::Data,
            size,
            relocs,
        });
    }
    pub fn add_bss(&mut self, name: &str, size: u64) {
        self.data.push(Data {
            name: name.into(),
            bytes: Vec::new(),
            kind: DataKind::Bss,
            size,
            relocs: Vec::new(),
        });
    }
}
#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}
#[derive(Clone, Debug)]
pub struct Reloc {
    pub offset: usize,
    pub symbol: String,
    pub kind: RelocKind,
    pub addend: i64,
}
#[derive(Clone, Debug, Default)]
pub struct Linked {
    pub code: Vec<u8>,
    pub symbols: Vec<Symbol>,
    pub data: Vec<Data>,
    pub relocs: Vec<Reloc>,
}
pub fn compile_module(module: &Module) -> Result<Linked, Error> {
    let mut result = Linked::default();
    for (name, f) in &module.functions {
        while result.code.len() % 16 != 0 {
            result.code.push(0x90);
        }
        let offset = result.code.len();
        let compiled = isel::compile_object(f)?;
        result.symbols.push(Symbol {
            name: name.clone(),
            offset,
            size: compiled.code.len(),
        });
        for r in compiled.relocs {
            result.relocs.push(Reloc {
                offset: offset + r.offset,
                symbol: r.symbol,
                kind: r.kind,
                addend: r.addend,
            });
        }
        result.code.extend_from_slice(&compiled.code);
    }
    let mut pending = Vec::new();
    for r in result.relocs.drain(..) {
        if r.kind == RelocKind::Call || r.kind == RelocKind::PcRelative {
            if let Some(s) = result.symbols.iter().find(|s| s.name == r.symbol) {
                let disp = i32::try_from(s.offset as i64 + r.addend - r.offset as i64)
                    .map_err(|_| Error::RelocationOutOfRange)?;
                result
                    .code
                    .get_mut(r.offset..r.offset + 4)
                    .ok_or(Error::InvalidFunction)?
                    .copy_from_slice(&disp.to_le_bytes());
                continue;
            }
        }
        pending.push(r);
    }
    result.relocs = pending;
    result.data = module.data.clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_ir::function::{Call, Opcode, Ret, RetPiece, Terminator};
    #[test]
    fn resolves_inter_function_call_displacement() {
        let mut caller = Function::new();
        let ty = caller.types.parse_type("u64").unwrap();
        let block = caller.append_block();
        let symbol = caller.intern_symbol("callee");
        let args = caller.intern_values(&[]);
        let result = caller.append_inst(
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
        caller.set_terminator(block, Terminator::Ret(Ret::one(result)));
        let mut callee = Function::new();
        let ty = callee.types.parse_type("u64").unwrap();
        let block = callee.append_block();
        let result = callee.append_inst(block, ty, Opcode::Iconst(42));
        callee.set_terminator(block, Terminator::Ret(Ret::one(result)));
        let mut module = Module::new();
        module.add_function("caller", caller);
        module.add_function("callee", callee);
        let linked = compile_module(&module).unwrap();
        assert!(linked.relocs.is_empty());
        assert_eq!(linked.symbols[1].offset % 16, 0);
        let syms: Vec<_> = linked
            .symbols
            .iter()
            .map(|s| super::super::disasm::Sym {
                name: s.name.clone(),
                offset: s.offset,
            })
            .collect();
        assert!(super::super::disasm::format_module(&linked.code, &syms).contains("<callee>"));
    }
}
