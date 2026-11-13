//! SPIR-V shader parsing and inspection (host-side tooling).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use rspirv::binary::Disassemble;
use rspirv::dr::{Instruction, Module, Operand};
use spirv::ExecutionModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Other,
}

#[derive(Debug, Clone)]
pub struct EntryPoint {
    pub name: String,
    pub stage: ShaderStage,
    pub execution_model: ExecutionModel,
    pub id: u32,
    pub interface_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct SpirvModule {
    pub module: Module,
    pub entry_points: Vec<EntryPoint>,
}

#[derive(Debug)]
pub enum SpirvError {
    ParseError(rspirv::binary::ParseState),
    InvalidData,
}

impl From<rspirv::binary::ParseState> for SpirvError {
    fn from(err: rspirv::binary::ParseState) -> Self {
        SpirvError::ParseError(err)
    }
}

impl SpirvModule {
    /// Parse SPIR-V binary from bytes (must be multiple of 4 bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SpirvError> {
        if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
            return Err(SpirvError::InvalidData);
        }
        let module = rspirv::dr::load_bytes(bytes)?;
        let mut entry_points = Vec::new();

        for inst in &module.entry_points {
            if inst.class.opcode == spirv::Op::EntryPoint
                && let Some(ep) = parse_entry_point(inst)
            {
                entry_points.push(ep);
            }
        }

        Ok(Self {
            module,
            entry_points,
        })
    }

    /// Parse SPIR-V binary from 32-bit words.
    pub fn from_words(words: &[u32]) -> Result<Self, SpirvError> {
        let module = rspirv::dr::load_words(words)?;
        let mut entry_points = Vec::new();

        for inst in &module.entry_points {
            if inst.class.opcode == spirv::Op::EntryPoint
                && let Some(ep) = parse_entry_point(inst)
            {
                entry_points.push(ep);
            }
        }

        Ok(Self {
            module,
            entry_points,
        })
    }

    /// Disassemble module into textual assembly representation.
    pub fn disassemble(&self) -> String {
        self.module.disassemble()
    }
}

fn parse_entry_point(inst: &Instruction) -> Option<EntryPoint> {
    if inst.operands.len() < 3 {
        return None;
    }
    let model = match inst.operands[0] {
        Operand::ExecutionModel(model) => model,
        _ => return None,
    };
    let id = match inst.operands[1] {
        Operand::IdRef(id) => id,
        _ => return None,
    };
    let name = match &inst.operands[2] {
        Operand::LiteralString(s) => s.clone(),
        _ => return None,
    };

    let stage = match model {
        ExecutionModel::Vertex => ShaderStage::Vertex,
        ExecutionModel::Fragment => ShaderStage::Fragment,
        ExecutionModel::GLCompute => ShaderStage::Compute,
        _ => ShaderStage::Other,
    };

    let mut interface_ids = Vec::new();
    for op in &inst.operands[3..] {
        if let Operand::IdRef(id) = op {
            interface_ids.push(*id);
        }
    }

    Some(EntryPoint {
        name,
        stage,
        execution_model: model,
        id,
        interface_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_error() {
        assert!(SpirvModule::from_bytes(&[]).is_err());
        assert!(SpirvModule::from_bytes(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_parse_real_shader() {
        let bytes = include_bytes!("../testdata/testdata_flip_frag.spv");
        let m = SpirvModule::from_bytes(bytes).expect("valid real fragment shader");
        assert_eq!(m.entry_points.len(), 1);
        assert_eq!(m.entry_points[0].name, "main");
        assert_eq!(m.entry_points[0].stage, ShaderStage::Fragment);
        let dis = m.disassemble();
        assert!(dis.contains("OpEntryPoint Fragment"));
    }

    #[test]
    fn test_parse_simple_spirv() {
        let buffer: &[u8] = &[
            0x03, 0x02, 0x23, 0x07, // Magic
            0x00, 0x00, 0x01, 0x00, // Version 1.0
            0x00, 0x00, 0x00, 0x00, // Generator 0
            0x01, 0x00, 0x00, 0x00, // Bound 1
            0x00, 0x00, 0x00, 0x00, // Reserved 0
            0x0e, 0x00, 0x03, 0x00, // OpMemoryModel length 3
            0x00, 0x00, 0x00, 0x00, // Logical
            0x01, 0x00, 0x00, 0x00, // GLSL450
        ];
        let m = SpirvModule::from_bytes(buffer).expect("valid minimal spirv");
        assert_eq!(m.entry_points.len(), 0);
        let dis = m.disassemble();
        assert!(dis.contains("OpMemoryModel Logical GLSL450"));
    }
}
