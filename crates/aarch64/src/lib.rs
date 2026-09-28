//! AArch64 execution-state definitions and instruction decoding.

use volt_jit_decoder::{DecodeError, Decoder};

/// A64 instructions supported by this frontend's initial decoder slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Nop,
    Movz {
        rd: u8,
        imm: u16,
        shift: u8,
        is_64_bit: bool,
    },
    Movk {
        rd: u8,
        imm: u16,
        shift: u8,
        is_64_bit: bool,
    },
    AddSubImm {
        rd: u8,
        rn: u8,
        imm: u32,
        subtract: bool,
        set_flags: bool,
        is_64_bit: bool,
    },
    Branch {
        displacement: i64,
    },
    BranchCond {
        displacement: i64,
        condition: u8,
    },
    Ret {
        rn: u8,
    },
}

/// Stateless A64 decoder.
#[derive(Clone, Copy, Debug, Default)]
pub struct Aarch64Decoder;

impl Decoder for Aarch64Decoder {
    type Word = u32;
    type Instruction = Instruction;

    fn decode(&self, word: u32) -> Result<Instruction, DecodeError<u32>> {
        decode(word).map_err(|_| DecodeError { word })
    }
}

/// Decode one little-endian A64 instruction word (already assembled to `u32`).
pub fn decode(word: u32) -> Result<Instruction, DecodeError<u32>> {
    let bad = || DecodeError { word };
    if word == 0xd503_201f {
        return Ok(Instruction::Nop);
    }
    if word & 0x7f80_0000 == 0x5280_0000 || word & 0x7f80_0000 == 0x7280_0000 {
        let is_64_bit = word >> 31 != 0;
        let opc = (word >> 29) & 3;
        if opc != 2 && opc != 3 {
            return Err(bad());
        }
        let hw = (word >> 21) & 3;
        if !is_64_bit && hw >= 2 {
            return Err(bad());
        }
        let shift = (hw * 16) as u8;
        let rd = (word & 31) as u8;
        let imm = ((word >> 5) & 0xffff) as u16;
        return if opc == 2 {
            Ok(Instruction::Movz {
                rd,
                imm,
                shift,
                is_64_bit,
            })
        } else {
            Ok(Instruction::Movk {
                rd,
                imm,
                shift,
                is_64_bit,
            })
        };
    }
    if word & 0x1f00_0000 == 0x1100_0000 {
        let is_64_bit = word >> 31 != 0;
        let subtract = (word >> 30) & 1 != 0;
        let set_flags = (word >> 29) & 1 != 0;
        let raw_imm = (word >> 10) & 0xfff;
        let imm = if word & (1 << 22) != 0 {
            raw_imm << 12
        } else {
            raw_imm
        };
        return Ok(Instruction::AddSubImm {
            rd: (word & 31) as u8,
            rn: ((word >> 5) & 31) as u8,
            imm,
            subtract,
            set_flags,
            is_64_bit,
        });
    }
    if word & 0xfc00_0000 == 0x1400_0000 {
        let raw = word & 0x03ff_ffff;
        return Ok(Instruction::Branch {
            displacement: (((raw << 6) as i32 >> 6) as i64) << 2,
        });
    }
    if word & 0xff00_0010 == 0x5400_0000 {
        let raw = (word >> 5) & 0x7ffff;
        return Ok(Instruction::BranchCond {
            displacement: (((raw << 13) as i32 >> 13) as i64) << 2,
            condition: (word & 15) as u8,
        });
    }
    if word & 0xffff_fc1f == 0xd65f_0000 {
        return Ok(Instruction::Ret {
            rn: ((word >> 5) & 31) as u8,
        });
    }
    Err(bad())
}

/// Per-vCPU AArch64 integer/control state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Cpu {
    /// General registers X0–X30.
    pub x: [u64; 31],
    /// Stack pointer (register 31 in SP-addressing forms).
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// NZCV in architectural bit positions 31:28.
    pub nzcv: u32,
}

impl Cpu {
    /// Read a register field as XZR/WZR when it is 31.
    #[inline]
    pub fn read_xzr(&self, register: u8) -> u64 {
        if register < 31 {
            self.x[register as usize]
        } else {
            0
        }
    }

    /// Write a register field; W writes zero-extend and register 31 discards.
    #[inline]
    pub fn write_xzr(&mut self, register: u8, value: u64, is_64_bit: bool) {
        if register < 31 {
            self.x[register as usize] = if is_64_bit {
                value
            } else {
                value as u32 as u64
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Instruction, decode};
    use volt_jit_decoder::Decoder;

    #[test]
    fn decodes_movz_and_branch() {
        assert_eq!(
            decode(0xd2c2_4681),
            Ok(Instruction::Movz {
                rd: 1,
                imm: 0x1234,
                shift: 32,
                is_64_bit: true,
            })
        );
        assert_eq!(
            decode(0x17ff_ffff),
            Ok(Instruction::Branch { displacement: -4 })
        );
    }

    #[test]
    fn generic_decoder_contract_preserves_rejected_word() {
        assert_eq!(
            super::Aarch64Decoder.decode(0xffff_ffff).unwrap_err().word,
            0xffff_ffff
        );
    }
}
