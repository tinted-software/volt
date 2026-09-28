//! AArch64 instruction-block lowering to native code through Pliron/LLVM.

use std::fmt;

use volt_jit_aarch64::{Cpu, Instruction, decode};
use volt_jit_llvm::{BackendError, JitEngine};

/// Translation or execution error.
#[derive(Debug)]
pub enum Error {
    /// Instruction byte stream is empty or not word-aligned.
    InvalidBlock,
    /// An instruction encoding is not supported by this frontend.
    Decode(u32),
    /// Instruction is decoded but not yet lowerable to the block ABI.
    Unsupported(u32),
    /// Pliron or LLVM rejected the generated module.
    Backend(BackendError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlock => write!(f, "empty or unaligned AArch64 block"),
            Self::Decode(word) => write!(f, "unsupported AArch64 instruction {word:#010x}"),
            Self::Unsupported(word) => write!(f, "AArch64 instruction not lowerable: {word:#010x}"),
            Self::Backend(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {}

/// Native code for a translated AArch64 block.
pub struct Block {
    _jit: JitEngine,
    entry: unsafe extern "C" fn(*mut u64) -> u64,
}

impl Block {
    /// Compile little-endian A64 instruction bytes at `guest_pc`.
    pub fn compile(guest_pc: u64, bytes: &[u8]) -> Result<Self, Error> {
        if bytes.is_empty() || bytes.len() % 4 != 0 {
            return Err(Error::InvalidBlock);
        }
        let mut body = String::new();
        let mut terminated = false;
        let mut next = guest_pc;
        let mut temp = 0usize;

        for encoded in bytes.chunks_exact(4) {
            if terminated {
                break;
            }
            let word = u32::from_le_bytes(encoded.try_into().expect("four-byte chunk"));
            let insn = decode(word).map_err(|_| Error::Decode(word))?;
            match insn {
                Instruction::Nop => next = next.wrapping_add(4),
                Instruction::Movz {
                    rd,
                    imm,
                    shift,
                    is_64_bit,
                } => {
                    if rd < 31 {
                        let mut value = (imm as u64) << shift;
                        if !is_64_bit {
                            value &= u32::MAX as u64;
                        }
                        body.push_str(&format!(
                            "  %z{temp} = getelementptr i64, ptr %regs, i64 {rd}\n  store i64 {value}, ptr %z{temp}, align 8\n"
                        ));
                        temp += 1;
                    }
                    next = next.wrapping_add(4);
                }
                Instruction::Movk {
                    rd,
                    imm,
                    shift,
                    is_64_bit,
                } => {
                    if rd < 31 {
                        body.push_str(&format!(
                            "  %p{temp} = getelementptr i64, ptr %regs, i64 {rd}\n  %v{temp} = load i64, ptr %p{temp}, align 8\n"
                        ));
                        let width_mask = if is_64_bit { u64::MAX } else { u32::MAX as u64 };
                        let keep = width_mask & !(0xffffu64 << shift);
                        let insert = ((imm as u64) << shift) & width_mask;
                        body.push_str(&format!(
                            "  %k{temp} = and i64 %v{temp}, {keep}\n  %m{temp} = or i64 %k{temp}, {insert}\n  store i64 %m{temp}, ptr %p{temp}, align 8\n"
                        ));
                        temp += 1;
                    }
                    next = next.wrapping_add(4);
                }
                Instruction::AddSubImm {
                    rd,
                    rn,
                    imm,
                    subtract,
                    set_flags,
                    is_64_bit,
                } => {
                    if set_flags || rd == 31 || rn == 31 {
                        return Err(Error::Unsupported(word));
                    }
                    body.push_str(&format!(
                        "  %p{temp} = getelementptr i64, ptr %regs, i64 {rn}\n  %v{temp} = load i64, ptr %p{temp}, align 8\n"
                    ));
                    let op = if subtract { "sub" } else { "add" };
                    body.push_str(&format!("  %a{temp} = {op} i64 %v{temp}, {imm}\n"));
                    let value = if is_64_bit {
                        format!("%a{temp}")
                    } else {
                        body.push_str(&format!("  %w{temp} = and i64 %a{temp}, 4294967295\n"));
                        format!("%w{temp}")
                    };
                    body.push_str(&format!(
                        "  %d{temp} = getelementptr i64, ptr %regs, i64 {rd}\n  store i64 {value}, ptr %d{temp}, align 8\n"
                    ));
                    temp += 1;
                    next = next.wrapping_add(4);
                }
                Instruction::Branch { displacement } => {
                    next = next.wrapping_add_signed(displacement);
                    body.push_str(&format!("  ret i64 {next}\n"));
                    terminated = true;
                }
                Instruction::Ret { rn } => {
                    if rn == 31 {
                        return Err(Error::Unsupported(word));
                    }
                    body.push_str(&format!(
                        "  %p{temp} = getelementptr i64, ptr %regs, i64 {rn}\n  %target{temp} = load i64, ptr %p{temp}, align 8\n  ret i64 %target{temp}\n"
                    ));
                    terminated = true;
                }
                Instruction::BranchCond { .. } => return Err(Error::Unsupported(word)),
            }
        }
        if !terminated {
            body.push_str(&format!("  ret i64 {next}\n"));
        }
        let ir = format!("define i64 @block(ptr %regs) {{\nentry:\n{body}}}\n");
        let jit = JitEngine::compile_llvm_ir(&ir).map_err(Error::Backend)?;
        let symbol = unsafe { jit.lookup::<unsafe extern "C" fn(*mut u64) -> u64>("block") }
            .map_err(|error| Error::Backend(BackendError::Llvm(error)))?;
        let entry = *symbol;
        Ok(Self { _jit: jit, entry })
    }

    /// Execute the block, updating guest registers and returning the next PC.
    pub fn run(&self, cpu: &mut Cpu) -> u64 {
        let next = unsafe { (self.entry)(cpu.x.as_mut_ptr()) };
        cpu.pc = next;
        next
    }
}

#[cfg(test)]
mod tests {
    use volt_jit_aarch64::Cpu;

    use super::Block;

    #[test]
    fn native_movz_block_updates_guest_state() {
        let instruction = 0xd280_0540u32.to_le_bytes();
        let block = Block::compile(0x1000, &instruction).expect("compile MOVZ block");
        let mut cpu = Cpu::default();
        assert_eq!(block.run(&mut cpu), 0x1004);
        assert_eq!(cpu.x[0], 42);
        assert_eq!(cpu.pc, 0x1004);
    }

    #[test]
    fn native_movk_add_and_return_use_live_registers() {
        let words = [0xd280_0020u32, 0xf2a0_0040, 0x9100_0c01, 0xd65f_0020];
        let bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let block = Block::compile(0x1000, &bytes).expect("compile arithmetic block");
        let mut cpu = Cpu::default();
        assert_eq!(block.run(&mut cpu), 0x0002_0004);
        assert_eq!(cpu.x[0], 0x0002_0001);
        assert_eq!(cpu.x[1], 0x0002_0004);
    }

    #[test]
    fn native_backward_branch_targets_its_own_pc() {
        let words = [0xd503_201fu32, 0x17ff_ffff];
        let bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let block = Block::compile(0x1000, &bytes).expect("compile branch block");
        let mut cpu = Cpu::default();
        assert_eq!(block.run(&mut cpu), 0x1000);
    }
}
