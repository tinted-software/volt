// Copyright 2026 LilithSemi
// SPDX-License-Identifier: Apache-2.0
// Modified: Rust port of Mirage lib/mirage-jit/user.zig.
//! Native translated AArch64 Linux userspace execution.
//! Scalar instructions only; signals, threads and file-backed/fixed mappings
//! are unsupported. Unknown syscalls return Linux -ENOSYS, never fake success.
pub mod loader;
pub mod memory;
pub mod syscalls;
pub use memory::Space;

use crate::aarch64::cpu::{Exclusive, Trap};
use crate::aarch64::host::{Dispatch, dispatch};
use crate::aarch64::{Cache, Cpu};
use crate::memory::{GuestMemory, MemoryError};
use std::ffi::OsString;
use std::path::Path;
use volt_target::aarch64::decode::{Instruction, SignExtend, SystemRegister};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Memory(MemoryError),
    Arm64(crate::aarch64::compile::Error),
    InvalidElf(&'static str),
    InvalidMapping,
    OverlappingMapping,
    AddressOverflow,
    ImageTooLarge,
    ArgumentsTooLarge,
    InvalidArgument,
    OutOfMemory,
    UnsupportedHost,
    InvalidAccessWidth,
    InvalidUserInstruction,
    Execution { pc: u64, cause: Box<Error> },
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "host I/O: {error}"),
            Self::Memory(error) => write!(f, "guest memory: {error:?}"),
            Self::Arm64(error) => write!(f, "AArch64 translation: {error:?}"),
            Self::InvalidElf(reason) => write!(f, "invalid ELF: {reason}"),
            Self::InvalidMapping => f.write_str("invalid guest mapping"),
            Self::OverlappingMapping => f.write_str("overlapping guest mappings"),
            Self::AddressOverflow => f.write_str("guest address overflow"),
            Self::ImageTooLarge => f.write_str("ELF image exceeds 1 GiB"),
            Self::ArgumentsTooLarge => f.write_str("initial arguments exceed guest stack"),
            Self::InvalidArgument => f.write_str("argument contains NUL"),
            Self::OutOfMemory => f.write_str("guest backing allocation failed"),
            Self::UnsupportedHost => f.write_str("requires Linux x86_64 or AArch64 host"),
            Self::InvalidAccessWidth => f.write_str("invalid guest access width"),
            Self::InvalidUserInstruction => f.write_str("invalid userspace instruction or trap"),
            Self::Execution { pc, cause } => write!(f, "guest pc=0x{pc:x}: {cause}"),
        }
    }
}
impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Execution { cause, .. } => Some(cause),
            _ => None,
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<MemoryError> for Error {
    fn from(error: MemoryError) -> Self {
        Self::Memory(error)
    }
}
impl From<crate::aarch64::compile::Error> for Error {
    fn from(error: crate::aarch64::compile::Error) -> Self {
        Self::Arm64(error)
    }
}

/// `args` includes argv[0]. Paths, arguments and environment use exact OS bytes.
/// A PT_INTERP pathname resolves directly in the host filesystem.
pub fn run(path: &Path, args: &[OsString], env: &[(OsString, OsString)]) -> Result<u8, Error> {
    if !cfg!(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )) {
        return Err(Error::UnsupportedHost);
    }
    let mut space = Space::new();
    let entry = loader::load(&mut space, path, args, env)?;
    let mut syscalls = syscalls::State::new(entry.brk)?;
    let mut cpu = Cpu::default();
    cpu.pc = entry.pc;
    cpu.sp = entry.sp;
    cpu.system.el = 0;
    cpu.system.spsel = false;
    cpu.system.daif = 0;
    let mut cache = Cache::new();
    let result = (|| {
        loop {
            cache.run_block(&mut cpu, |_, address, into| fetch(&space, address, into))?;
            match cpu.trap {
                Trap::None => {}
                Trap::Load | Trap::Store => service(&mut space, &mut cpu)?,
                Trap::Atomic => atomic(&mut space, &mut cpu)?,
                Trap::SimdStruct => simd_struct(&mut space, &mut cpu)?,
                Trap::Host => {
                    let word = cpu.value as u32;
                    crate::aarch64::host::run(&mut cpu, word);
                    cpu.trap = Trap::None;
                }
                Trap::Svc => {
                    cpu.trap = Trap::None;
                    cpu.monitor_valid = false;
                    if let Some(status) = syscalls.dispatch(&mut space, &mut cpu)? {
                        return Ok(status);
                    }
                }
                Trap::Isb | Trap::Tlbi => cpu.trap = Trap::None,
                Trap::DcZva => {
                    let address = cpu.address & !63;
                    space.write(address, &[0; 64])?;
                    cpu.monitor_valid = false;
                    cpu.trap = Trap::None;
                }
                _ => return Err(Error::InvalidUserInstruction),
            }
        }
    })();
    result.map_err(|cause| Error::Execution {
        pc: cpu.pc,
        cause: Box::new(cause),
    })
}
fn fetch(
    space: &Space,
    address: u64,
    into: &mut [u8],
) -> Result<usize, crate::aarch64::compile::Error> {
    space.check(address, 4, 4)?;
    let mut bytes = [0; 4];
    if let Ok(contents) = space.slice(address, 4) {
        bytes.copy_from_slice(contents);
    } else {
        for (i, out) in bytes.iter_mut().enumerate() {
            *out = space.slice(address + i as u64, 1)?[0];
        }
    }
    let word = u32::from_le_bytes(bytes);
    let dispatched =
        dispatch(word).map_err(|_| crate::aarch64::compile::Error::Decode { word, pc: address })?;
    if let Dispatch::Guest(instruction) = &dispatched {
        if !user_instruction(instruction) {
            return Err(crate::aarch64::compile::Error::InvalidUserInstruction);
        }
        if let Instruction::CacheOp(sys) = instruction {
            // The decoder groups EL0 and privileged cache operations together.
            // Only IC IVAU and DC CVAC/CVAU/CIVAC are permitted in userspace.
            if sys.op1 != 3 || sys.op2 != 1 || !matches!(sys.crm, 5 | 10 | 11 | 14) {
                return Err(crate::aarch64::compile::Error::InvalidUserInstruction);
            }
        }
    }
    if into.len() < 4 {
        return Err(crate::aarch64::compile::Error::InvalidInstructionLength);
    }
    into[..4].copy_from_slice(&bytes);
    Ok(4)
}
fn user_instruction(instruction: &Instruction) -> bool {
    match instruction {
        Instruction::System(operand) => match operand.register {
            SystemRegister::Nzcv
            | SystemRegister::TpidrEl0
            | SystemRegister::Fpcr
            | SystemRegister::Fpsr => true,
            SystemRegister::TpidrroEl0
            | SystemRegister::CtrEl0
            | SystemRegister::DczidEl0
            | SystemRegister::CntvctEl0
            | SystemRegister::CntfrqEl0 => operand.read,
            _ => false,
        },
        Instruction::Eret
        | Instruction::Psci
        | Instruction::Tlbi(_)
        | Instruction::Wfi
        | Instruction::AddressTranslate(_) => false,
        _ => true,
    }
}
/// Retire the translator's deferred scalar memory access, including pairs,
/// sign extension, post-index writeback and the local exclusive monitor.
pub fn service(space: &mut Space, cpu: &mut Cpu) -> Result<(), Error> {
    let loading = cpu.trap == Trap::Load;
    if cpu.exclusive == Exclusive::Store {
        let held = cpu.monitor_valid
            && cpu.monitor_address == cpu.address
            && cpu.monitor_width == cpu.width;
        cpu.monitor_valid = false;
        if !held {
            if cpu.status_dest != 31 {
                cpu.x[cpu.status_dest as usize] = 1;
            }
            cpu.exclusive = Exclusive::None;
            cpu.trap = Trap::None;
            return Ok(());
        }
    }
    let first = (cpu.address, cpu.width, cpu.dest, cpu.value, cpu.load_signed);
    access(
        space,
        cpu,
        loading,
        first.0,
        first.1,
        first.2,
        first.3,
        first.4,
        cpu.vector_dest,
    )?;
    cpu.vector_dest = false;
    if cpu.second_pending {
        let second = (
            cpu.second_address,
            cpu.second_width,
            cpu.second_dest,
            cpu.second_value,
            cpu.second_signed,
        );
        let second_vector = cpu.second_vector;
        access(
            space,
            cpu,
            loading,
            second.0,
            second.1,
            second.2,
            second.3,
            second.4,
            second_vector,
        )?;
        cpu.second_pending = false;
        cpu.second_signed = SignExtend::None;
        cpu.second_vector = false;
    }
    if cpu.writeback {
        if cpu.writeback_dest == 31 {
            cpu.sp = cpu.writeback_value;
        } else {
            cpu.x[cpu.writeback_dest as usize] = cpu.writeback_value;
        }
        cpu.writeback = false;
    }
    if cpu.exclusive == Exclusive::Load {
        cpu.monitor_valid = true;
        cpu.monitor_address = cpu.address;
        cpu.monitor_width = cpu.width;
    } else if !loading {
        cpu.monitor_valid = false;
        if cpu.exclusive == Exclusive::Store && cpu.status_dest != 31 {
            cpu.x[cpu.status_dest as usize] = 0;
        }
    }
    cpu.exclusive = Exclusive::None;
    cpu.load_signed = SignExtend::None;
    cpu.trap = Trap::None;
    Ok(())
}
/// Retire a SIMD structure load/store through the address space, then apply the
/// post-index writeback. The registers change only if every element succeeded.
fn simd_struct(space: &mut Space, cpu: &mut Cpu) -> Result<(), Error> {
    let desc = cpu.structure;
    let mut registers = cpu.v;
    crate::aarch64::simd_struct::execute(
        &desc,
        &mut registers,
        cpu.address,
        |address, width, store| -> Result<u64, Error> {
            let step = usize::from(width);
            if let Some(value) = store {
                space.check(address, step, 2)?;
                space.write(address, &value.to_le_bytes()[..step])?;
                return Ok(0);
            }
            space.check(address, step, 1)?;
            let mut bytes = [0; 8];
            space.read(address, &mut bytes[..step])?;
            Ok(u64::from_le_bytes(bytes))
        },
    )?;
    cpu.v = registers;
    if cpu.writeback {
        if cpu.writeback_dest == 31 {
            cpu.sp = cpu.writeback_value;
        } else {
            cpu.x[cpu.writeback_dest as usize] = cpu.writeback_value;
        }
        cpu.writeback = false;
    }
    cpu.trap = Trap::None;
    Ok(())
}
/// Retire an LSE atomic. The user-mode machine runs a single vCPU, so a plain
/// read-modify-write through the address space is atomic.
fn atomic(space: &mut Space, cpu: &mut Cpu) -> Result<(), Error> {
    let (kind, address, width) = (cpu.atomic_kind, cpu.address, cpu.width);
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(Error::InvalidAccessWidth);
    }
    let (words, step) = (kind.words(), usize::from(width));
    let span = words * step;
    // Read and write permission: the operation does both.
    space.check(address, span, 1 | 2)?;
    let mut bytes = [0u8; 16];
    space.read(address, &mut bytes[..span])?;
    let mut old = [0u64; 2];
    for (slot, value) in old.iter_mut().enumerate().take(words) {
        let mut word = [0u8; 8];
        word[..step].copy_from_slice(&bytes[slot * step..][..step]);
        *value = u64::from_le_bytes(word);
    }
    let new = if kind.is_exclusive_pair() {
        cpu.exclusive_pair(kind, old)
    } else {
        kind.apply(old, cpu.atomic_operands, width)
    };
    if new != old {
        for (slot, value) in new.iter().enumerate().take(words) {
            bytes[slot * step..][..step].copy_from_slice(&value.to_le_bytes()[..step]);
        }
        space.write(address, &bytes[..span])?;
        cpu.monitor_valid = false;
    }
    for (slot, value) in old.into_iter().enumerate().take(words) {
        let dest = usize::from(cpu.atomic_dests[slot]);
        if dest < 31 {
            cpu.x[dest] = value;
        }
    }
    cpu.trap = Trap::None;
    Ok(())
}
fn access(
    space: &mut Space,
    cpu: &mut Cpu,
    loading: bool,
    address: u64,
    width: u8,
    dest: u8,
    value: u64,
    signed: SignExtend,
    vector: bool,
) -> Result<(), Error> {
    if !matches!(width, 1 | 2 | 4 | 8 | 16) {
        return Err(Error::InvalidAccessWidth);
    }
    space.check(address, width as usize, if loading { 1 } else { 2 })?;
    if !loading {
        if vector {
            let v = cpu.v[dest as usize];
            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&v[0].to_le_bytes());
            bytes[8..16].copy_from_slice(&v[1].to_le_bytes());
            space.write(address, &bytes[..width as usize])?;
            return Ok(());
        }
        space.write(address, &value.to_le_bytes()[..width as usize])?;
        return Ok(());
    }
    if vector {
        let mut bytes = [0u8; 16];
        space.read(address, &mut bytes[..width as usize])?;
        let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let hi = if width == 16 {
            u64::from_le_bytes(bytes[8..16].try_into().unwrap())
        } else {
            0
        };
        let bits = (width as u32).min(8) * 8;
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        cpu.v[dest as usize] = [lo & mask, hi];
        return Ok(());
    }
    if dest == 31 {
        return Ok(());
    }
    let mut bytes = [0; 8];
    space.read(address, &mut bytes[..width as usize])?;
    let value = u64::from_le_bytes(bytes);
    let drop = 64 - u32::from(width) * 8;
    cpu.x[dest as usize] = match signed {
        SignExtend::None => value,
        SignExtend::To32 => (((value << drop) as i64 >> drop) as u64) & 0xffff_ffff,
        SignExtend::To64 => ((value << drop) as i64 >> drop) as u64,
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_pair_writeback_and_exclusives() {
        let mut space = Space::new();
        space.map(0x1000, 4).unwrap();
        space.map(0x1004, 12).unwrap();
        space.write(0x1000, &[0xff; 8]).unwrap();
        let mut cpu = Cpu::default();
        cpu.trap = Trap::Load;
        cpu.address = 0x1000;
        cpu.width = 4;
        cpu.dest = 0;
        cpu.load_signed = SignExtend::To64;
        cpu.second_pending = true;
        cpu.second_address = 0x1004;
        cpu.second_width = 4;
        cpu.second_dest = 1;
        cpu.second_signed = SignExtend::To32;
        cpu.writeback = true;
        cpu.writeback_dest = 31;
        cpu.writeback_value = 0x1008;
        cpu.exclusive = Exclusive::Load;
        service(&mut space, &mut cpu).unwrap();
        assert_eq!(cpu.x[0], u64::MAX);
        assert_eq!(cpu.x[1], 0xffff_ffff);
        assert_eq!(cpu.sp, 0x1008);
        assert!(cpu.monitor_valid);
        assert!(!cpu.second_pending);
        cpu.trap = Trap::Store;
        cpu.exclusive = Exclusive::Store;
        cpu.status_dest = 2;
        cpu.value = 0x12345678;
        service(&mut space, &mut cpu).unwrap();
        assert_eq!(cpu.x[2], 0);
        assert!(!cpu.monitor_valid);
        cpu.trap = Trap::Store;
        cpu.exclusive = Exclusive::Store;
        cpu.value = 0;
        service(&mut space, &mut cpu).unwrap();
        assert_eq!(cpu.x[2], 1);
        let mut bytes = [0; 4];
        space.read(0x1000, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), 0x12345678);
    }
    #[test]
    fn fetch_rejects_privilege_and_respects_execute_permissions() {
        let mut space = Space::new();
        space.map(0x1000, 4096).unwrap();
        space.write(0x1000, &0xd69f03e0u32.to_le_bytes()).unwrap(); // ERET
        space.protect(0x1000, 4096, 4).unwrap();
        assert!(matches!(
            fetch(&space, 0x1000, &mut [0; 4]),
            Err(crate::aarch64::compile::Error::InvalidUserInstruction)
        ));
        space
            .initialize(0x1000, &0xd503201fu32.to_le_bytes())
            .unwrap(); // NOP
        assert_eq!(fetch(&space, 0x1000, &mut [0; 4]).unwrap(), 4);
        space.protect(0x1000, 4096, 1).unwrap();
        assert!(fetch(&space, 0x1000, &mut [0; 4]).is_err());
    }
}
