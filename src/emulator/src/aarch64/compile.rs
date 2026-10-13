use super::{
    cpu::{Cpu, Trap},
    environment::CpuEnvironment,
    host::{ends_block, runs_on_host},
};
use crate::memory::MemoryError;
use core::fmt;
use volt_isa_aarch64::decode::decode;
use volt_lift_aarch64::{LiftError, Lifter};
use volt_target::native;

#[derive(Debug)]
pub enum Error {
    Memory(MemoryError),
    InvalidUserInstruction,
    Decode { word: u32, pc: u64 },
    Native(native::Error),
    EmptyBlock,
    UnsupportedInstruction,
    AddressOverflow,
    InvalidInstructionLength,
    Lift(LiftError),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(e) => write!(f, "{e}"),
            Self::Decode { word, pc } => write!(f, "unsupported instruction {word:08x} at {pc:x}"),
            Self::Native(e) => write!(f, "{e}"),
            Self::InvalidUserInstruction => {
                f.write_str("instruction is not permitted in user mode")
            }
            Self::EmptyBlock => f.write_str("empty instruction block"),
            Self::UnsupportedInstruction => f.write_str("unsupported instruction encoding"),
            Self::AddressOverflow => f.write_str("instruction address overflow"),
            Self::InvalidInstructionLength => f.write_str("invalid instruction fetch length"),
            Self::Lift(e) => write!(f, "{e}"),
        }
    }
}
impl core::error::Error for Error {}
impl From<MemoryError> for Error {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}
impl From<LiftError> for Error {
    fn from(e: LiftError) -> Self {
        Self::Lift(e)
    }
}
impl From<native::Error> for Error {
    fn from(e: native::Error) -> Self {
        Self::Native(e)
    }
}
pub struct Block {
    /// Keeps the executable mapping `entry` points into alive.
    _image: native::JittedFunction,
    entry: unsafe extern "C" fn(*mut Cpu) -> u64,
}
impl Block {
    fn new(image: native::JittedFunction) -> Self {
        // SAFETY: the generated function has exactly this signature, and the
        // pointer is only used while `_image` keeps its mapping alive.
        let entry = unsafe { image.entry() };
        Self {
            _image: image,
            entry,
        }
    }
    pub fn run(&self, cpu: &mut Cpu) -> u64 {
        cpu.trap = Trap::None;
        // SAFETY: see `new`; `cpu` is the exclusive, live CPU state the code expects.
        let next = unsafe { (self.entry)(cpu) };
        cpu.pc = next;
        next
    }
}
pub fn compile(guest_pc: u64, bytes: &[u8]) -> Result<Block, Error> {
    compile_with(guest_pc, bytes, false)
}

/// Compile a block. With `inline_memory`, plain loads and stores get an inline
/// data-TLB fast path and do not end the block; the caller's block scan must
/// use `host::ends_block(_, true)` to match.
pub fn compile_with(guest_pc: u64, bytes: &[u8], inline_memory: bool) -> Result<Block, Error> {
    if bytes.is_empty() {
        return Err(Error::EmptyBlock);
    }
    if bytes.len() % 4 != 0 {
        return Err(Error::UnsupportedInstruction);
    }
    let mut lifter = Lifter::new(CpuEnvironment::new(inline_memory));
    let mut pc = guest_pc;
    let mut ended = false;
    for chunk in bytes.chunks_exact(4) {
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        match decode(word) {
            Ok(instruction) if runs_on_host(&instruction) => {
                // The host runs this word (see `host::run`), and it ends the block.
                let (builder, env) = lifter.parts();
                env.host(builder, pc, word);
                ended = true;
            }
            Ok(instruction) => {
                ended = ends_block(&instruction, inline_memory);
                lifter.lift(pc, instruction)?;
            }
            Err(_) => {
                eprintln!("compile decode failure at pc {pc:x}: word {word:08x}, full block:");
                for (i, c) in bytes.chunks_exact(4).enumerate() {
                    let w = u32::from_le_bytes(c.try_into().unwrap());
                    eprintln!(
                        "  +{:04x} ({:x}): {:08x}",
                        i * 4,
                        guest_pc + i as u64 * 4,
                        w
                    );
                }
                return Err(Error::Decode { word, pc });
            }
        }
        pc = pc.wrapping_add(4);
        if ended {
            break;
        }
    }
    if !ended {
        lifter.exit(pc);
    }
    let (function, _) = lifter.into_parts();
    Ok(Block::new(native::compile_owned(function)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
    #[test]
    fn nzcv_write_read_and_branch_preserve_packed_flags() {
        // MSR NZCV,X0; MRS X1,NZCV; B.EQ +8.
        let block = compile(0x1000, &bytes(&[0xd51b4200, 0xd53b4201, 0x54000040])).unwrap();
        let mut cpu = Cpu::default();
        cpu.x[0] = 0x4000000f;
        assert_eq!(block.run(&mut cpu), 0x1010);
        assert_eq!(cpu.flags, 0x40000000);
        assert_eq!(cpu.x[1], 0x40000000);
        assert_eq!(cpu.system.cntvct_el0, 3);
        cpu.x[0] = 0x8000000f;
        assert_eq!(block.run(&mut cpu), 0x100c);
        assert_eq!(cpu.flags, 0x80000000);
    }
    #[test]
    fn pair_post_index_records_deferred_access_and_writeback() {
        // LDP X0,X1,[X2],#16.
        let block = compile(0x8000, &bytes(&[0xa8c10440])).unwrap();
        let mut cpu = Cpu::default();
        cpu.x[2] = 0x100;
        assert_eq!(block.run(&mut cpu), 0x8004);
        assert_eq!(cpu.trap, Trap::Load);
        assert_eq!(cpu.address, 0x100);
        assert_eq!(cpu.dest, 0);
        assert_eq!(cpu.width, 8);
        assert!(cpu.second_pending);
        assert_eq!(cpu.second_address, 0x108);
        assert_eq!(cpu.second_dest, 1);
        assert!(cpu.writeback);
        assert_eq!(cpu.writeback_dest, 2);
        assert_eq!(cpu.writeback_value, 0x110);
        assert_eq!(cpu.x[2], 0x100);
        assert_eq!(cpu.system.cntvct_el0, 1);
    }

    #[test]
    fn zero_register_destinations_do_not_overwrite_sp() {
        // ORR XZR,X0,X0; ADR XZR,. ; MRS XZR,NZCV.
        let block = compile(0x1000, &bytes(&[0xaa00001f, 0x1000001f, 0xd53b421f])).unwrap();
        let mut cpu = Cpu::default();
        cpu.sp = 0x9000;
        cpu.x[0] = 0x1234;
        block.run(&mut cpu);
        assert_eq!(cpu.sp, 0x9000);
        // CBZ XZR,+8 must take the branch independently of SP.
        let branch = compile(0x2000, &bytes(&[0xb400005f])).unwrap();
        assert_eq!(branch.run(&mut cpu), 0x2008);
    }
    // ---- inline data-TLB fast path ----
    use super::super::cpu::{DTLB_INVALID, Dtlb};
    const SVC: u32 = 0xd4000001;
    /// Guest page 0x10000 backed by `ram`, readable and (optionally) writable.
    fn map_page(cpu: &mut Cpu, ram: &mut [u8; 4096], writable: bool) {
        let page = 0x10000u64;
        let entry = &mut cpu.dtlb.entries[Dtlb::index(page)];
        entry.read = page;
        entry.write = if writable { page } else { DTLB_INVALID };
        entry.addend = (ram.as_mut_ptr() as u64).wrapping_sub(page);
    }
    fn run_inline(cpu: &mut Cpu, words: &[u32]) {
        let block = compile_with(0x2000, &bytes(words), true).unwrap();
        block.run(cpu);
    }
    #[test]
    fn inline_load_hits_without_leaving_the_block() {
        let mut ram = [0u8; 4096];
        ram[8..16].copy_from_slice(&0x1122334455667788u64.to_le_bytes());
        let mut cpu = Cpu::default();
        map_page(&mut cpu, &mut ram, false);
        cpu.x[0] = 0x10008;
        run_inline(&mut cpu, &[0xf9400001, SVC]); // ldr x1,[x0]; svc
        assert_eq!(cpu.x[1], 0x1122334455667788);
        assert_eq!(cpu.trap, Trap::Svc, "continued past the load");
    }
    #[test]
    fn inline_load_miss_defers_to_the_machine() {
        let mut cpu = Cpu::default();
        cpu.x[0] = 0x10008;
        run_inline(&mut cpu, &[0xf9400001, SVC]);
        assert_eq!(cpu.trap, Trap::Load);
        assert_eq!(cpu.address, 0x10008);
        assert_eq!((cpu.width, cpu.dest), (8, 1));
        assert_eq!(cpu.pc, 0x2004, "resumes after the load");
    }
    #[test]
    fn inline_store_needs_write_permission() {
        let mut ram = [0u8; 4096];
        let mut cpu = Cpu::default();
        cpu.x[0] = 0x10000;
        cpu.x[1] = 0xdead_beef_cafe_f00d;
        map_page(&mut cpu, &mut ram, true);
        run_inline(&mut cpu, &[0xf9000401, SVC]); // str x1,[x0,#8]
        assert_eq!(&ram[8..16], &0xdead_beef_cafe_f00du64.to_le_bytes());
        assert_eq!(cpu.trap, Trap::Svc);
        // Read-only entry: the store must go to the machine and leave RAM alone.
        let mut ram = [0u8; 4096];
        let mut cpu = Cpu::default();
        cpu.x[0] = 0x10000;
        cpu.x[1] = 1;
        map_page(&mut cpu, &mut ram, false);
        run_inline(&mut cpu, &[0xf9000401, SVC]);
        assert_eq!(cpu.trap, Trap::Store);
        assert_eq!(ram[8], 0);
    }
    #[test]
    fn inline_sizes_and_sign_extension() {
        let mut ram = [0u8; 4096];
        ram[0..8].copy_from_slice(&0xffff_ffff_8765_f0f1u64.to_le_bytes());
        let mut cpu = Cpu::default();
        map_page(&mut cpu, &mut ram, true);
        let cases: [(u32, u64); 7] = [
            (0x39400001, 0xf1),                  // ldrb w1
            (0x39800001, 0xffff_ffff_ffff_fff1), // ldrsb x1
            (0x39c00001, 0xffff_fff1),           // ldrsb w1
            (0x79400001, 0xf0f1),                // ldrh w1
            (0x79800001, 0xffff_ffff_ffff_f0f1), // ldrsh x1
            (0xb9400001, 0x8765_f0f1),           // ldr w1
            (0xb9800001, 0xffff_ffff_8765_f0f1), // ldrsw x1
        ];
        for (word, want) in cases {
            cpu.x[0] = 0x10000;
            cpu.x[1] = 0x5555_5555_5555_5555;
            run_inline(&mut cpu, &[word, SVC]);
            assert_eq!(cpu.x[1], want, "{word:#x}");
            assert_eq!(cpu.trap, Trap::Svc, "{word:#x} should hit");
        }
        // Narrow stores touch only their own bytes.
        ram.fill(0xaa);
        cpu.x[0] = 0x10010;
        cpu.x[1] = 0x1122_3344_5566_7788;
        run_inline(&mut cpu, &[0x39000001, SVC]); // strb
        run_inline(&mut cpu, &[0x79000001, SVC]); // strh at same address
        assert_eq!(&ram[0x10..0x13], &[0x88, 0x77, 0xaa]);
        cpu.x[0] = 0x10020;
        run_inline(&mut cpu, &[0xb9000001, SVC]); // str w1
        assert_eq!(&ram[0x20..0x25], &[0x88, 0x77, 0x66, 0x55, 0xaa]);
    }
    #[test]
    fn inline_post_index_updates_the_base_on_both_paths() {
        let mut ram = [0u8; 4096];
        ram[0..8].copy_from_slice(&7u64.to_le_bytes());
        let mut cpu = Cpu::default();
        map_page(&mut cpu, &mut ram, false);
        cpu.x[0] = 0x10000;
        run_inline(&mut cpu, &[0xf8408401, SVC]); // ldr x1,[x0],#8
        assert_eq!((cpu.x[1], cpu.x[0]), (7, 0x10008));
        assert_eq!(cpu.trap, Trap::Svc);
        // Miss: the machine applies the writeback after the access.
        let mut cpu = Cpu::default();
        cpu.x[0] = 0x10000;
        run_inline(&mut cpu, &[0xf8408401, SVC]);
        assert_eq!(cpu.trap, Trap::Load);
        assert!(cpu.writeback);
        assert_eq!((cpu.writeback_dest, cpu.writeback_value), (0, 0x10008));
        assert_eq!(
            cpu.x[0], 0x10000,
            "base untouched until the access completes"
        );
    }
    #[test]
    fn inline_pair_loads_and_stores_both_registers_in_place() {
        let mut ram = [0u8; 4096];
        ram[0..8].copy_from_slice(&0x1111u64.to_le_bytes());
        ram[8..16].copy_from_slice(&0x2222u64.to_le_bytes());
        let mut cpu = Cpu::default();
        map_page(&mut cpu, &mut ram, true);
        cpu.x[0] = 0x10000;
        run_inline(&mut cpu, &[0xa9400801, SVC]); // ldp x1,x2,[x0]
        assert_eq!(cpu.trap, Trap::Svc, "a hit must not leave the block");
        assert_eq!((cpu.x[1], cpu.x[2]), (0x1111, 0x2222));
        cpu.x[1] = 0xaaaa;
        cpu.x[2] = 0xbbbb;
        cpu.x[0] = 0x10010;
        run_inline(&mut cpu, &[0xa9000801, SVC]); // stp x1,x2,[x0]
        assert_eq!(cpu.trap, Trap::Svc);
        assert_eq!(u64::from_le_bytes(ram[16..24].try_into().unwrap()), 0xaaaa);
        assert_eq!(u64::from_le_bytes(ram[24..32].try_into().unwrap()), 0xbbbb);
    }
    #[test]
    fn inline_pair_straddling_a_page_defers_both_accesses_to_the_machine() {
        let mut ram = [0u8; 4096];
        let mut cpu = Cpu::default();
        map_page(&mut cpu, &mut ram, true);
        // Aligned, and the first half is in the mapped page, but the second
        // half is on the next page.
        cpu.x[0] = 0x10ff8;
        run_inline(&mut cpu, &[0xa9400801, SVC]);
        assert_eq!(cpu.trap, Trap::Load);
        assert!(cpu.second_pending);
        assert_eq!((cpu.address, cpu.second_address), (0x10ff8, 0x11000));
    }
    #[test]
    fn inline_pre_index_updates_the_base_only_once_the_access_succeeds() {
        let mut ram = [0u8; 4096];
        ram[8..16].copy_from_slice(&5u64.to_le_bytes());
        ram[16..24].copy_from_slice(&6u64.to_le_bytes());
        let mut cpu = Cpu::default();
        map_page(&mut cpu, &mut ram, false);
        cpu.x[0] = 0x10000;
        run_inline(&mut cpu, &[0xa9c08801, SVC]); // ldp x1,x2,[x0,#8]!
        assert_eq!((cpu.x[1], cpu.x[2], cpu.x[0]), (5, 6, 0x10008));
        // Miss: a faulting access must leave the base for the retry.
        let mut cpu = Cpu::default();
        cpu.x[0] = 0x10000;
        run_inline(&mut cpu, &[0xa9c08801, SVC]);
        assert_eq!(cpu.trap, Trap::Load);
        assert_eq!(cpu.address, 0x10008);
        assert!(cpu.writeback);
        assert_eq!((cpu.writeback_dest, cpu.writeback_value), (0, 0x10008));
        assert_eq!(cpu.x[0], 0x10000);
    }
    #[test]
    fn inline_misaligned_and_wrong_page_accesses_miss() {
        let mut ram = [0u8; 4096];
        let mut cpu = Cpu::default();
        map_page(&mut cpu, &mut ram, true);
        cpu.x[0] = 0x10004; // 8-byte load, 4-byte aligned only
        run_inline(&mut cpu, &[0xf9400001, SVC]);
        assert_eq!(cpu.trap, Trap::Load);
        let mut cpu2 = Cpu::default();
        map_page(&mut cpu2, &mut ram, true);
        cpu2.x[0] = 0x10000 + 0x1000 * 256; // same dtlb index, different page
        run_inline(&mut cpu2, &[0xf9400001, SVC]);
        assert_eq!(cpu2.trap, Trap::Load);
    }
}
