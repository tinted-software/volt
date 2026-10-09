//! The emulator's [`Environment`]: guest state is the [`Cpu`] struct, memory goes
//! through the inline data TLB and the trap-and-resume slow path, and system
//! instructions read and write [`CpuSystem`].
use super::cpu::{Cpu, DTLB_ENTRIES, Exclusive as CpuExclusive, System as CpuSystem, Trap};
use core::mem::offset_of;
use volt_ir::function::{BinOp as B, CmpOp as C, Value};
use volt_lift_aarch64::{Builder, Environment, LiftError, MemoryAccess, Writeback};
use volt_target::aarch64::decode::*;

/// Lowers guest instructions against [`Cpu`]: the lifter's machine policy.
pub(super) struct CpuEnvironment {
    /// Plain loads and stores get an inline data-TLB fast path and do not end
    /// the block; the block scan must use `Instruction::terminates_with(true)`.
    inline_memory: bool,
    /// Guest instructions started but not yet added to `cntvct_el0`. The
    /// counter is bumped once per exit path (`exit`) and before anything that
    /// reads it, instead of with a load/add/store per instruction.
    pending: u64,
}
impl CpuEnvironment {
    pub(super) fn new(inline_memory: bool) -> Self {
        Self {
            inline_memory,
            pending: 0,
        }
    }
    /// A word the host runs (see `host::run`) instead of the lifter: it
    /// counts as a retired instruction, hands the raw word to the machine, and
    /// ends the block.
    pub(super) fn host(&mut self, b: &Builder, pc: u64, word: u32) {
        self.pending += 1;
        b.store(offset_of!(Cpu, value), b.k(b.i64, word as u64));
        self.trap(b, pc.wrapping_add(4), Trap::Host);
    }
    fn add_counter(&self, b: &Builder, n: u64) {
        let at = offset_of!(Cpu, system) + offset_of!(CpuSystem, cntvct_el0);
        b.store(at, b.imm(b.i64, B::Add, b.load(b.i64, at), n));
    }
    /// Make `cntvct_el0` in memory current on the path being emitted.
    fn flush_counter(&mut self, b: &Builder) {
        let n = core::mem::replace(&mut self.pending, 0);
        if n != 0 {
            self.add_counter(b, n);
        }
    }
    /// Leave the function with the counter current. `pending` is not cleared:
    /// sibling paths (slow blocks) share the main path's count.
    fn leave(&self, b: &Builder, next_pc: Value) {
        if self.pending != 0 {
            self.add_counter(b, self.pending);
        }
        b.set_ret(next_pc)
    }
    /// Hand the base-register update of an access to the machine, which applies
    /// it once the access has completed.
    fn defer_writeback(&self, b: &Builder, w: Writeback) {
        b.store(offset_of!(Cpu, writeback_value), w.value);
        b.byte(offset_of!(Cpu, writeback_dest), w.register as u64);
        b.byte(offset_of!(Cpu, writeback), 1);
    }

    /// Probe the inline data TLB for `address`. Returns `(hit, host_pointer)`;
    /// the pointer is only meaningful when `hit`. A hit requires the page to be
    /// cached for this access kind and the access to be naturally aligned (so it
    /// cannot cross a page), because the tag compare covers the low `size - 1`
    /// address bits too.
    fn dtlb_probe(&self, b: &Builder, address: Value, size: Size, write: bool) -> (Value, Value) {
        let n = size as u64;
        // Entry offset = ((address >> 12) & (ENTRIES - 1)) * 32 = (address >> 7) & mask.
        let index = b.imm(
            b.i64,
            B::BitAnd,
            b.imm(b.i64, B::Shr, address, 7),
            (DTLB_ENTRIES as u64 - 1) << 5,
        );
        let entry = b.bin(b.ptr, B::Add, b.state(), index);
        // Constant displacements fold into the load addressing.
        let field = |at: usize| {
            b.load_ptr(
                b.i64,
                b.imm(b.ptr, B::Add, entry, (offset_of!(Cpu, dtlb) + at) as u64),
            )
        };
        let tag = field(if write { 8 } else { 0 });
        let addend = field(16);
        let host = b.bin(b.ptr, B::Add, address, addend);
        let want = b.imm(b.i64, B::BitAnd, address, !0xfffu64 | (n - 1));
        // Last, so the backend can fuse it into the branch that consumes it.
        let hit = b.cmp(C::Eq, tag, want);
        (hit, host)
    }
    fn fast_load(&self, b: &Builder, host: Value, size: Size, rt: u8, signed: SignExtend) {
        let bits = size as u16 * 8;
        let raw = b.load_ptr(b.int(bits), host);
        if rt == 31 {
            return;
        }
        let mut v = b.cv(b.i64, raw);
        if signed != SignExtend::None && bits < 64 {
            let n = 64 - bits as u64;
            let extended = b.shift(Width::X64, b.imm(b.i64, B::Shl, v, n), Shift::Asr, n);
            v = if signed == SignExtend::To32 {
                b.imm(b.i64, B::BitAnd, extended, 0xffff_ffff)
            } else {
                extended
            };
        }
        b.put(rt, v);
    }
    fn fast_store(&self, b: &Builder, host: Value, size: Size, rt: u8) {
        let t = b.int(size as u16 * 8);
        b.store_ptr(host, b.cv(t, b.reg(b.i64, rt, true)));
    }
    /// Emit a load or store with an inline fast path. On a data-TLB hit the
    /// access happens in place and emission continues in a fresh block; on a
    /// miss control goes to a slow block that hands the access to the machine
    /// (`slow` fills in the request) and returns.
    fn memory_access(
        &self,
        b: &Builder,
        address: Value,
        size: Size,
        op: MemoryOp,
        rt: u8,
        signed: SignExtend,
        post: Option<Writeback>,
        slow: impl FnOnce(&Self),
    ) {
        let write = op == MemoryOp::Store;
        let (hit, host) = self.dtlb_probe(b, address, size, write);
        let (fast_block, slow_block) = (b.new_block(), b.new_block());
        b.branch_if(hit, fast_block, slow_block);
        b.set_block(slow_block);
        slow(self);
        b.set_block(fast_block);
        if write {
            self.fast_store(b, host, size, rt);
        } else {
            self.fast_load(b, host, size, rt, signed);
        }
        if let Some(w) = post {
            b.put_sp(w.register, w.value);
        }
    }
    /// Inline fast path for a GPR load/store pair: one data-TLB probe covers
    /// both accesses when the first is naturally aligned and the second starts
    /// in the same page. Anything else takes `slow`, which hands the whole pair
    /// to the machine.
    fn pair_access(
        &self,
        b: &Builder,
        address: Value,
        a: Pair,
        post: Option<Writeback>,
        slow: impl FnOnce(&Self),
    ) {
        let write = a.op == MemoryOp::Store;
        let n = a.size as u64;
        let (hit, host) = self.dtlb_probe(b, address, a.size, write);
        let (check_block, fast_block, slow_block) = (b.new_block(), b.new_block(), b.new_block());
        b.branch_if(hit, check_block, slow_block);
        // `address` is aligned to `n`, so the second access leaves the page
        // exactly when it starts on a page boundary.
        b.set_block(check_block);
        let second_at = b.imm(b.i64, B::Add, address, n);
        let offset = b.imm(b.i64, B::BitAnd, second_at, 0xfff);
        let same_page = b.cmp(C::Ne, offset, b.k(b.i64, 0));
        b.branch_if(same_page, fast_block, slow_block);
        b.set_block(slow_block);
        slow(self);
        b.set_block(fast_block);
        let second = b.imm(b.ptr, B::Add, host, n);
        if write {
            self.fast_store(b, host, a.size, a.rt);
            self.fast_store(b, second, a.size, a.rt2);
        } else {
            self.fast_load(b, host, a.size, a.rt, a.signed);
            self.fast_load(b, second, a.size, a.rt2, a.signed);
        }
        if let Some(w) = post {
            b.put_sp(w.register, w.value);
        }
    }
    fn trap(&self, b: &Builder, pc: u64, t: Trap) {
        b.byte(offset_of!(Cpu, trap), t as u64);
        self.leave(b, b.k(b.i64, pc))
    }

    /// Slow path for a whole SIMD vector or vector-pair access: describe the
    /// full instruction in the CPU request fields (`vector_dest`, optionally
    /// `second_*`) and trap out; the machine completes every byte and resumes
    /// after the instruction.
    fn simd_trap_access(
        &self,
        b: &Builder,
        address: Value,
        bytes: u64,
        rt: u8,
        rt2: Option<u8>,
        post: Option<Writeback>,
        pc: u64,
        trap: Trap,
    ) {
        b.store(offset_of!(Cpu, address), address);
        b.byte(offset_of!(Cpu, width), bytes);
        b.byte(offset_of!(Cpu, dest), rt as u64);
        b.byte(offset_of!(Cpu, vector_dest), 1);
        if let Some(rt2) = rt2 {
            let second = b.imm(b.i64, B::Add, address, if bytes == 16 { 16 } else { 8 });
            b.store(offset_of!(Cpu, second_address), second);
            b.byte(offset_of!(Cpu, second_width), bytes);
            b.byte(offset_of!(Cpu, second_dest), rt2 as u64);
            b.byte(offset_of!(Cpu, second_vector), 1);
            b.byte(offset_of!(Cpu, second_pending), 1);
        }
        if let Some(w) = post {
            self.defer_writeback(b, w);
        }
        self.trap(b, pc.wrapping_add(4), trap);
    }
    /// Describe a GPR pair access to the machine: the first access in the usual
    /// request fields, the second as `second_*`, completed after the first.
    fn setup_pair(&self, b: &Builder, address: Value, a: Pair) {
        self.setup_memory(b, address, a.size, a.rt, a.signed, a.op);
        b.store(
            offset_of!(Cpu, second_address),
            b.imm(b.i64, B::Add, address, a.size as u64),
        );
        b.byte(offset_of!(Cpu, second_width), a.size as u64);
        b.byte(offset_of!(Cpu, second_dest), a.rt2 as u64);
        if a.signed != SignExtend::None {
            b.byte(offset_of!(Cpu, second_signed), a.signed as u64)
        }
        if a.op == MemoryOp::Store {
            b.store(offset_of!(Cpu, second_value), b.reg(b.i64, a.rt2, true))
        }
        b.byte(offset_of!(Cpu, second_pending), 1);
    }
    fn setup_memory(
        &self,
        b: &Builder,
        address: Value,
        size: Size,
        rt: u8,
        signed: SignExtend,
        op: MemoryOp,
    ) {
        b.store(offset_of!(Cpu, address), address);
        b.byte(offset_of!(Cpu, width), size as u64);
        b.byte(offset_of!(Cpu, dest), rt as u64);
        if signed != SignExtend::None {
            b.byte(offset_of!(Cpu, load_signed), signed as u64)
        }
        if op == MemoryOp::Store {
            b.store(offset_of!(Cpu, value), b.reg(b.i64, rt, true))
        }
    }

    /// Apply a base-register update the way a non-inline access does: a
    /// pre-index update immediately, a post-index one via the machine.
    fn writeback_eager(&self, b: &Builder, post: Option<Writeback>) {
        match post {
            Some(w) if w.pre_index => b.put_sp(w.register, w.value),
            Some(w) => self.defer_writeback(b, w),
            None => {}
        }
    }
    fn scalar(&self, b: &Builder, pc: u64, a: Memory, address: Value, post: Option<Writeback>) {
        let trap = if a.op == MemoryOp::Store {
            Trap::Store
        } else {
            Trap::Load
        };
        if self.inline_memory {
            self.memory_access(b, address, a.size, a.op, a.rt, a.signed, post, |env| {
                env.setup_memory(b, address, a.size, a.rt, a.signed, a.op);
                if let Some(w) = post {
                    env.defer_writeback(b, w);
                }
                env.trap(b, pc.wrapping_add(4), trap);
            });
        } else {
            self.writeback_eager(b, post);
            self.setup_memory(b, address, a.size, a.rt, a.signed, a.op);
            self.trap(b, pc.wrapping_add(4), trap);
        }
    }
    fn literal(&self, b: &Builder, pc: u64, a: Literal, address: Value) {
        if self.inline_memory {
            self.memory_access(
                b,
                address,
                a.size,
                MemoryOp::Load,
                a.rt,
                SignExtend::None,
                None,
                |env| {
                    env.setup_memory(b, address, a.size, a.rt, SignExtend::None, MemoryOp::Load);
                    env.trap(b, pc.wrapping_add(4), Trap::Load);
                },
            );
        } else {
            self.setup_memory(b, address, a.size, a.rt, SignExtend::None, MemoryOp::Load);
            self.trap(b, pc.wrapping_add(4), Trap::Load);
        }
    }
    fn pair(&self, b: &Builder, pc: u64, a: Pair, address: Value, post: Option<Writeback>) {
        let trap = if a.op == MemoryOp::Store {
            Trap::Store
        } else {
            Trap::Load
        };
        if self.inline_memory {
            self.pair_access(b, address, a, post, |env| {
                env.setup_pair(b, address, a);
                if let Some(w) = post {
                    env.defer_writeback(b, w);
                }
                env.trap(b, pc.wrapping_add(4), trap);
            });
        } else {
            self.writeback_eager(b, post);
            self.setup_pair(b, address, a);
            self.trap(b, pc.wrapping_add(4), trap);
        }
    }
    fn simd_pair(
        &self,
        b: &Builder,
        pc: u64,
        a: SimdPair,
        address: Value,
        post: Option<Writeback>,
    ) {
        let trap = if a.op == MemoryOp::Store {
            Trap::Store
        } else {
            Trap::Load
        };
        if !self.inline_memory {
            self.simd_trap_access(
                b,
                address,
                a.bytes as u64,
                a.rt,
                Some(a.rt2),
                post,
                pc,
                trap,
            );
        } else {
            let at_rt = b.v_at(a.rt);
            let at_rt2 = b.v_at(a.rt2);
            let slow_block = b.new_block();
            let store = a.op == MemoryOp::Store;
            let (bytes, chunk_count) = if a.bytes == 16 { (16, 4) } else { (8, 2) };
            let starts: [Value; 4] = [
                address,
                b.imm(b.i64, B::Add, address, 8),
                b.imm(b.i64, B::Add, address, 16),
                b.imm(b.i64, B::Add, address, 24),
            ];
            // stores: (reg_offset_base, half) per chunk
            let src: [(usize, u64); 4] = if bytes == 16 {
                [(at_rt, 0), (at_rt + 8, 0), (at_rt2, 0), (at_rt2 + 8, 0)]
            } else {
                [(at_rt, 0), (at_rt2, 0), (at_rt, 0), (at_rt, 0)]
            };
            let mut hosts: [Option<Value>; 4] = [None; 4];
            for i in 0..chunk_count {
                let next = b.new_block();
                let (hit, host) = self.dtlb_probe(b, starts[i], Size::Double, store);
                b.branch_if(hit, next, slow_block);
                b.set_block(next);
                hosts[i] = Some(host);
            }
            if store {
                for i in 0..chunk_count {
                    let host = hosts[i].unwrap();
                    b.store_ptr(host, b.load(b.i64, src[i].0));
                }
            } else {
                for i in 0..chunk_count {
                    let host = hosts[i].unwrap();
                    b.store(src[i].0, b.load_ptr(b.i64, host));
                }
                // zero the untransferred half of each 128-bit register
                if bytes == 8 {
                    b.store(at_rt + 8, b.k(b.i64, 0));
                    b.store(at_rt2 + 8, b.k(b.i64, 0));
                }
            }
            if let Some(w) = post {
                b.put_sp(w.register, w.value);
            }
            // slow path, built last so every fast edge into it exists.
            let resume = b.block();
            b.set_block(slow_block);
            self.simd_trap_access(b, address, bytes as u64, a.rt, Some(a.rt2), post, pc, trap);
            b.set_block(resume);
        }
    }
    fn simd_memory(
        &self,
        b: &Builder,
        pc: u64,
        a: SimdMemory,
        address: Value,
        post: Option<Writeback>,
    ) {
        let trap = if a.op == MemoryOp::Store {
            Trap::Store
        } else {
            Trap::Load
        };
        if !self.inline_memory {
            self.simd_trap_access(b, address, a.bytes as u64, a.rt, None, post, pc, trap);
        } else {
            let at_rt = b.v_at(a.rt);
            let slow_block = b.new_block();
            let store = a.op == MemoryOp::Store;
            // One chunk for widths up to 8 bytes, two 8-byte chunks for 16.
            let wide = a.bytes == 16;
            let size = match a.bytes {
                1 => Size::Byte,
                2 => Size::Half,
                4 => Size::Word,
                _ => Size::Double,
            };
            let chunk_count = if wide { 2 } else { 1 };
            let starts: [Value; 2] = [address, b.imm(b.i64, B::Add, address, 8)];
            let chunk_ty = b.int((a.bytes.min(8) as u16) * 8);
            let src: [usize; 2] = [at_rt, at_rt + 8];
            let mut hosts: [Option<Value>; 2] = [None; 2];
            for i in 0..chunk_count {
                let next = b.new_block();
                let (hit, host) = self.dtlb_probe(b, starts[i], size, store);
                b.branch_if(hit, next, slow_block);
                b.set_block(next);
                hosts[i] = Some(host);
            }
            if store {
                for i in 0..chunk_count {
                    let host = hosts[i].unwrap();
                    b.store_ptr(host, b.cv(chunk_ty, b.load(b.i64, src[i])));
                }
            } else {
                for i in 0..chunk_count {
                    let host = hosts[i].unwrap();
                    b.store(src[i], b.cv(b.i64, b.load_ptr(chunk_ty, host)));
                }
                // Scalar loads zero the whole upper register, not just the
                // untransferred half.
                if a.bytes <= 8 {
                    b.store(at_rt + 8, b.k(b.i64, 0));
                    if a.bytes < 8 {
                        let kept = (1u64 << (a.bytes * 8)) - 1;
                        let lo = b.imm(b.i64, B::BitAnd, b.load(b.i64, at_rt), kept);
                        b.store(at_rt, lo);
                    }
                }
            }
            if let Some(w) = post {
                b.put_sp(w.register, w.value);
            }
            let resume = b.block();
            b.set_block(slow_block);
            self.simd_trap_access(b, address, a.bytes as u64, a.rt, None, post, pc, trap);
            b.set_block(resume);
        }
    }
    fn exclusive(&self, b: &Builder, pc: u64, a: Exclusive) {
        self.setup_memory(
            b,
            b.reg(b.i64, a.rn, false),
            a.size,
            a.rt,
            SignExtend::None,
            a.op,
        );
        if a.op == MemoryOp::Store {
            b.byte(offset_of!(Cpu, status_dest), a.rs as u64)
        }
        b.byte(
            offset_of!(Cpu, exclusive),
            if a.op == MemoryOp::Store {
                CpuExclusive::Store as u64
            } else {
                CpuExclusive::Load as u64
            },
        );
        self.trap(
            b,
            pc.wrapping_add(4),
            if a.op == MemoryOp::Store {
                Trap::Store
            } else {
                Trap::Load
            },
        );
    }
    fn atomic(&self, b: &Builder, pc: u64, a: Atomic) {
        // Read-modify-write needs a host atomic, so it always takes the machine
        // slow path. `dests` receive the words read (31 discards) and `operands`
        // are what the machine combines with memory (see `AtomicKind::apply`).
        let (dests, operands): ([u8; 2], &[u8]) = match a.kind {
            AtomicKind::Casp => ([a.rs, a.rs + 1], &[a.rs, a.rs + 1, a.rt, a.rt + 1]),
            AtomicKind::Cas => ([a.rs, 31], &[a.rs, a.rt]),
            AtomicKind::LoadPair => ([a.rt, a.rt2], &[]),
            AtomicKind::StorePair => ([31, 31], &[a.rt, a.rt2]),
            _ => ([a.rt, 31], &[a.rs]),
        };
        b.store(offset_of!(Cpu, address), b.reg(b.i64, a.rn, false));
        b.byte(offset_of!(Cpu, width), a.size as u64);
        b.byte(offset_of!(Cpu, dest), dests[0] as u64);
        b.byte(offset_of!(Cpu, atomic_kind), a.kind as u64);
        for (slot, &dest) in dests.iter().enumerate() {
            b.byte(offset_of!(Cpu, atomic_dests) + slot, dest as u64);
        }
        for (slot, &reg) in operands.iter().enumerate() {
            // Register numbers past 30 read as XZR.
            b.store(
                offset_of!(Cpu, atomic_operands) + slot * 8,
                b.reg(b.i64, reg, true),
            );
        }
        if a.kind == AtomicKind::StorePair {
            b.byte(offset_of!(Cpu, status_dest), a.rs as u64);
        }
        self.trap(b, pc.wrapping_add(4), Trap::Atomic);
    }
    fn simd_struct(&self, b: &Builder, pc: u64, a: SimdStruct) {
        use volt_target::aarch64::simd_struct::StructDesc;
        // Element-wise accesses with lane merging need the machine slow path.
        let d = a.desc;
        let address = b.reg(b.i64, a.rn, false);
        b.store(offset_of!(Cpu, address), address);
        for (field, value) in [
            (offset_of!(StructDesc, store), u64::from(d.store)),
            (offset_of!(StructDesc, shape), d.shape as u64),
            (offset_of!(StructDesc, registers), u64::from(d.registers)),
            (
                offset_of!(StructDesc, interleaved),
                u64::from(d.interleaved),
            ),
            (offset_of!(StructDesc, esize), u64::from(d.esize)),
            (offset_of!(StructDesc, lane), u64::from(d.lane)),
            (offset_of!(StructDesc, q), u64::from(d.q)),
            (offset_of!(StructDesc, rt), u64::from(d.rt)),
        ] {
            b.byte(offset_of!(Cpu, structure) + field, value);
        }
        let step = match a.post {
            StructPost::None => None,
            StructPost::Immediate => Some(b.k(b.i64, d.bytes())),
            StructPost::Register(rm) => Some(b.reg(b.i64, rm, true)),
        };
        if let Some(step) = step {
            b.store(
                offset_of!(Cpu, writeback_value),
                b.bin(b.i64, B::Add, address, step),
            );
            b.byte(offset_of!(Cpu, writeback_dest), a.rn as u64);
            b.byte(offset_of!(Cpu, writeback), 1);
        }
        self.trap(b, pc.wrapping_add(4), Trap::SimdStruct);
    }

    fn lower_system(&mut self, b: &Builder, a: System) {
        use SystemRegister::*;
        self.flush_counter(b);
        let at = offset_of!(Cpu, system);
        let level = match a.register {
            SpEl0 => Some(0u64),
            SpEl1 => Some(1),
            SpEl2 => Some(2),
            _ => None,
        };
        if let Some(level) = level {
            let current = b.cv(b.i64, b.load(b.i8, at + offset_of!(CpuSystem, el)));
            let at_level = b.cmp(C::Eq, current, b.k(b.i64, level));
            let live = if level <= 1 {
                let el1 = b.cmp(C::Eq, current, b.k(b.i64, 1));
                let selected = b.cmp(
                    if level == 0 { C::Eq } else { C::Ne },
                    b.load(b.i8, at + offset_of!(CpuSystem, spsel)),
                    b.k(b.i8, 0),
                );
                let el1_selected = b.bin(b.boolean, B::BitAnd, el1, selected);
                if level == 0 {
                    b.bin(b.boolean, B::BitOr, at_level, el1_selected)
                } else {
                    el1_selected
                }
            } else {
                at_level
            };
            let p = b.sel(
                b.ptr,
                live,
                b.addr(b.reg_at(31)),
                b.addr(at + offset_of!(CpuSystem, sp_el) + level as usize * 8),
            );
            if a.read {
                b.put(a.rt, b.load_ptr(b.i64, p))
            } else {
                b.store_ptr(p, b.reg(b.i64, a.rt, true))
            }
            return;
        }
        if let Some((el1, el2)) = vhe_banks(a.register) {
            // HCR_EL2.E2H: EL1 register names accessed from EL2 reach the EL2 banks.
            let current = b.cv(b.i64, b.load(b.i8, at + offset_of!(CpuSystem, el)));
            let el2_mode = b.cmp(C::Eq, current, b.k(b.i64, 2));
            let e2h = b.cmp(
                C::Ne,
                b.imm(
                    b.i64,
                    B::BitAnd,
                    b.load(b.i64, at + offset_of!(CpuSystem, hcr_el2)),
                    super::cpu::HCR_E2H,
                ),
                b.k(b.i64, 0),
            );
            let redirect = b.bin(b.boolean, B::BitAnd, el2_mode, e2h);
            let p = b.sel(b.ptr, redirect, b.addr(at + el2), b.addr(at + el1));
            if a.read {
                b.put(a.rt, b.load_ptr(b.i64, p));
            } else {
                b.store_ptr(p, b.reg(b.i64, a.rt, true));
            }
            return;
        }
        match a.register {
            CurrentEl => b.put(
                a.rt,
                b.imm(
                    b.i64,
                    B::Shl,
                    b.cv(b.i64, b.load(b.i8, at + offset_of!(CpuSystem, el))),
                    2,
                ),
            ),
            Pan | PanSet => {
                // PSTATE.PAN lives in bit 22 of the register form.
                let pan = at + offset_of!(CpuSystem, pan);
                if a.read {
                    b.put(a.rt, b.imm(b.i64, B::Shl, b.load(b.i64, pan), 22));
                } else if a.register == PanSet {
                    b.store(pan, b.k(b.i64, u64::from(a.immediate & 1)));
                } else {
                    let bit = b.imm(b.i64, B::Shr, b.reg(b.i64, a.rt, true), 22);
                    b.store(pan, b.imm(b.i64, B::BitAnd, bit, 1));
                }
            }
            RazWi | AppleZero => {
                if a.read {
                    b.put(a.rt, b.k(b.i64, 0));
                }
            }
            SpSel | SpSelSet => {
                let selected = b.load(b.i8, at + offset_of!(CpuSystem, spsel));
                if a.read {
                    b.put(a.rt, b.cv(b.i64, selected));
                    return;
                }
                let new = if a.register == SpSelSet {
                    b.k(b.i8, u64::from(a.immediate & 1))
                } else {
                    b.cv(b.i8, b.imm(b.i64, B::BitAnd, b.reg(b.i64, a.rt, true), 1))
                };
                // The live stack pointer belongs to bank `SP_EL1` when SPSel is 1
                // and `SP_EL0` when it is 0. Park it in its bank, then load the
                // other one. Selecting the bank already in use is a store and a
                // reload of the same slot.
                let banks = at + offset_of!(CpuSystem, sp_el);
                let bank = |select: Value| {
                    b.sel(
                        b.ptr,
                        b.cmp(C::Ne, select, b.k(b.i8, 0)),
                        b.addr(banks + 8),
                        b.addr(banks),
                    )
                };
                let live = b.load(b.i64, b.reg_at(31));
                b.store_ptr(bank(selected), live);
                let incoming = b.load_ptr(b.i64, bank(new));
                b.store(b.reg_at(31), incoming);
                b.store(at + offset_of!(CpuSystem, spsel), new);
            }
            DaifSet | DaifClear => {
                let p = at + offset_of!(CpuSystem, daif);
                let before = b.load(b.i64, p);
                let updated = b.imm(
                    b.i64,
                    if a.register == DaifSet {
                        B::BitOr
                    } else {
                        B::BitAnd
                    },
                    before,
                    if a.register == DaifSet {
                        a.immediate as u64
                    } else {
                        (a.immediate as u64) ^ 0xf
                    },
                );
                b.store(p, updated);
            }
            Nzcv => {
                if a.read {
                    b.put(a.rt, b.cv(b.i64, b.load(b.i32, offset_of!(Cpu, flags))))
                } else {
                    b.store(
                        offset_of!(Cpu, flags),
                        b.imm(
                            b.i32,
                            B::BitAnd,
                            b.cv(b.i32, b.reg(b.i64, a.rt, true)),
                            0xf0000000,
                        ),
                    )
                }
            }
            OslsrEl1 => {
                let either = b.bin(
                    b.i64,
                    B::BitOr,
                    b.load(b.i64, at + offset_of!(CpuSystem, oslar_el1)),
                    b.load(b.i64, at + offset_of!(CpuSystem, osdlr_el1)),
                );
                let locked = b.cmp(C::Ne, either, b.k(b.i64, 0));
                b.put(a.rt, b.sel(b.i64, locked, b.k(b.i64, 2), b.k(b.i64, 0)));
            }
            CcsidrEl1 => {
                // CSSELR bit 0 is InD, bits 3:1 the level - 1; L2 has no separate I-side.
                use super::identification::{CCSIDR_L1D, CCSIDR_L1I, CCSIDR_L2};
                let select = b.imm(
                    b.i64,
                    B::BitAnd,
                    b.load(b.i64, at + offset_of!(CpuSystem, csselr_el1)),
                    0xf,
                );
                let is = |n: u64| b.cmp(C::Eq, select, b.k(b.i64, n));
                let l1_instruction =
                    b.sel(b.i64, is(1), b.k(b.i64, CCSIDR_L1I), b.k(b.i64, CCSIDR_L2));
                b.put(
                    a.rt,
                    b.sel(b.i64, is(0), b.k(b.i64, CCSIDR_L1D), l1_instruction),
                );
            }
            CntvTvalEl0 | CntpTvalEl0 => {
                // The timer value is the signed 32-bit distance to the compare value:
                // reads give the low 32 bits of `CVAL - count`, writes `CVAL = count + sext(value)`.
                // The physical timer shares the virtual count, as CNTVOFF_EL2 is zero without EL2.
                let counter = b.load(b.i64, at + offset_of!(CpuSystem, cntvct_el0));
                let compare = at
                    + if a.register == CntpTvalEl0 {
                        offset_of!(CpuSystem, cntp_cval_el0)
                    } else {
                        offset_of!(CpuSystem, cntv_cval_el0)
                    };
                if a.read {
                    let distance = b.bin(b.i64, B::Sub, b.load(b.i64, compare), counter);
                    b.put(a.rt, b.imm(b.i64, B::BitAnd, distance, 0xffff_ffff));
                } else {
                    let delta = b.extend(b.reg(b.i64, a.rt, true), Extend::Sxtw);
                    b.store(compare, b.bin(b.i64, B::Add, counter, delta));
                }
            }
            CntpctEl0 => {
                // Writes are ignored: the counter is not writable from software.
                if a.read {
                    b.put(a.rt, b.load(b.i64, at + offset_of!(CpuSystem, cntvct_el0)));
                }
            }
            IccIar1El1 => {
                // No group 1 interrupt is delivered by this model, so acknowledge reads spurious.
                if a.read {
                    b.put(a.rt, b.k(b.i64, 1023));
                }
            }
            IccEoir1El1 => {}
            Daif => {
                let p = at + offset_of!(CpuSystem, daif);
                if a.read {
                    b.put(a.rt, b.imm(b.i64, B::Shl, b.load(b.i64, p), 6))
                } else {
                    b.store(
                        p,
                        b.imm(
                            b.i64,
                            B::BitAnd,
                            b.imm(b.i64, B::Shr, b.reg(b.i64, a.rt, true), 6),
                            0xf,
                        ),
                    );
                }
            }
            _ => {
                if let Some(constant) = super::identification::value(a.register) {
                    b.put(a.rt, b.k(b.i64, constant));
                    return;
                }
                let offset =
                    system_offset(a.register).expect("decoded system register has storage");
                let p = at + offset;
                if a.read {
                    let v = if a.register == CntvCtlEl0 {
                        let counter = b.load(b.i64, at + offset_of!(CpuSystem, cntvct_el0));
                        let compare = b.load(b.i64, at + offset_of!(CpuSystem, cntv_cval_el0));
                        let reached = b.cmp(C::Uge, counter, compare);
                        b.bin(
                            b.i64,
                            B::BitOr,
                            b.load(b.i64, p),
                            b.sel(b.i64, reached, b.k(b.i64, 4), b.k(b.i64, 0)),
                        )
                    } else {
                        b.load(b.i64, p)
                    };
                    b.put(a.rt, v);
                } else {
                    let from = b.reg(b.i64, a.rt, true);
                    b.store(
                        p,
                        if a.register == CntvCtlEl0 {
                            b.imm(b.i64, B::BitAnd, from, 3)
                        } else {
                            from
                        },
                    );
                }
            }
        }
    }
}
fn system_offset(r: SystemRegister) -> Option<usize> {
    use SystemRegister::*;
    Some(match r {
        SpsrEl1 => offset_of!(CpuSystem, spsr_el1),
        ElrEl1 => offset_of!(CpuSystem, elr_el1),
        EsrEl1 => offset_of!(CpuSystem, esr_el1),
        FarEl1 => offset_of!(CpuSystem, far_el1),
        ParEl1 => offset_of!(CpuSystem, par_el1),
        IsrEl1 => offset_of!(CpuSystem, isr_el1),
        IccSreEl1 => offset_of!(CpuSystem, icc_sre_el1),
        IccPmrEl1 => offset_of!(CpuSystem, icc_pmr_el1),
        IccBpr0El1 => offset_of!(CpuSystem, icc_bpr0_el1),
        IccCtlrEl1 => offset_of!(CpuSystem, icc_ctlr_el1),
        IccIgrpen0El1 => offset_of!(CpuSystem, icc_igrpen0_el1),
        IccIgrpen1El1 => offset_of!(CpuSystem, icc_igrpen1_el1),
        IccBpr1El1 => offset_of!(CpuSystem, icc_bpr1_el1),
        CntpCtlEl0 => offset_of!(CpuSystem, cntp_ctl_el0),
        CntpCvalEl0 => offset_of!(CpuSystem, cntp_cval_el0),
        ActlrEl1 => offset_of!(CpuSystem, actlr_el1),
        VbarEl1 => offset_of!(CpuSystem, vbar_el1),
        SctlrEl2 => offset_of!(CpuSystem, sctlr_el2),
        TcrEl2 => offset_of!(CpuSystem, tcr_el2),
        Ttbr0El2 => offset_of!(CpuSystem, ttbr0_el2),
        Ttbr1El2 => offset_of!(CpuSystem, ttbr1_el2),
        MairEl2 => offset_of!(CpuSystem, mair_el2),
        VbarEl2 => offset_of!(CpuSystem, vbar_el2),
        ElrEl2 => offset_of!(CpuSystem, elr_el2),
        SpsrEl2 => offset_of!(CpuSystem, spsr_el2),
        EsrEl2 => offset_of!(CpuSystem, esr_el2),
        FarEl2 => offset_of!(CpuSystem, far_el2),
        HcrEl2 => offset_of!(CpuSystem, hcr_el2),
        TpidrEl2 => offset_of!(CpuSystem, tpidr_el2),
        CntvoffEl2 => offset_of!(CpuSystem, cntvoff_el2),
        CpacrEl1 => offset_of!(CpuSystem, cpacr_el1),
        SctlrEl1 => offset_of!(CpuSystem, sctlr_el1),
        Ttbr0El1 => offset_of!(CpuSystem, ttbr0_el1),
        Ttbr1El1 => offset_of!(CpuSystem, ttbr1_el1),
        TcrEl1 => offset_of!(CpuSystem, tcr_el1),
        MairEl1 => offset_of!(CpuSystem, mair_el1),
        TpidrEl0 => offset_of!(CpuSystem, tpidr_el0),
        TpidrroEl0 => offset_of!(CpuSystem, tpidrro_el0),
        CntvctEl0 => offset_of!(CpuSystem, cntvct_el0),
        CntvCtlEl0 => offset_of!(CpuSystem, cntv_ctl_el0),
        CntvCvalEl0 => offset_of!(CpuSystem, cntv_cval_el0),
        TpidrEl1 => offset_of!(CpuSystem, tpidr_el1),
        ContextidrEl1 => offset_of!(CpuSystem, contextidr_el1),
        CsselrEl1 => offset_of!(CpuSystem, csselr_el1),
        MdscrEl1 => offset_of!(CpuSystem, mdscr_el1),
        CntkctlEl1 => offset_of!(CpuSystem, cntkctl_el1),
        OslarEl1 => offset_of!(CpuSystem, oslar_el1),
        OsdlrEl1 => offset_of!(CpuSystem, osdlr_el1),
        Fpcr => offset_of!(CpuSystem, fpcr),
        Fpsr => offset_of!(CpuSystem, fpsr),
        Daif => offset_of!(CpuSystem, daif),
        Apple(slot) => offset_of!(CpuSystem, apple) + 8 * usize::from(slot),
        // After `system` in `Cpu`: relative to it, like every other register here.
        AppleBank(slot) => {
            offset_of!(Cpu, apple_bank) - offset_of!(Cpu, system) + 8 * usize::from(slot)
        }
        MidrEl1 => offset_of!(CpuSystem, midr_el1),
        IdAa64pfr0El1 => offset_of!(CpuSystem, id_aa64pfr0_el1),
        IdAa64mmfr0El1 => offset_of!(CpuSystem, id_aa64mmfr0_el1),
        PacKey(slot) => offset_of!(CpuSystem, pac_keys) + 8 * usize::from(slot),
        IdAa64isar1El1 => offset_of!(CpuSystem, id_aa64isar1_el1),
        _ => return None,
    })
}

/// EL1 registers that `HCR_EL2.E2H` redirects to their EL2 banks when accessed from EL2:
/// (EL1 field offset, EL2 field offset).
fn vhe_banks(r: SystemRegister) -> Option<(usize, usize)> {
    use SystemRegister::*;
    Some(match r {
        SctlrEl1 => (
            offset_of!(CpuSystem, sctlr_el1),
            offset_of!(CpuSystem, sctlr_el2),
        ),
        Ttbr0El1 => (
            offset_of!(CpuSystem, ttbr0_el1),
            offset_of!(CpuSystem, ttbr0_el2),
        ),
        Ttbr1El1 => (
            offset_of!(CpuSystem, ttbr1_el1),
            offset_of!(CpuSystem, ttbr1_el2),
        ),
        TcrEl1 => (
            offset_of!(CpuSystem, tcr_el1),
            offset_of!(CpuSystem, tcr_el2),
        ),
        MairEl1 => (
            offset_of!(CpuSystem, mair_el1),
            offset_of!(CpuSystem, mair_el2),
        ),
        VbarEl1 => (
            offset_of!(CpuSystem, vbar_el1),
            offset_of!(CpuSystem, vbar_el2),
        ),
        ElrEl1 => (
            offset_of!(CpuSystem, elr_el1),
            offset_of!(CpuSystem, elr_el2),
        ),
        SpsrEl1 => (
            offset_of!(CpuSystem, spsr_el1),
            offset_of!(CpuSystem, spsr_el2),
        ),
        EsrEl1 => (
            offset_of!(CpuSystem, esr_el1),
            offset_of!(CpuSystem, esr_el2),
        ),
        FarEl1 => (
            offset_of!(CpuSystem, far_el1),
            offset_of!(CpuSystem, far_el2),
        ),
        _ => return None,
    })
}
impl Environment for CpuEnvironment {
    fn x_offset(&self, r: u8) -> usize {
        offset_of!(Cpu, x) + r as usize * 8
    }
    fn sp_offset(&self) -> usize {
        offset_of!(Cpu, sp)
    }
    fn flags_offset(&self) -> usize {
        offset_of!(Cpu, flags)
    }
    fn v_offset(&self, r: u8) -> usize {
        offset_of!(Cpu, v) + r as usize * 16
    }
    fn begin_instruction(&mut self, _: &Builder) {
        self.pending += 1;
    }
    fn exit(&mut self, b: &Builder, next_pc: Value) {
        self.leave(b, next_pc)
    }
    fn memory(&mut self, b: &Builder, pc: u64, access: MemoryAccess<'_>) -> Result<(), LiftError> {
        let MemoryAccess {
            insn,
            address,
            writeback,
        } = access;
        match (*insn, address) {
            (Instruction::Memory(a), Some(address)) => self.scalar(b, pc, a, address, writeback),
            (Instruction::Literal(a), Some(address)) => self.literal(b, pc, a, address),
            (Instruction::Pair(a), Some(address)) => self.pair(b, pc, a, address, writeback),
            (Instruction::SimdLiteral(a), Some(address)) => {
                self.simd_trap_access(b, address, a.bytes as u64, a.rt, None, None, pc, Trap::Load)
            }
            (Instruction::SimdPair(a), Some(address)) => {
                self.simd_pair(b, pc, a, address, writeback)
            }
            (Instruction::SimdMemory(a), Some(address)) => {
                self.simd_memory(b, pc, a, address, writeback)
            }
            (Instruction::Exclusive(a), _) => self.exclusive(b, pc, a),
            (Instruction::Atomic(a), _) => self.atomic(b, pc, a),
            (Instruction::SimdStruct(a), _) => self.simd_struct(b, pc, a),
            _ => return Err(LiftError::Unsupported),
        }
        Ok(())
    }
    fn event(&mut self, b: &Builder, pc: u64, insn: &Instruction) -> Result<(), LiftError> {
        match *insn {
            Instruction::System(a) => self.lower_system(b, a),
            Instruction::Clrex(_) => b.byte(offset_of!(Cpu, monitor_valid), 0),
            Instruction::Sev | Instruction::Sevl => b.byte(offset_of!(Cpu, event_set), 1),
            // Cache maintenance by address has no effect on this machine.
            Instruction::CacheOp(_) => {}
            Instruction::AddressTranslate(a) => {
                b.store(offset_of!(Cpu, address), b.reg(b.i64, a.rt, true));
                b.store(
                    offset_of!(Cpu, value),
                    b.k(b.i64, u64::from(a.write) | u64::from(a.user) << 1),
                );
                self.trap(b, pc.wrapping_add(4), Trap::AddressTranslate);
            }
            Instruction::Wfe => self.trap(b, pc.wrapping_add(4), Trap::Wfe),
            Instruction::Brk(imm) => {
                b.store(offset_of!(Cpu, address), b.k(b.i64, u64::from(imm)));
                self.trap(b, pc, Trap::Brk)
            }
            Instruction::Isb(_) => self.trap(b, pc.wrapping_add(4), Trap::Isb),
            Instruction::Tlbi(_) => self.trap(b, pc.wrapping_add(4), Trap::Tlbi),
            Instruction::DcZva(rt) => {
                b.store(offset_of!(Cpu, address), b.reg(b.i64, rt, true));
                self.trap(b, pc.wrapping_add(4), Trap::DcZva)
            }
            Instruction::Eret => self.trap(b, pc.wrapping_add(4), Trap::Eret),
            Instruction::Svc(imm) => {
                b.store(offset_of!(Cpu, address), b.k(b.i64, u64::from(imm)));
                self.trap(b, pc.wrapping_add(4), Trap::Svc)
            }
            Instruction::Psci => self.trap(b, pc.wrapping_add(4), Trap::Psci),
            Instruction::Wfi => self.trap(b, pc.wrapping_add(4), Trap::Wfi),
            _ => return Err(LiftError::Unsupported),
        }
        Ok(())
    }
}
