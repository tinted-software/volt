use super::{
    cpu::{Cpu, DTLB_ENTRIES, Exclusive as CpuExclusive, System as CpuSystem, Trap},
    decode::*,
};
use crate::memory::MemoryError;
use core::{
    cell::{Cell, RefCell},
    fmt,
    mem::offset_of,
};
use volt_ir::{
    function::{self as ir, BinOp as B, CmpOp as C, Function, Opcode, Value},
    types::{IntDesc, Type, TypeKind},
};
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
        }
    }
}
impl core::error::Error for Error {}
impl From<MemoryError> for Error {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
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
struct Lower {
    f: RefCell<Function>,
    /// Block currently being emitted into. Memory fast paths switch it.
    block: Cell<ir::Block>,
    cpu: Value,
    u64: Type,
    u32: Type,
    u8: Type,
    boolean: Type,
    ptr: Type,
    /// Guest instructions started but not yet added to `cntvct_el0`. The
    /// counter is bumped once per exit path (`ret`) and before anything that
    /// reads it, instead of with a load/add/store per instruction.
    pending: Cell<u64>,
}
impl Lower {
    fn new() -> Self {
        let mut f = Function::new();
        // A typical block is a few blocks and about a hundred instructions.
        f.reserve(8, 128, 128);
        let u64 = f.types.intern(TypeKind::Int(IntDesc {
            signed: false,
            bits: 64,
        }));
        let u32 = f.types.intern(TypeKind::Int(IntDesc {
            signed: false,
            bits: 32,
        }));
        let u8 = f.types.intern(TypeKind::Int(IntDesc {
            signed: false,
            bits: 8,
        }));
        let boolean = f.types.intern(TypeKind::Bool);
        let ptr = f.types.ptr_global();
        let block = f.append_block();
        let cpu = f.append_block_param(block, ptr);
        Self {
            f: RefCell::new(f),
            block: Cell::new(block),
            cpu,
            u64,
            u32,
            u8,
            boolean,
            ptr,
            pending: Cell::new(0),
        }
    }
    fn emit(&self, t: Type, op: Opcode) -> Value {
        self.f.borrow_mut().append_inst(self.block.get(), t, op)
    }
    fn k(&self, t: Type, n: u64) -> Value {
        self.emit(t, Opcode::Iconst(n as i64))
    }
    fn bin(&self, t: Type, op: B, a: Value, b: Value) -> Value {
        self.emit(t, Opcode::Arith(ir::Arith { op, lhs: a, rhs: b }))
    }
    fn imm(&self, t: Type, op: B, a: Value, n: u64) -> Value {
        // `ArithImm` instead of `Arith` over an `Iconst`: the backend encodes
        // small immediates directly and `addrfold` only folds `ArithImm` adds
        // into load/store addressing.
        self.emit(
            t,
            Opcode::ArithImm(ir::ArithImm {
                op,
                lhs: a,
                imm: n as i64,
            }),
        )
    }
    fn cv(&self, t: Type, v: Value) -> Value {
        let same = { self.f.borrow().value_type(v) == t };
        if same {
            v
        } else {
            self.emit(t, Opcode::Convert(ir::Convert { value: v }))
        }
    }
    fn cmp(&self, op: C, a: Value, b: Value) -> Value {
        self.emit(
            self.boolean,
            Opcode::Icmp(ir::Compare { op, lhs: a, rhs: b }),
        )
    }
    fn sel(&self, t: Type, c: Value, a: Value, b: Value) -> Value {
        self.emit(
            t,
            Opcode::Select(ir::Select {
                cond: c,
                then: a,
                else_: b,
            }),
        )
    }
    fn addr(&self, at: usize) -> Value {
        self.imm(self.ptr, B::Add, self.cpu, at as u64)
    }
    fn load_ptr(&self, t: Type, p: Value) -> Value {
        self.emit(
            t,
            Opcode::Load(ir::Load {
                ptr: p,
                volatile: false,
            }),
        )
    }
    fn load(&self, t: Type, at: usize) -> Value {
        self.load_ptr(t, self.addr(at))
    }
    fn store_ptr(&self, p: Value, v: Value) {
        self.f.borrow_mut().append_store(self.block.get(), v, p)
    }
    fn store(&self, at: usize, v: Value) {
        self.store_ptr(self.addr(at), v)
    }
    fn byte(&self, at: usize, v: u64) {
        self.store(at, self.k(self.u8, v))
    }
    fn reg_at(r: u8) -> usize {
        if r == 31 {
            offset_of!(Cpu, sp)
        } else {
            offset_of!(Cpu, x) + r as usize * 8
        }
    }
    fn reg(&self, t: Type, r: u8, zero: bool) -> Value {
        if zero && r == 31 {
            self.k(t, 0)
        } else {
            self.cv(t, self.load(self.u64, Self::reg_at(r)))
        }
    }
    fn put(&self, r: u8, v: Value) {
        if r != 31 {
            self.put_sp(r, v)
        }
    }
    fn put_sp(&self, r: u8, v: Value) {
        self.store(Self::reg_at(r), self.cv(self.u64, v))
    }
    fn ty(&self, w: Width) -> Type {
        if w == Width::X64 { self.u64 } else { self.u32 }
    }
    fn signed(&self, w: Width) -> Type {
        self.f.borrow_mut().types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: w as u16,
        }))
    }
    fn add_counter(&self, n: u64) {
        let at = offset_of!(Cpu, system) + offset_of!(CpuSystem, cntvct_el0);
        self.store(at, self.imm(self.u64, B::Add, self.load(self.u64, at), n));
    }
    /// Make `cntvct_el0` in memory current on the path being emitted.
    fn flush_counter(&self) {
        let n = self.pending.replace(0);
        if n != 0 {
            self.add_counter(n);
        }
    }
    fn ret(&self, v: Value) {
        // Every exit leaves with the counter current. `pending` is not
        // cleared: sibling paths (slow blocks) share the main path's count.
        let n = self.pending.get();
        if n != 0 {
            self.add_counter(n);
        }
        self.f
            .borrow_mut()
            .set_terminator(self.block.get(), ir::Terminator::Ret(ir::Ret::one(v)))
    }
    fn not(&self, v: Value) -> Value {
        self.imm(self.boolean, B::BitXor, v, 1)
    }
    fn flag(&self, flags: Value, bit: u32) -> Value {
        self.cmp(
            C::Ne,
            self.imm(self.u32, B::BitAnd, flags, 1 << bit),
            self.k(self.u32, 0),
        )
    }
    fn condition(&self, c: Condition) -> Value {
        let f = self.load(self.u32, offset_of!(Cpu, flags));
        let n = self.flag(f, 31);
        let z = self.flag(f, 30);
        let carry = self.flag(f, 29);
        let v = self.flag(f, 28);
        let changed = self.bin(self.boolean, B::BitXor, n, v);
        let ordered = self.not(changed);
        let positive = self.not(z);
        match c {
            Condition::Eq => z,
            Condition::Ne => positive,
            Condition::Cs => carry,
            Condition::Cc => self.not(carry),
            Condition::Mi => n,
            Condition::Pl => self.not(n),
            Condition::Vs => v,
            Condition::Vc => self.not(v),
            Condition::Hi => self.bin(self.boolean, B::BitAnd, carry, positive),
            Condition::Ls => self.not(self.bin(self.boolean, B::BitAnd, carry, positive)),
            Condition::Ge => ordered,
            Condition::Lt => changed,
            Condition::Gt => self.bin(self.boolean, B::BitAnd, positive, ordered),
            Condition::Le => self.not(self.bin(self.boolean, B::BitAnd, positive, ordered)),
            Condition::Al | Condition::Nv => self.k(self.boolean, 1),
        }
    }
    fn pack(&self, n: Value, z: Value, c: Value, v: Value) -> Value {
        let mut out = self.k(self.u32, 0);
        for (bit, shift) in [(n, 31), (z, 30), (c, 29), (v, 28)] {
            out = self.bin(
                self.u32,
                B::BitOr,
                out,
                self.imm(self.u32, B::Shl, self.cv(self.u32, bit), shift),
            );
        }
        out
    }
    fn flags(
        &self,
        w: Width,
        op: ArithOp,
        a: Value,
        b: Value,
        r: Value,
        carry: Option<Value>,
    ) -> Value {
        let t = self.ty(w);
        let zero = self.k(t, 0);
        let n = self.cmp(
            C::Ne,
            self.imm(t, B::BitAnd, r, 1u64 << ((w as u32) - 1)),
            zero,
        );
        let z = self.cmp(C::Eq, r, zero);
        let c = carry.unwrap_or_else(|| {
            if op == ArithOp::Sub {
                self.cmp(C::Ge, a, b)
            } else {
                self.cmp(C::Lt, r, a)
            }
        });
        let s = self.signed(w);
        let sz = self.k(s, 0);
        let an = self.cmp(C::Lt, self.cv(s, a), sz);
        let bn = self.cmp(C::Lt, self.cv(s, b), sz);
        let rn = self.cmp(C::Lt, self.cv(s, r), sz);
        let diff = self.bin(self.boolean, B::BitXor, an, bn);
        let flip = self.bin(self.boolean, B::BitXor, an, rn);
        let v = self.bin(
            self.boolean,
            B::BitAnd,
            if op == ArithOp::Sub {
                diff
            } else {
                self.not(diff)
            },
            flip,
        );
        self.pack(n, z, c, v)
    }
    fn logic_flags(&self, w: Width, r: Value) {
        let t = self.ty(w);
        let zero = self.k(t, 0);
        let n = self.cmp(
            C::Ne,
            self.imm(t, B::BitAnd, r, 1u64 << ((w as u32) - 1)),
            zero,
        );
        let z = self.cmp(C::Eq, r, zero);
        let no = self.k(self.boolean, 0);
        self.store(offset_of!(Cpu, flags), self.pack(n, z, no, no));
    }
    fn shift(&self, w: Width, v: Value, s: Shift, n: u64) -> Value {
        if n == 0 {
            return v;
        }
        let t = self.ty(w);
        match s {
            Shift::Lsl => self.imm(t, B::Shl, v, n),
            Shift::Lsr => self.imm(t, B::Shr, v, n),
            Shift::Asr => {
                let signed = self.signed(w);
                self.cv(t, self.imm(signed, B::Shr, self.cv(signed, v), n))
            }
            Shift::Ror => self.bin(
                t,
                B::BitOr,
                self.imm(t, B::Shr, v, n),
                self.imm(t, B::Shl, v, w as u64 - n),
            ),
        }
    }
    fn extend(&self, v: Value, e: Extend) -> Value {
        match e {
            Extend::Uxtx | Extend::Sxtx => v,
            Extend::Uxtb => self.imm(self.u64, B::BitAnd, v, 0xff),
            Extend::Uxth => self.imm(self.u64, B::BitAnd, v, 0xffff),
            Extend::Uxtw => self.imm(self.u64, B::BitAnd, v, 0xffffffff),
            Extend::Sxtb | Extend::Sxth | Extend::Sxtw => {
                let n = match e {
                    Extend::Sxtb => 56,
                    Extend::Sxth => 48,
                    _ => 32,
                };
                let raised = self.imm(self.u64, B::Shl, v, n);
                self.shift(Width::X64, raised, Shift::Asr, n)
            }
        }
    }
    fn address(&self, rn: u8, a: Addressing) -> Value {
        let base = self.reg(self.u64, rn, false);
        match a {
            Addressing::Offset(n) => self.imm(self.u64, B::Add, base, n as u64),
            Addressing::PreIndex(n) => {
                let a = self.imm(self.u64, B::Add, base, n as u64);
                self.put_sp(rn, a);
                a
            }
            Addressing::PostIndex(n) => {
                self.store(
                    offset_of!(Cpu, writeback_value),
                    self.imm(self.u64, B::Add, base, n as u64),
                );
                self.byte(offset_of!(Cpu, writeback_dest), rn as u64);
                self.byte(offset_of!(Cpu, writeback), 1);
                base
            }
            Addressing::Register { rm, extend, amount } => {
                let v = self.extend(self.reg(self.u64, rm, true), extend);
                self.bin(
                    self.u64,
                    B::Add,
                    base,
                    self.imm(self.u64, B::Shl, v, amount as u64),
                )
            }
        }
    }
    fn branch(&self, pc: u64, offset: i64, c: Value) {
        self.ret(self.sel(
            self.u64,
            c,
            self.k(self.u64, pc.wrapping_add(offset as u64)),
            self.k(self.u64, pc.wrapping_add(4)),
        ))
    }
    fn uint(&self, bits: u16) -> Type {
        self.f.borrow_mut().types.intern(TypeKind::Int(IntDesc {
            signed: false,
            bits,
        }))
    }
    fn new_block(&self) -> ir::Block {
        self.f.borrow_mut().append_block()
    }
    fn branch_if(&self, c: Value, then: ir::Block, otherwise: ir::Block) {
        self.f.borrow_mut().append_if(
            self.block.get(),
            c,
            ir::EdgeDesc::bare(then),
            ir::EdgeDesc::bare(otherwise),
        )
    }
    /// Like `address`, but a base-register writeback (pre- or post-index) is
    /// returned instead of applied or scheduled for the machine, so each path
    /// can apply it its own way once the access has succeeded. An access that
    /// faults must leave the base register untouched so the retry sees it.
    fn address_split(&self, rn: u8, a: Addressing) -> (Value, Option<(u8, Value)>) {
        match a {
            Addressing::PostIndex(n) => {
                let base = self.reg(self.u64, rn, false);
                (base, Some((rn, self.imm(self.u64, B::Add, base, n as u64))))
            }
            Addressing::PreIndex(n) => {
                let base = self.reg(self.u64, rn, false);
                let address = self.imm(self.u64, B::Add, base, n as u64);
                (address, Some((rn, address)))
            }
            other => (self.address(rn, other), None),
        }
    }
    /// Probe the inline data TLB for `address`. Returns `(hit, host_pointer)`;
    /// the pointer is only meaningful when `hit`. A hit requires the page to be
    /// cached for this access kind and the access to be naturally aligned (so it
    /// cannot cross a page), because the tag compare covers the low `size - 1`
    /// address bits too.
    fn dtlb_probe(&self, address: Value, size: Size, write: bool) -> (Value, Value) {
        let n = size as u64;
        // Entry offset = ((address >> 12) & (ENTRIES - 1)) * 32 = (address >> 7) & mask.
        let index = self.imm(
            self.u64,
            B::BitAnd,
            self.imm(self.u64, B::Shr, address, 7),
            (DTLB_ENTRIES as u64 - 1) << 5,
        );
        let entry = self.bin(self.ptr, B::Add, self.cpu, index);
        // Constant displacements fold into the load addressing.
        let field = |at: usize| {
            self.load_ptr(
                self.u64,
                self.imm(self.ptr, B::Add, entry, (offset_of!(Cpu, dtlb) + at) as u64),
            )
        };
        let tag = field(if write { 8 } else { 0 });
        let addend = field(16);
        let host = self.bin(self.ptr, B::Add, address, addend);
        let want = self.imm(self.u64, B::BitAnd, address, !0xfffu64 | (n - 1));
        // Last, so the backend can fuse it into the branch that consumes it.
        let hit = self.cmp(C::Eq, tag, want);
        (hit, host)
    }
    fn fast_load(&self, host: Value, size: Size, rt: u8, signed: SignExtend) {
        let bits = size as u16 * 8;
        let raw = self.load_ptr(self.uint(bits), host);
        if rt == 31 {
            return;
        }
        let mut v = self.cv(self.u64, raw);
        if signed != SignExtend::None && bits < 64 {
            let n = 64 - bits as u64;
            let extended = self.shift(Width::X64, self.imm(self.u64, B::Shl, v, n), Shift::Asr, n);
            v = if signed == SignExtend::To32 {
                self.imm(self.u64, B::BitAnd, extended, 0xffff_ffff)
            } else {
                extended
            };
        }
        self.put(rt, v);
    }
    fn fast_store(&self, host: Value, size: Size, rt: u8) {
        let t = self.uint(size as u16 * 8);
        self.store_ptr(host, self.cv(t, self.reg(self.u64, rt, true)));
    }
    /// Emit a load or store with an inline fast path. On a data-TLB hit the
    /// access happens in place and emission continues in a fresh block; on a
    /// miss control goes to a slow block that hands the access to the machine
    /// (`slow` fills in the request) and returns.
    fn memory_access(
        &self,
        address: Value,
        size: Size,
        op: MemoryOp,
        rt: u8,
        signed: SignExtend,
        post: Option<(u8, Value)>,
        slow: impl FnOnce(&Lower),
    ) {
        let write = op == MemoryOp::Store;
        let (hit, host) = self.dtlb_probe(address, size, write);
        let (fast_block, slow_block) = (self.new_block(), self.new_block());
        self.branch_if(hit, fast_block, slow_block);
        self.block.set(slow_block);
        slow(self);
        self.block.set(fast_block);
        if write {
            self.fast_store(host, size, rt);
        } else {
            self.fast_load(host, size, rt, signed);
        }
        if let Some((rn, value)) = post {
            self.put_sp(rn, value);
        }
    }
    /// Inline fast path for a GPR load/store pair: one data-TLB probe covers
    /// both accesses when the first is naturally aligned and the second starts
    /// in the same page. Anything else takes `slow`, which hands the whole pair
    /// to the machine.
    fn pair_access(
        &self,
        address: Value,
        a: Pair,
        post: Option<(u8, Value)>,
        slow: impl FnOnce(&Lower),
    ) {
        let write = a.op == MemoryOp::Store;
        let n = a.size as u64;
        let (hit, host) = self.dtlb_probe(address, a.size, write);
        let (check_block, fast_block, slow_block) =
            (self.new_block(), self.new_block(), self.new_block());
        self.branch_if(hit, check_block, slow_block);
        // `address` is aligned to `n`, so the second access leaves the page
        // exactly when it starts on a page boundary.
        self.block.set(check_block);
        let second_at = self.imm(self.u64, B::Add, address, n);
        let offset = self.imm(self.u64, B::BitAnd, second_at, 0xfff);
        let same_page = self.cmp(C::Ne, offset, self.k(self.u64, 0));
        self.branch_if(same_page, fast_block, slow_block);
        self.block.set(slow_block);
        slow(self);
        self.block.set(fast_block);
        let second = self.imm(self.ptr, B::Add, host, n);
        if write {
            self.fast_store(host, a.size, a.rt);
            self.fast_store(second, a.size, a.rt2);
        } else {
            self.fast_load(host, a.size, a.rt, a.signed);
            self.fast_load(second, a.size, a.rt2, a.signed);
        }
        if let Some((rn, value)) = post {
            self.put_sp(rn, value);
        }
    }
    fn trap(&self, pc: u64, t: Trap) {
        self.byte(offset_of!(Cpu, trap), t as u64);
        self.ret(self.k(self.u64, pc))
    }

    /// Slow path for a whole SIMD vector or vector-pair access: describe the
    /// full instruction in the CPU request fields (`vector_dest`, optionally
    /// `second_*`) and trap out; the machine completes every byte and resumes
    /// after the instruction.
    fn simd_trap_access(
        &self,
        address: Value,
        bytes: u64,
        rt: u8,
        rt2: Option<u8>,
        post: Option<(u8, Value)>,
        pc: u64,
        trap: Trap,
    ) {
        self.store(offset_of!(Cpu, address), address);
        self.byte(offset_of!(Cpu, width), bytes);
        self.byte(offset_of!(Cpu, dest), rt as u64);
        self.byte(offset_of!(Cpu, vector_dest), 1);
        if let Some(rt2) = rt2 {
            let second = self.imm(self.u64, B::Add, address, if bytes == 16 { 16 } else { 8 });
            self.store(offset_of!(Cpu, second_address), second);
            self.byte(offset_of!(Cpu, second_width), bytes);
            self.byte(offset_of!(Cpu, second_dest), rt2 as u64);
            self.byte(offset_of!(Cpu, second_vector), 1);
            self.byte(offset_of!(Cpu, second_pending), 1);
        }
        if let Some((rn, value)) = post {
            self.store(offset_of!(Cpu, writeback_value), value);
            self.byte(offset_of!(Cpu, writeback_dest), rn as u64);
            self.byte(offset_of!(Cpu, writeback), 1);
        }
        self.trap(pc.wrapping_add(4), trap);
    }
}
fn arithmetic_op(op: ArithOp) -> B {
    if op == ArithOp::Sub { B::Sub } else { B::Add }
}
fn logic_op(op: LogicOp) -> B {
    match op {
        LogicOp::BitAnd => B::BitAnd,
        LogicOp::BitOr => B::BitOr,
        LogicOp::BitXor => B::BitXor,
    }
}
fn low_mask(n: u8) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

pub fn compile(guest_pc: u64, bytes: &[u8]) -> Result<Block, Error> {
    compile_with(guest_pc, bytes, false)
}

/// Compile a block. With `inline_memory`, plain loads and stores get an inline
/// data-TLB fast path and do not end the block; the caller's block scan must
/// use `Instruction::terminates_with(true)` to match.
pub fn compile_with(guest_pc: u64, bytes: &[u8], inline_memory: bool) -> Result<Block, Error> {
    if bytes.is_empty() {
        return Err(Error::EmptyBlock);
    }
    if bytes.len() % 4 != 0 {
        return Err(Error::UnsupportedInstruction);
    }
    let l = Lower::new();
    let mut pc = guest_pc;
    let mut ended = false;
    for chunk in bytes.chunks_exact(4) {
        l.pending.set(l.pending.get() + 1);
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        let instruction = match decode(word) {
            Ok(insn) => insn,
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
        };
        ended = instruction.terminates_with(inline_memory);
        match instruction {
            Instruction::Movz(a) | Instruction::Movn(a) => {
                let t = l.ty(a.width);
                let bits = (a.immediate as u64) << a.shift;
                let value = if matches!(instruction, Instruction::Movn(_)) {
                    !bits
                } else {
                    bits
                };
                if a.rd != 31 {
                    l.put(a.rd, l.k(t, value));
                }
            }
            Instruction::Movk(a) => {
                if a.rd != 31 {
                    let t = l.ty(a.width);
                    let old = l.reg(t, a.rd, false);
                    let kept = l.imm(t, B::BitAnd, old, !(0xffffu64 << a.shift));
                    l.put(
                        a.rd,
                        l.imm(t, B::BitOr, kept, (a.immediate as u64) << a.shift),
                    );
                }
            }
            Instruction::ArithImm(a) => {
                let t = l.ty(a.width);
                let lhs = l.reg(t, a.rn, false);
                let rhs = l.k(t, a.immediate as u64);
                let r = l.bin(t, arithmetic_op(a.op), lhs, rhs);
                if a.flags {
                    l.put(a.rd, r);
                    l.store(
                        offset_of!(Cpu, flags),
                        l.flags(a.width, a.op, lhs, rhs, r, None),
                    );
                } else {
                    l.put_sp(a.rd, r)
                }
            }
            Instruction::ArithReg(a) => {
                let t = l.ty(a.width);
                let lhs = l.reg(t, a.rn, true);
                let rhs = l.shift(a.width, l.reg(t, a.rm, true), a.shift, a.amount as u64);
                let r = l.bin(t, arithmetic_op(a.op), lhs, rhs);
                if a.rd != 31 {
                    l.put(a.rd, r)
                }
                if a.flags {
                    l.store(
                        offset_of!(Cpu, flags),
                        l.flags(a.width, a.op, lhs, rhs, r, None),
                    );
                }
            }
            Instruction::ArithExt(a) => {
                let t = l.ty(a.width);
                let lhs = l.reg(t, a.rn, false);
                let extended = l.extend(l.reg(l.u64, a.rm, true), a.extend);
                let rhs = l.cv(t, l.imm(l.u64, B::Shl, extended, a.amount as u64));
                let r = l.bin(t, arithmetic_op(a.op), lhs, rhs);
                if a.flags {
                    l.put(a.rd, r);
                    l.store(
                        offset_of!(Cpu, flags),
                        l.flags(a.width, a.op, lhs, rhs, r, None),
                    );
                } else {
                    l.put_sp(a.rd, r)
                }
            }
            Instruction::LogicReg(a) => {
                let t = l.ty(a.width);
                let lhs = l.reg(t, a.rn, true);
                let shifted = l.shift(a.width, l.reg(t, a.rm, true), a.shift, a.amount as u64);
                let rhs = if a.invert {
                    l.imm(t, B::BitXor, shifted, u64::MAX)
                } else {
                    shifted
                };
                let r = l.bin(t, logic_op(a.op), lhs, rhs);
                l.put(a.rd, r);
                if a.flags {
                    l.logic_flags(a.width, r)
                }
            }
            Instruction::LogicImm(a) => {
                let t = l.ty(a.width);
                let r = l.imm(t, logic_op(a.op), l.reg(t, a.rn, true), a.immediate);
                if a.flags {
                    l.put(a.rd, r);
                    l.logic_flags(a.width, r)
                } else {
                    l.put_sp(a.rd, r)
                }
            }
            Instruction::ArithCarry(a) => {
                let t = l.ty(a.width);
                let lhs = l.reg(t, a.rn, true);
                let loaded = l.reg(t, a.rm, true);
                let rhs = if a.op == ArithOp::Sub {
                    l.imm(t, B::BitXor, loaded, u64::MAX)
                } else {
                    loaded
                };
                let carry = l.cv(
                    t,
                    l.imm(
                        l.u32,
                        B::BitAnd,
                        l.imm(l.u32, B::Shr, l.load(l.u32, offset_of!(Cpu, flags)), 29),
                        1,
                    ),
                );
                let partial = l.bin(t, B::Add, lhs, rhs);
                let r = l.bin(t, B::Add, partial, carry);
                if a.rd != 31 {
                    l.put(a.rd, r)
                }
                if a.flags {
                    let out = l.bin(
                        l.boolean,
                        B::BitOr,
                        l.cmp(C::Lt, partial, lhs),
                        l.cmp(C::Lt, r, partial),
                    );
                    l.store(
                        offset_of!(Cpu, flags),
                        l.flags(a.width, ArithOp::Add, lhs, rhs, r, Some(out)),
                    );
                }
            }
            Instruction::CondCompare(a) => {
                let t = l.ty(a.width);
                let lhs = l.reg(t, a.rn, true);
                let rhs = match a.operand {
                    CompareOperand::Register(r) => l.reg(t, r, true),
                    CompareOperand::Immediate(n) => l.k(t, n as u64),
                };
                let r = l.bin(t, arithmetic_op(a.op), lhs, rhs);
                let flags = l.flags(a.width, a.op, lhs, rhs, r, None);
                l.store(
                    offset_of!(Cpu, flags),
                    l.sel(
                        l.u32,
                        l.condition(a.cond),
                        flags,
                        l.k(l.u32, (a.nzcv as u64) << 28),
                    ),
                );
            }
            Instruction::Variable(a) => lower_variable(&l, a),
            Instruction::Bitfield(a) => lower_bitfield(&l, a),
            Instruction::Unary(a) => lower_unary(&l, a),
            Instruction::Extract(a) => {
                let t = l.ty(a.width);
                let low = l.reg(t, a.rm, true);
                let r = if a.lsb == 0 {
                    low
                } else {
                    l.bin(
                        t,
                        B::BitOr,
                        l.imm(t, B::Shr, low, a.lsb as u64),
                        l.imm(
                            t,
                            B::Shl,
                            l.reg(t, a.rn, true),
                            a.width as u64 - a.lsb as u64,
                        ),
                    )
                };
                if a.rd != 31 {
                    l.put(a.rd, r)
                }
            }
            Instruction::Mul(a) => {
                let t = l.ty(a.width);
                let p = l.bin(t, B::Mul, l.reg(t, a.rn, true), l.reg(t, a.rm, true));
                let r = l.bin(t, arithmetic_op(a.op), l.reg(t, a.ra, true), p);
                if a.rd != 31 {
                    l.put(a.rd, r)
                }
            }
            Instruction::MulLong(a) => {
                let e = if a.signed { Extend::Sxtw } else { Extend::Uxtw };
                let p = l.bin(
                    l.u64,
                    B::Mul,
                    l.extend(l.reg(l.u64, a.rn, true), e),
                    l.extend(l.reg(l.u64, a.rm, true), e),
                );
                let r = l.bin(l.u64, arithmetic_op(a.op), l.reg(l.u64, a.ra, true), p);
                if a.rd != 31 {
                    l.put(a.rd, r)
                }
            }
            Instruction::MulHigh(a) => lower_high(&l, a),
            Instruction::Adr(a) => {
                let here = if a.page { pc & 0xfffffffffffff000 } else { pc };
                l.put(a.rd, l.k(l.u64, here.wrapping_add(a.offset as u64)));
            }
            Instruction::Csel(a) => {
                let t = l.ty(a.width);
                let lhs = l.reg(t, a.rn, true);
                let rhs = l.reg(t, a.rm, true);
                let alt = match a.opc {
                    0 => rhs,
                    1 => l.imm(t, B::Add, rhs, 1),
                    2 => l.imm(t, B::BitXor, rhs, u64::MAX),
                    3 => l.bin(t, B::Sub, l.k(t, 0), rhs),
                    _ => unreachable!(),
                };
                if a.rd != 31 {
                    l.put(a.rd, l.sel(t, l.condition(a.cond), lhs, alt));
                }
            }
            Instruction::TestBranch(a) => {
                let t = l.ty(a.width);
                let v = l.reg(t, a.rt, true);
                let subject = if let Some(bit) = a.bit_index {
                    l.imm(t, B::BitAnd, v, 1u64 << bit)
                } else {
                    v
                };
                l.branch(
                    pc,
                    a.offset,
                    l.cmp(if a.negate { C::Ne } else { C::Eq }, subject, l.k(t, 0)),
                );
            }
            Instruction::BCond(a) => l.branch(pc, a.offset, l.condition(a.cond)),
            Instruction::B(offset) => l.ret(l.k(l.u64, pc.wrapping_add(offset as u64))),
            Instruction::Call(a) => {
                if a.link {
                    l.put(30, l.k(l.u64, pc.wrapping_add(4)))
                }
                l.ret(l.k(l.u64, pc.wrapping_add(a.target as u64)));
            }
            Instruction::Indirect(a) => {
                let target = l.reg(l.u64, a.rn, false);
                if a.link {
                    l.put(30, l.k(l.u64, pc.wrapping_add(4)))
                }
                l.ret(target);
            }
            Instruction::System(a) => lower_system(&l, a),
            Instruction::Memory(a) => {
                let trap = if a.op == MemoryOp::Store {
                    Trap::Store
                } else {
                    Trap::Load
                };
                if inline_memory {
                    let (address, post) = l.address_split(a.rn, a.addressing);
                    l.memory_access(address, a.size, a.op, a.rt, a.signed, post, |l| {
                        setup_memory(l, address, a.size, a.rt, a.signed, a.op);
                        if let Some((rn, value)) = post {
                            l.store(offset_of!(Cpu, writeback_value), value);
                            l.byte(offset_of!(Cpu, writeback_dest), rn as u64);
                            l.byte(offset_of!(Cpu, writeback), 1);
                        }
                        l.trap(pc.wrapping_add(4), trap);
                    });
                } else {
                    let address = l.address(a.rn, a.addressing);
                    setup_memory(&l, address, a.size, a.rt, a.signed, a.op);
                    l.trap(pc.wrapping_add(4), trap);
                }
            }
            Instruction::Literal(a) => {
                let address = l.k(l.u64, pc.wrapping_add(a.offset as u64));
                if inline_memory {
                    l.memory_access(
                        address,
                        a.size,
                        MemoryOp::Load,
                        a.rt,
                        SignExtend::None,
                        None,
                        |l| {
                            setup_memory(
                                l,
                                address,
                                a.size,
                                a.rt,
                                SignExtend::None,
                                MemoryOp::Load,
                            );
                            l.trap(pc.wrapping_add(4), Trap::Load);
                        },
                    );
                } else {
                    setup_memory(&l, address, a.size, a.rt, SignExtend::None, MemoryOp::Load);
                    l.trap(pc.wrapping_add(4), Trap::Load);
                }
            }
            Instruction::SimdLiteral(a) => {
                let address = l.k(l.u64, pc.wrapping_add(a.offset as u64));
                l.simd_trap_access(address, a.bytes as u64, a.rt, None, None, pc, Trap::Load);
            }
            Instruction::Exclusive(a) => {
                setup_memory(
                    &l,
                    l.reg(l.u64, a.rn, false),
                    a.size,
                    a.rt,
                    SignExtend::None,
                    a.op,
                );
                if a.op == MemoryOp::Store {
                    l.byte(offset_of!(Cpu, status_dest), a.rs as u64)
                }
                l.byte(
                    offset_of!(Cpu, exclusive),
                    if a.op == MemoryOp::Store {
                        CpuExclusive::Store as u64
                    } else {
                        CpuExclusive::Load as u64
                    },
                );
                l.trap(
                    pc.wrapping_add(4),
                    if a.op == MemoryOp::Store {
                        Trap::Store
                    } else {
                        Trap::Load
                    },
                );
            }
            Instruction::Atomic(a) => {
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
                l.store(offset_of!(Cpu, address), l.reg(l.u64, a.rn, false));
                l.byte(offset_of!(Cpu, width), a.size as u64);
                l.byte(offset_of!(Cpu, dest), dests[0] as u64);
                l.byte(offset_of!(Cpu, atomic_kind), a.kind as u64);
                for (slot, &dest) in dests.iter().enumerate() {
                    l.byte(offset_of!(Cpu, atomic_dests) + slot, dest as u64);
                }
                for (slot, &reg) in operands.iter().enumerate() {
                    // Register numbers past 30 read as XZR.
                    l.store(
                        offset_of!(Cpu, atomic_operands) + slot * 8,
                        l.reg(l.u64, reg, true),
                    );
                }
                if a.kind == AtomicKind::StorePair {
                    l.byte(offset_of!(Cpu, status_dest), a.rs as u64);
                }
                l.trap(pc.wrapping_add(4), Trap::Atomic);
            }
            Instruction::SimdStruct(a) => {
                use super::simd_struct::StructDesc;
                // Element-wise accesses with lane merging need the machine slow path.
                let d = a.desc;
                let address = l.reg(l.u64, a.rn, false);
                l.store(offset_of!(Cpu, address), address);
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
                    l.byte(offset_of!(Cpu, structure) + field, value);
                }
                let step = match a.post {
                    StructPost::None => None,
                    StructPost::Immediate => Some(l.k(l.u64, d.bytes())),
                    StructPost::Register(rm) => Some(l.reg(l.u64, rm, true)),
                };
                if let Some(step) = step {
                    l.store(
                        offset_of!(Cpu, writeback_value),
                        l.bin(l.u64, B::Add, address, step),
                    );
                    l.byte(offset_of!(Cpu, writeback_dest), a.rn as u64);
                    l.byte(offset_of!(Cpu, writeback), 1);
                }
                l.trap(pc.wrapping_add(4), Trap::SimdStruct);
            }
            Instruction::AddressTranslate(a) => {
                l.store(offset_of!(Cpu, address), l.reg(l.u64, a.rt, true));
                l.store(
                    offset_of!(Cpu, value),
                    l.k(l.u64, u64::from(a.write) | u64::from(a.user) << 1),
                );
                l.trap(pc.wrapping_add(4), Trap::AddressTranslate);
            }
            Instruction::Host(word) => {
                // The host runs this word: see `host::run`.
                l.store(offset_of!(Cpu, value), l.k(l.u64, word as u64));
                l.trap(pc.wrapping_add(4), Trap::Host);
            }
            Instruction::Pair(a) => {
                let trap = if a.op == MemoryOp::Store {
                    Trap::Store
                } else {
                    Trap::Load
                };
                if inline_memory {
                    let (address, post) = l.address_split(a.rn, a.addressing);
                    l.pair_access(address, a, post, |l| {
                        setup_pair(l, address, a);
                        if let Some((rn, value)) = post {
                            l.store(offset_of!(Cpu, writeback_value), value);
                            l.byte(offset_of!(Cpu, writeback_dest), rn as u64);
                            l.byte(offset_of!(Cpu, writeback), 1);
                        }
                        l.trap(pc.wrapping_add(4), trap);
                    });
                } else {
                    let address = l.address(a.rn, a.addressing);
                    setup_pair(&l, address, a);
                    l.trap(pc.wrapping_add(4), trap);
                }
            }
            Instruction::Clrex => l.byte(offset_of!(Cpu, monitor_valid), 0),
            Instruction::Sev | Instruction::Sevl => l.byte(offset_of!(Cpu, event_set), 1),
            Instruction::SimdImm(a) => {
                let at = offset_of!(Cpu, v) + a.rd as usize * 16;
                match a.combine {
                    ImmCombine::Set => {
                        l.store(at, l.k(l.u64, a.low));
                        l.store(at + 8, l.k(l.u64, a.high));
                    }
                    ImmCombine::Or | ImmCombine::AndNot => {
                        // A 64-bit form (`q == false`) also clears the upper half.
                        let (op, imm) = match a.combine {
                            ImmCombine::Or => (B::BitOr, a.low),
                            _ => (B::BitAnd, !a.low),
                        };
                        l.store(at, l.imm(l.u64, op, l.load(l.u64, at), imm));
                        l.store(
                            at + 8,
                            if a.q {
                                l.imm(l.u64, op, l.load(l.u64, at + 8), imm)
                            } else {
                                l.k(l.u64, 0)
                            },
                        );
                    }
                }
            }
            Instruction::SimdMove { rd, rn } => {
                let at_d = offset_of!(Cpu, v) + rd as usize * 16;
                let at_n = offset_of!(Cpu, v) + rn as usize * 16;
                let lo = l.load(l.u64, at_n);
                let hi = l.load(l.u64, at_n + 8);
                l.store(at_d, lo);
                l.store(at_d + 8, hi);
            }
            Instruction::SimdInsert(a) => {
                // Replace one lane of the destination with the low bits of a GPR.
                let ebits = (1u64 << a.esize) * 8;
                let emask = if ebits == 64 {
                    u64::MAX
                } else {
                    (1u64 << ebits) - 1
                };
                let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                let per_dw = (64 / ebits) as usize;
                let word = a.index as usize / per_dw;
                let shift = (a.index as usize % per_dw) as u64 * ebits;
                let old = l.load(l.u64, at_d + word * 8);
                let cleared = l.bin(l.u64, B::BitAnd, old, l.k(l.u64, !(emask << shift)));
                let value = l.bin(
                    l.u64,
                    B::BitAnd,
                    l.reg(l.u64, a.rn, true),
                    l.k(l.u64, emask),
                );
                l.store(
                    at_d + word * 8,
                    l.bin(l.u64, B::BitOr, cleared, l.imm(l.u64, B::Shl, value, shift)),
                );
            }
            Instruction::SimdInsertElement(a) => {
                // Copy one lane of Vn over one lane of Vd. The source is read before
                // the store, so `rd == rn` is fine.
                let ebits = (1u64 << a.esize) * 8;
                let emask = if ebits == 64 {
                    u64::MAX
                } else {
                    (1u64 << ebits) - 1
                };
                let per_dw = (64 / ebits) as usize;
                let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                let (src_word, src_shift) = (
                    a.src as usize / per_dw,
                    (a.src as usize % per_dw) as u64 * ebits,
                );
                let (dst_word, dst_shift) = (
                    a.dst as usize / per_dw,
                    (a.dst as usize % per_dw) as u64 * ebits,
                );
                let source = l.load(l.u64, at_n + src_word * 8);
                let value = l.imm(
                    l.u64,
                    B::BitAnd,
                    l.imm(l.u64, B::Shr, source, src_shift),
                    emask,
                );
                let old = l.load(l.u64, at_d + dst_word * 8);
                let cleared = l.imm(l.u64, B::BitAnd, old, !(emask << dst_shift));
                l.store(
                    at_d + dst_word * 8,
                    l.bin(
                        l.u64,
                        B::BitOr,
                        cleared,
                        l.imm(l.u64, B::Shl, value, dst_shift),
                    ),
                );
            }
            Instruction::SimdExtract(a) => {
                // Move one lane into the low bits of a GPR, zero-extended.
                let ebits = (1u64 << a.esize) * 8;
                let emask = if ebits == 64 {
                    u64::MAX
                } else {
                    (1u64 << ebits) - 1
                };
                let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                let per_dw = (64 / ebits) as usize;
                let word = a.index as usize / per_dw;
                let shift = (a.index as usize % per_dw) as u64 * ebits;
                let source = l.load(l.u64, at_n + word * 8);
                let value = l.imm(l.u64, B::BitAnd, l.imm(l.u64, B::Shr, source, shift), emask);
                l.put(a.rd, value);
            }
            Instruction::SimdDupElement(a) => {
                let ebits = (1u64 << a.esize) * 8;
                let emask = if ebits == 64 {
                    u64::MAX
                } else {
                    (1u64 << ebits) - 1
                };
                let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                let vn_lo = l.load(l.u64, at_n);
                let vn_hi = l.load(l.u64, at_n + 8);
                // Locate the requested element across the 128-bit source.
                let lane = a.index as usize;
                let lanes_per_dw = (64 / ebits) as usize;
                let src = if lane / lanes_per_dw == 0 {
                    vn_lo
                } else {
                    vn_hi
                };
                let shift = (lane % lanes_per_dw) as u64 * ebits;
                let element = l.imm(l.u64, B::BitAnd, l.imm(l.u64, B::Shr, src, shift), emask);
                // Replicate the element across the destination lanes.
                let mut acc = l.k(l.u64, 0);
                let lanes = (64 / ebits) as usize;
                for lane in 0..lanes {
                    acc = l.bin(
                        l.u64,
                        B::BitOr,
                        acc,
                        l.imm(l.u64, B::Shl, element, lane as u64 * ebits),
                    );
                }
                let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                l.store(at_d, acc);
                l.store(at_d + 8, if a.q { acc } else { l.k(l.u64, 0) });
            }
            Instruction::SimdDup { rd, rn, esize, q } => {
                let r = l.reg(l.u64, rn, true);
                let val64 = match esize {
                    0 => {
                        let b = l.imm(l.u64, B::BitAnd, r, 0xff);
                        l.bin(l.u64, B::Mul, b, l.k(l.u64, 0x0101_0101_0101_0101))
                    }
                    1 => {
                        let h = l.imm(l.u64, B::BitAnd, r, 0xffff);
                        l.bin(l.u64, B::Mul, h, l.k(l.u64, 0x0001_0001_0001_0001))
                    }
                    2 => {
                        let w = l.imm(l.u64, B::BitAnd, r, 0xffff_ffff);
                        l.bin(l.u64, B::BitOr, w, l.imm(l.u64, B::Shl, w, 32))
                    }
                    _ => r,
                };
                let at = offset_of!(Cpu, v) + rd as usize * 16;
                l.store(at, val64);
                let hi = if q { val64 } else { l.k(l.u64, 0) };
                l.store(at + 8, hi);
            }
            Instruction::SimdExt(a) => {
                let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                let at_m = offset_of!(Cpu, v) + a.rm as usize * 16;
                let vn_lo = l.load(l.u64, at_n);
                let vn_hi = l.load(l.u64, at_n + 8);
                let vm_lo = l.load(l.u64, at_m);
                let vm_hi = l.load(l.u64, at_m + 8);
                let (lo, hi) = if a.imm == 0 {
                    (vn_lo, vn_hi)
                } else if a.imm == 8 {
                    (vn_hi, vm_lo)
                } else if a.imm < 8 {
                    let s = a.imm as u64 * 8;
                    let r = (8 - a.imm as u64) * 8;
                    let lo = l.bin(
                        l.u64,
                        B::BitOr,
                        l.imm(l.u64, B::Shr, vn_lo, s),
                        l.imm(l.u64, B::Shl, vn_hi, r),
                    );
                    let hi = l.bin(
                        l.u64,
                        B::BitOr,
                        l.imm(l.u64, B::Shr, vn_hi, s),
                        l.imm(l.u64, B::Shl, vm_lo, r),
                    );
                    (lo, hi)
                } else {
                    let shift = a.imm as u64 - 8;
                    let s = shift * 8;
                    let r = (8 - shift) * 8;
                    let lo = l.bin(
                        l.u64,
                        B::BitOr,
                        l.imm(l.u64, B::Shr, vn_hi, s),
                        l.imm(l.u64, B::Shl, vm_lo, r),
                    );
                    let hi = l.bin(
                        l.u64,
                        B::BitOr,
                        l.imm(l.u64, B::Shr, vm_lo, s),
                        l.imm(l.u64, B::Shl, vm_hi, r),
                    );
                    (lo, hi)
                };
                l.store(at_d, lo);
                l.store(at_d + 8, hi);
            }
            Instruction::SimdCompareZero(a) => {
                let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                let vn_lo = l.load(l.u64, at_n);
                let vn_hi = l.load(l.u64, at_n + 8);
                let ebits = (1u64 << a.size) * 8;
                let emask = if ebits == 64 {
                    u64::MAX
                } else {
                    (1u64 << ebits) - 1
                };
                let halves = if a.q { 2 } else { 1 };
                for half in 0..2 {
                    if half >= halves {
                        l.store(at_d + half * 8, l.k(l.u64, 0));
                        continue;
                    }
                    let v = if half == 0 { vn_lo } else { vn_hi };
                    let mut acc = l.k(l.u64, 0);
                    let lanes = 8 >> a.size;
                    for lane in 0..lanes {
                        let shift = lane as u64 * ebits;
                        let field = l.imm(l.u64, B::BitAnd, l.imm(l.u64, B::Shr, v, shift), emask);
                        let is_zero = l.cmp(C::Eq, field, l.k(l.u64, 0));
                        let bits = l.sel(l.u64, is_zero, l.k(l.u64, emask), l.k(l.u64, 0));
                        acc = l.bin(l.u64, B::BitOr, acc, l.imm(l.u64, B::Shl, bits, shift));
                    }
                    l.store(at_d + half * 8, acc);
                }
            }
            Instruction::SimdTable(a) => {
                // Table bytes live in `len + 1` consecutive registers. Each index
                // byte selects a table byte; out-of-range indices leave zero for
                // `tbl` and keep the destination byte for `tbx`.
                let at_m = offset_of!(Cpu, v) + a.rm as usize * 16;
                let vm_lo = l.load(l.u64, at_m);
                let vm_hi = l.load(l.u64, at_m + 8);
                let regs = a.len as usize + 1;
                let mut table = Vec::with_capacity(regs * 2);
                for i in 0..regs {
                    let reg = (a.rn as usize + i) % 32;
                    let at = offset_of!(Cpu, v) + reg * 16;
                    table.push(l.load(l.u64, at));
                    table.push(l.load(l.u64, at + 8));
                }
                let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                let old_lo = if a.extend {
                    Some(l.load(l.u64, at_d))
                } else {
                    None
                };
                let old_hi = if a.extend {
                    Some(l.load(l.u64, at_d + 8))
                } else {
                    None
                };
                let halves = if a.q { 2 } else { 1 };
                let limit = (16 * regs) as u64;
                for half in 0..2 {
                    if half >= halves {
                        l.store(at_d + half * 8, l.k(l.u64, 0));
                        continue;
                    }
                    let indices = if half == 0 { vm_lo } else { vm_hi };
                    let old = if half == 0 { old_lo } else { old_hi };
                    let mut acc = l.k(l.u64, 0);
                    for byte in 0..8u64 {
                        let shift = byte * 8;
                        let index =
                            l.imm(l.u64, B::BitAnd, l.imm(l.u64, B::Shr, indices, shift), 0xff);
                        let in_range = l.cmp(C::Lt, index, l.k(l.u64, limit));
                        let mut value = l.k(l.u64, 0);
                        let sel = |c: Value, x: Value, y: Value| l.sel(l.u64, c, x, y);
                        for (b, entry) in table.iter().enumerate() {
                            let here = l.cmp(
                                C::Eq,
                                l.bin(l.u64, B::Shr, index, l.k(l.u64, 3)),
                                l.k(l.u64, b as u64),
                            );
                            let byte_at = l.imm(
                                l.u64,
                                B::BitAnd,
                                l.bin(
                                    l.u64,
                                    B::Shr,
                                    *entry,
                                    l.bin(
                                        l.u64,
                                        B::Shl,
                                        l.imm(l.u64, B::BitAnd, index, 7),
                                        l.k(l.u64, 3),
                                    ),
                                ),
                                0xff,
                            );
                            value =
                                sel(l.bin(l.boolean, B::BitAnd, in_range, here), byte_at, value);
                        }
                        if a.extend {
                            value = sel(
                                in_range,
                                value,
                                l.imm(
                                    l.u64,
                                    B::BitAnd,
                                    l.imm(l.u64, B::Shr, old.unwrap(), shift),
                                    0xff,
                                ),
                            );
                        }
                        acc = l.bin(l.u64, B::BitOr, acc, l.imm(l.u64, B::Shl, value, shift));
                    }
                    l.store(at_d + half * 8, acc);
                }
            }
            Instruction::SimdFmov(a) => {
                // Pure bit move; no FP arithmetic involved.
                if a.to_fp {
                    let v = l.reg(l.u64, a.rn, true);
                    let bits = if a.double {
                        v
                    } else {
                        l.imm(l.u64, B::BitAnd, v, 0xffff_ffff)
                    };
                    let at = offset_of!(Cpu, v) + a.rd as usize * 16;
                    l.store(at, bits);
                    l.store(at + 8, l.k(l.u64, 0));
                } else {
                    let at = offset_of!(Cpu, v) + a.rn as usize * 16;
                    let v = l.load(l.u64, at);
                    let bits = if a.double {
                        v
                    } else {
                        l.imm(l.u64, B::BitAnd, v, 0xffff_ffff)
                    };
                    l.put(a.rd, bits);
                }
            }
            Instruction::SimdAlu(a) => {
                use crate::aarch64::decode::SimdAluOp;
                // Bitwise logical operations work on whole halves, lane width free.
                if matches!(
                    a.op,
                    SimdAluOp::AddV | SimdAluOp::UMinV | SimdAluOp::UMaxV | SimdAluOp::UAddLV
                ) {
                    // Horizontal add, minimum, maximum or widening add: reduce every
                    // lane of the source and store the result in lane 0, zeroing the
                    // rest. `UAddLV` keeps twice the lane width.
                    let ebits = (1u64 << a.size) * 8;
                    let mask_of = |bits: u64| {
                        if bits >= 64 {
                            u64::MAX
                        } else {
                            (1u64 << bits) - 1
                        }
                    };
                    let emask = mask_of(ebits);
                    let result_mask = if a.op == SimdAluOp::UAddLV {
                        mask_of(ebits * 2)
                    } else {
                        emask
                    };
                    let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                    let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                    let vn_lo = l.load(l.u64, at_n);
                    let vn_hi = l.load(l.u64, at_n + 8);
                    let halves = if a.q { 2 } else { 1 };
                    let mut total = if a.op == SimdAluOp::UMinV {
                        l.k(l.u64, emask)
                    } else {
                        l.k(l.u64, 0)
                    };
                    for half in 0..halves {
                        let v = if half == 0 { vn_lo } else { vn_hi };
                        for lane in 0..(8 >> a.size) {
                            let shift = lane as u64 * ebits;
                            let field =
                                l.imm(l.u64, B::BitAnd, l.imm(l.u64, B::Shr, v, shift), emask);
                            total = match a.op {
                                SimdAluOp::UMinV => {
                                    l.sel(l.u64, l.cmp(C::Lt, field, total), field, total)
                                }
                                SimdAluOp::UMaxV => {
                                    l.sel(l.u64, l.cmp(C::Gt, field, total), field, total)
                                }
                                _ => l.bin(l.u64, B::Add, total, field),
                            };
                        }
                    }
                    l.store(
                        at_d,
                        l.bin(l.u64, B::BitAnd, total, l.k(l.u64, result_mask)),
                    );
                    l.store(at_d + 8, l.k(l.u64, 0));
                } else if matches!(
                    a.op,
                    SimdAluOp::And
                        | SimdAluOp::Bic
                        | SimdAluOp::Orr
                        | SimdAluOp::Orn
                        | SimdAluOp::Eor
                        | SimdAluOp::Bsl
                        | SimdAluOp::Bit
                        | SimdAluOp::Bif
                ) {
                    let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                    let at_m = offset_of!(Cpu, v) + a.rm as usize * 16;
                    let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                    for half in 0..2 {
                        if half == 1 && !a.q {
                            l.store(at_d + 8, l.k(l.u64, 0));
                            continue;
                        }
                        let x = l.load(l.u64, at_n + half * 8);
                        let y = l.load(l.u64, at_m + half * 8);
                        let not = |v| l.bin(l.u64, B::BitXor, v, l.k(l.u64, u64::MAX));
                        let r = match a.op {
                            SimdAluOp::And => l.bin(l.u64, B::BitAnd, x, y),
                            SimdAluOp::Bic => l.bin(l.u64, B::BitAnd, x, not(y)),
                            SimdAluOp::Orr => l.bin(l.u64, B::BitOr, x, y),
                            SimdAluOp::Orn => l.bin(l.u64, B::BitOr, x, not(y)),
                            SimdAluOp::Eor => l.bin(l.u64, B::BitXor, x, y),
                            // The destination is both a source mask and the
                            // merge base for BSL/BIT/BIF.
                            _ => {
                                let d = l.load(l.u64, at_d + half * 8);
                                match a.op {
                                    SimdAluOp::Bsl => l.bin(
                                        l.u64,
                                        B::BitOr,
                                        l.bin(l.u64, B::BitAnd, x, d),
                                        l.bin(l.u64, B::BitAnd, y, not(d)),
                                    ),
                                    _ => {
                                        let selected = match a.op {
                                            SimdAluOp::Bit => l.bin(
                                                l.u64,
                                                B::BitAnd,
                                                l.bin(l.u64, B::BitXor, d, x),
                                                y,
                                            ),
                                            _ => l.bin(
                                                l.u64,
                                                B::BitAnd,
                                                l.bin(l.u64, B::BitXor, d, x),
                                                not(y),
                                            ),
                                        };
                                        l.bin(l.u64, B::BitXor, d, selected)
                                    }
                                }
                            }
                        };
                        l.store(at_d + half * 8, r);
                    }
                } else {
                    let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                    let at_m = offset_of!(Cpu, v) + a.rm as usize * 16;
                    let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                    let vn_lo = l.load(l.u64, at_n);
                    let vn_hi = l.load(l.u64, at_n + 8);
                    let vm_lo = l.load(l.u64, at_m);
                    let vm_hi = l.load(l.u64, at_m + 8);
                    let need_d = matches!(a.op, SimdAluOp::Mla | SimdAluOp::Mls);
                    // Pairwise forms consume adjacent lanes of the full source pair:
                    // lane 0/1 of Vn and Vm, then lane 2/3, and so on.
                    if matches!(
                        a.op,
                        SimdAluOp::UMaxP
                            | SimdAluOp::SMaxP
                            | SimdAluOp::UMinP
                            | SimdAluOp::SMinP
                            | SimdAluOp::AddP
                    ) {
                        let ebits = (1u64 << a.size) * 8;
                        let emask = if ebits == 64 {
                            u64::MAX
                        } else {
                            (1u64 << ebits) - 1
                        };
                        let at_n = offset_of!(Cpu, v) + a.rn as usize * 16;
                        let at_m = offset_of!(Cpu, v) + a.rm as usize * 16;
                        let at_d = offset_of!(Cpu, v) + a.rd as usize * 16;
                        let s64 = l.signed(Width::X64);
                        let scmp =
                            |op: C, x: Value, y: Value| l.cmp(op, l.cv(s64, x), l.cv(s64, y));
                        let ssel = |c: Value, x: Value, y: Value| l.sel(l.u64, c, x, y);
                        let sources = [
                            l.load(l.u64, at_n),
                            l.load(l.u64, at_n + 8),
                            l.load(l.u64, at_m),
                            l.load(l.u64, at_m + 8),
                        ];
                        // Each source is a 128-bit vector. Lane `2k` pairs with lane
                        // `2k + 1` and the result lands in lane `k`; the reduction of Vn
                        // fills the destination's low half and Vm its high half.
                        let per_source = (128 / ebits) as usize;
                        let reduced_lanes = per_source / 2;
                        let mut out = [l.k(l.u64, 0); 2];
                        for (which, half) in [(0usize, 0usize), (2usize, 1usize)] {
                            let mut acc = l.k(l.u64, 0);
                            for k in 0..reduced_lanes {
                                let lanes_per_dw = (64 / ebits) as usize;
                                let read = |lane: usize| {
                                    let double = lane / lanes_per_dw;
                                    let shift = (lane % lanes_per_dw) as u64 * ebits;
                                    l.imm(
                                        l.u64,
                                        B::BitAnd,
                                        l.imm(l.u64, B::Shr, sources[which + double], shift),
                                        emask,
                                    )
                                };
                                let lo = read(2 * k);
                                let hi = read(2 * k + 1);
                                let r = match a.op {
                                    SimdAluOp::AddP => l.bin(l.u64, B::Add, lo, hi),
                                    SimdAluOp::UMaxP => ssel(l.cmp(C::Gt, lo, hi), lo, hi),
                                    SimdAluOp::UMinP => ssel(l.cmp(C::Lt, lo, hi), lo, hi),
                                    SimdAluOp::SMaxP => ssel(scmp(C::Gt, lo, hi), lo, hi),
                                    _ => ssel(scmp(C::Lt, lo, hi), lo, hi),
                                };
                                let masked = l.bin(l.u64, B::BitAnd, r, l.k(l.u64, emask));
                                // per_source lanes reduce to exactly 64 bits, so the
                                // reduced lanes always pack into one doubleword.
                                acc = l.bin(
                                    l.u64,
                                    B::BitOr,
                                    acc,
                                    l.imm(l.u64, B::Shl, masked, k as u64 * ebits),
                                );
                            }
                            out[half] = acc;
                        }
                        l.store(at_d, out[0]);
                        l.store(at_d + 8, if a.q { out[1] } else { l.k(l.u64, 0) });
                    } else {
                        let vd_lo = if need_d {
                            Some(l.load(l.u64, at_d))
                        } else {
                            None
                        };
                        let vd_hi = if need_d {
                            Some(l.load(l.u64, at_d + 8))
                        } else {
                            None
                        };
                        let ebits = (1u64 << a.size) * 8;
                        let emask = if ebits == 64 {
                            u64::MAX
                        } else {
                            (1u64 << ebits) - 1
                        };
                        let s64 = l.signed(Width::X64);
                        let smax = u64::MAX >> (65 - ebits);
                        let smin = !smax;
                        let halves = if a.q { 2 } else { 1 };
                        for half in 0..2 {
                            if half >= halves {
                                l.store(at_d + half * 8, l.k(l.u64, 0));
                                continue;
                            }
                            let vn = if half == 0 { vn_lo } else { vn_hi };
                            let vm = if half == 0 { vm_lo } else { vm_hi };
                            let vd = if half == 0 { vd_lo } else { vd_hi };
                            let mut acc = l.k(l.u64, 0);
                            let lanes = 8 >> a.size;
                            for lane in 0..lanes {
                                let shift = lane as u64 * ebits;
                                let fld = |v| {
                                    l.imm(l.u64, B::BitAnd, l.imm(l.u64, B::Shr, v, shift), emask)
                                };
                                let ext = |v: Value| {
                                    let up = l.bin(s64, B::Shl, l.cv(s64, v), l.k(s64, 64 - ebits));
                                    l.imm(s64, B::Shr, up, 64 - ebits)
                                };
                                let an = fld(vn);
                                let bn = fld(vm);
                                let sa = ext(an);
                                let sb = ext(bn);
                                let dn = vd.map(|v| fld(v));
                                let scmp = |op: C, x: Value, y: Value| {
                                    l.cmp(op, l.cv(s64, x), l.cv(s64, y))
                                };
                                let ssel = |c: Value, x: Value, y: Value| l.sel(l.u64, c, x, y);
                                let ssat = |v: Value| {
                                    let over = scmp(C::Gt, v, l.k(l.u64, smax));
                                    let under = scmp(C::Lt, v, l.k(l.u64, smin));
                                    ssel(over, l.k(l.u64, smax), ssel(under, l.k(l.u64, smin), v))
                                };
                                let r = match a.op {
                                    SimdAluOp::Add => l.bin(l.u64, B::Add, an, bn),
                                    SimdAluOp::Sub => l.bin(l.u64, B::Sub, an, bn),
                                    SimdAluOp::Mul => l.bin(l.u64, B::Mul, an, bn),
                                    SimdAluOp::Mla => l.bin(
                                        l.u64,
                                        B::Add,
                                        dn.unwrap(),
                                        l.bin(l.u64, B::Mul, an, bn),
                                    ),
                                    SimdAluOp::Mls => l.bin(
                                        l.u64,
                                        B::Sub,
                                        dn.unwrap(),
                                        l.bin(l.u64, B::Mul, an, bn),
                                    ),
                                    SimdAluOp::CmEq => {
                                        ssel(l.cmp(C::Eq, an, bn), l.k(l.u64, emask), l.k(l.u64, 0))
                                    }
                                    SimdAluOp::CmGt => {
                                        ssel(scmp(C::Gt, sa, sb), l.k(l.u64, emask), l.k(l.u64, 0))
                                    }
                                    SimdAluOp::CmGe => {
                                        ssel(scmp(C::Ge, sa, sb), l.k(l.u64, emask), l.k(l.u64, 0))
                                    }
                                    SimdAluOp::CmHi => {
                                        ssel(l.cmp(C::Gt, an, bn), l.k(l.u64, emask), l.k(l.u64, 0))
                                    }
                                    SimdAluOp::CmHs => {
                                        ssel(l.cmp(C::Ge, an, bn), l.k(l.u64, emask), l.k(l.u64, 0))
                                    }
                                    SimdAluOp::CmTst => ssel(
                                        l.cmp(
                                            C::Ne,
                                            l.bin(l.u64, B::BitAnd, an, bn),
                                            l.k(l.u64, 0),
                                        ),
                                        l.k(l.u64, emask),
                                        l.k(l.u64, 0),
                                    ),
                                    SimdAluOp::UMin => ssel(l.cmp(C::Lt, an, bn), an, bn),
                                    SimdAluOp::UMax => ssel(l.cmp(C::Gt, an, bn), an, bn),
                                    SimdAluOp::SMin => ssel(scmp(C::Lt, sa, sb), an, bn),
                                    SimdAluOp::SMax => ssel(scmp(C::Gt, sa, sb), an, bn),
                                    SimdAluOp::UAbd => {
                                        let d = l.bin(l.u64, B::Sub, an, bn);
                                        ssel(
                                            l.cmp(C::Ge, an, bn),
                                            d,
                                            l.bin(l.u64, B::Sub, l.k(l.u64, 0), d),
                                        )
                                    }
                                    SimdAluOp::SAbd => {
                                        let d = l.bin(s64, B::Sub, sa, sb);
                                        let neg = l.bin(s64, B::Sub, l.k(s64, 0), d);
                                        l.cv(
                                            l.u64,
                                            ssel(
                                                scmp(C::Ge, sa, sb),
                                                l.cv(l.u64, d),
                                                l.cv(l.u64, neg),
                                            ),
                                        )
                                    }
                                    SimdAluOp::SQAdd => {
                                        ssat(l.cv(l.u64, l.bin(s64, B::Add, sa, sb)))
                                    }
                                    SimdAluOp::UQAdd => {
                                        let s = l.bin(l.u64, B::Add, an, bn);
                                        let carry = l.cmp(C::Lt, s, an);
                                        let big = l.cmp(C::Gt, s, l.k(l.u64, emask));
                                        ssel(
                                            l.bin(l.u64, B::BitOr, carry, big),
                                            l.k(l.u64, emask),
                                            s,
                                        )
                                    }
                                    SimdAluOp::SQSub => {
                                        ssat(l.cv(l.u64, l.bin(s64, B::Sub, sa, sb)))
                                    }
                                    SimdAluOp::UQSub => {
                                        let borrow = l.cmp(C::Gt, bn, an);
                                        ssel(borrow, l.k(l.u64, 0), l.bin(l.u64, B::Sub, an, bn))
                                    }
                                    SimdAluOp::SHAdd
                                    | SimdAluOp::UHAdd
                                    | SimdAluOp::SHSub
                                    | SimdAluOp::UHSub
                                    | SimdAluOp::SRHAdd
                                    | SimdAluOp::URHAdd => {
                                        let signed_op = matches!(
                                            a.op,
                                            SimdAluOp::SHAdd | SimdAluOp::SHSub | SimdAluOp::SRHAdd
                                        );
                                        // Signed forms operate on sign-extended lanes, which
                                        // keeps the halving shift meaningful for negatives.
                                        let (op_a, op_b) = if signed_op {
                                            (l.cv(l.u64, sa), l.cv(l.u64, sb))
                                        } else {
                                            (an, bn)
                                        };
                                        let shr = |v: Value| {
                                            if signed_op {
                                                l.cv(l.u64, l.imm(s64, B::Shr, l.cv(s64, v), 1))
                                            } else {
                                                l.imm(l.u64, B::Shr, v, 1)
                                            }
                                        };
                                        let k = l.bin(l.u64, B::Sub, shr(op_a), shr(op_b));
                                        let k = if matches!(
                                            a.op,
                                            SimdAluOp::SHAdd
                                                | SimdAluOp::UHAdd
                                                | SimdAluOp::SRHAdd
                                                | SimdAluOp::URHAdd
                                        ) {
                                            l.bin(l.u64, B::Add, shr(op_a), shr(op_b))
                                        } else {
                                            k
                                        };
                                        let a0 = l.bin(l.u64, B::BitAnd, op_a, l.k(l.u64, 1));
                                        let b0 = l.bin(l.u64, B::BitAnd, op_b, l.k(l.u64, 1));
                                        let corr = match a.op {
                                            SimdAluOp::SHAdd | SimdAluOp::UHAdd => {
                                                l.bin(l.u64, B::BitAnd, a0, b0)
                                            }
                                            SimdAluOp::SRHAdd | SimdAluOp::URHAdd => {
                                                l.bin(l.u64, B::BitOr, a0, b0)
                                            }
                                            _ => l.bin(
                                                l.u64,
                                                B::BitAnd,
                                                l.bin(l.u64, B::BitXor, a0, l.k(l.u64, 1)),
                                                b0,
                                            ),
                                        };
                                        if matches!(
                                            a.op,
                                            SimdAluOp::SHAdd
                                                | SimdAluOp::UHAdd
                                                | SimdAluOp::SRHAdd
                                                | SimdAluOp::URHAdd
                                        ) {
                                            l.bin(l.u64, B::Add, k, corr)
                                        } else {
                                            l.bin(l.u64, B::Sub, k, corr)
                                        }
                                    }
                                    SimdAluOp::SSHl
                                    | SimdAluOp::USHl
                                    | SimdAluOp::SQShl
                                    | SimdAluOp::UQShl
                                    | SimdAluOp::SQRShl
                                    | SimdAluOp::UQRShl => {
                                        let signed_op = matches!(
                                            a.op,
                                            SimdAluOp::SSHl | SimdAluOp::SQShl | SimdAluOp::SQRShl
                                        );
                                        let rounding =
                                            matches!(a.op, SimdAluOp::SQRShl | SimdAluOp::UQRShl);
                                        let saturating = matches!(
                                            a.op,
                                            SimdAluOp::SQShl
                                                | SimdAluOp::UQShl
                                                | SimdAluOp::SQRShl
                                                | SimdAluOp::UQRShl
                                        );
                                        // Shift amount: the lane's low byte, sign-extended to
                                        // 64 bits (negative means a right shift).
                                        let amt = l.cv(
                                            s64,
                                            l.bin(
                                                s64,
                                                B::Shl,
                                                l.cv(
                                                    s64,
                                                    l.bin(l.u64, B::BitAnd, bn, l.k(l.u64, 0xff)),
                                                ),
                                                l.k(s64, 56),
                                            ),
                                        );
                                        let amt = l.cv(s64, l.bin(s64, B::Shr, amt, l.k(s64, 56)));
                                        let left = scmp(C::Ge, amt, l.k(l.u64, 0));
                                        let n = ssel(
                                            left,
                                            l.cv(l.u64, amt),
                                            l.cv(l.u64, l.bin(s64, B::Sub, l.k(s64, 0), amt)),
                                        );
                                        let shl = |v: Value, k: Value| l.bin(l.u64, B::Shl, v, k);
                                        let sar = |v: Value, k: Value| {
                                            l.cv(l.u64, l.bin(s64, B::Shr, l.cv(s64, v), k))
                                        };
                                        let shr = |v: Value, k: Value| l.bin(l.u64, B::Shr, v, k);
                                        // Beyond the lane width: a plain right shift keeps the
                                        // sign fill (or zero), a rounding shift collapses to zero,
                                        // and a saturating left shift pins to the lane bound.
                                        let sign_fill = if signed_op {
                                            ssel(
                                                scmp(C::Lt, sa, l.k(l.u64, 0)),
                                                l.k(l.u64, emask),
                                                l.k(l.u64, 0),
                                            )
                                        } else {
                                            l.k(l.u64, 0)
                                        };
                                        let sat_fill = if signed_op {
                                            ssel(
                                                scmp(C::Lt, sa, l.k(l.u64, 0)),
                                                l.k(l.u64, smin),
                                                l.k(l.u64, smax),
                                            )
                                        } else {
                                            l.k(l.u64, emask)
                                        };
                                        let far = ssel(
                                            left,
                                            if saturating { sat_fill } else { l.k(l.u64, 0) },
                                            if rounding { l.k(l.u64, 0) } else { sign_fill },
                                        );
                                        // Shift when the amount stays inside the lane width.
                                        // Clamping to `ebits - 1` keeps the host shift defined.
                                        let kc = ssel(
                                            l.cmp(C::Ge, n, l.k(l.u64, ebits)),
                                            l.k(l.u64, ebits - 1),
                                            n,
                                        );
                                        let rnd = if rounding {
                                            shl(
                                                l.k(l.u64, 1),
                                                l.bin(l.u64, B::Sub, kc, l.k(l.u64, 1)),
                                            )
                                        } else {
                                            l.k(l.u64, 0)
                                        };
                                        let opnd = if signed_op { l.cv(l.u64, sa) } else { an };
                                        let t = if rounding {
                                            // The rounding constant must not leak out of the lane.
                                            l.bin(
                                                l.u64,
                                                B::BitAnd,
                                                l.bin(l.u64, B::Add, opnd, rnd),
                                                l.k(l.u64, emask),
                                            )
                                        } else {
                                            opnd
                                        };
                                        let right = if rounding {
                                            if signed_op { sar(t, kc) } else { shr(t, kc) }
                                        } else if signed_op {
                                            sar(opnd, kc)
                                        } else {
                                            shr(opnd, kc)
                                        };
                                        let shifted = shl(opnd, kc);
                                        let moved = ssel(left, shifted, right);
                                        let moved =
                                            ssel(l.cmp(C::Ge, n, l.k(l.u64, ebits)), far, moved);
                                        if !saturating {
                                            moved
                                        } else if signed_op {
                                            // Saturation only matters for left shifts; the
                                            // right-shift result already fits the lane. Below
                                            // the 64-bit lane the shifted value is exact, so the
                                            // lane bounds test it directly; a 64-bit lane needs
                                            // the round-trip check because the shift wraps.
                                            let ov = if ebits == 64 {
                                                l.cmp(C::Ne, sar(shifted, kc), opnd)
                                            } else {
                                                l.bin(
                                                    l.boolean,
                                                    B::BitOr,
                                                    scmp(C::Gt, shifted, l.k(l.u64, smax)),
                                                    scmp(C::Lt, shifted, l.k(l.u64, smin)),
                                                )
                                            };
                                            let sat = ssel(
                                                scmp(C::Lt, sa, l.k(l.u64, 0)),
                                                l.k(l.u64, smin),
                                                l.k(l.u64, smax),
                                            );
                                            ssel(left, ssel(ov, sat, moved), moved)
                                        } else {
                                            let ov = if ebits == 64 {
                                                l.cmp(C::Ne, shr(shifted, kc), an)
                                            } else {
                                                l.cmp(C::Gt, shifted, l.k(l.u64, emask))
                                            };
                                            ssel(left, ssel(ov, l.k(l.u64, emask), moved), moved)
                                        }
                                    }
                                    _ => l.k(l.u64, 0),
                                };
                                let masked = l.bin(l.u64, B::BitAnd, r, l.k(l.u64, emask));
                                acc = l.bin(
                                    l.u64,
                                    B::BitOr,
                                    acc,
                                    l.imm(l.u64, B::Shl, masked, shift),
                                );
                            }
                            l.store(at_d + half * 8, acc);
                        }
                    }
                }
            }
            Instruction::SimdPair(a) => {
                let trap = if a.op == MemoryOp::Store {
                    Trap::Store
                } else {
                    Trap::Load
                };
                let (address, post) = l.address_split(a.rn, a.addressing);
                if !inline_memory {
                    l.simd_trap_access(address, a.bytes as u64, a.rt, Some(a.rt2), post, pc, trap);
                } else {
                    let at_rt = offset_of!(Cpu, v) + a.rt as usize * 16;
                    let at_rt2 = offset_of!(Cpu, v) + a.rt2 as usize * 16;
                    let slow_block = l.new_block();
                    let store = a.op == MemoryOp::Store;
                    let (bytes, chunk_count) = if a.bytes == 16 { (16, 4) } else { (8, 2) };
                    let starts: [Value; 4] = [
                        address,
                        l.imm(l.u64, B::Add, address, 8),
                        l.imm(l.u64, B::Add, address, 16),
                        l.imm(l.u64, B::Add, address, 24),
                    ];
                    // stores: (reg_offset_base, half) per chunk
                    let src: [(usize, u64); 4] = if bytes == 16 {
                        [(at_rt, 0), (at_rt + 8, 0), (at_rt2, 0), (at_rt2 + 8, 0)]
                    } else {
                        [(at_rt, 0), (at_rt2, 0), (at_rt, 0), (at_rt, 0)]
                    };
                    let mut hosts: [Option<Value>; 4] = [None; 4];
                    for i in 0..chunk_count {
                        let next = l.new_block();
                        let (hit, host) = l.dtlb_probe(starts[i], Size::Double, store);
                        l.branch_if(hit, next, slow_block);
                        l.block.set(next);
                        hosts[i] = Some(host);
                    }
                    if store {
                        for i in 0..chunk_count {
                            let host = hosts[i].unwrap();
                            l.store_ptr(host, l.load(l.u64, src[i].0));
                        }
                    } else {
                        for i in 0..chunk_count {
                            let host = hosts[i].unwrap();
                            l.store(src[i].0, l.load_ptr(l.u64, host));
                        }
                        // zero the untransferred half of each 128-bit register
                        if bytes == 8 {
                            l.store(at_rt + 8, l.k(l.u64, 0));
                            l.store(at_rt2 + 8, l.k(l.u64, 0));
                        }
                    }
                    if let Some((rn, value)) = post {
                        l.put_sp(rn, value);
                    }
                    // slow path, built last so every fast edge into it exists.
                    let resume = l.block.get();
                    l.block.set(slow_block);
                    l.simd_trap_access(address, bytes as u64, a.rt, Some(a.rt2), post, pc, trap);
                    l.block.set(resume);
                }
            }
            Instruction::SimdMemory(a) => {
                let trap = if a.op == MemoryOp::Store {
                    Trap::Store
                } else {
                    Trap::Load
                };
                let (address, post) = l.address_split(a.rn, a.addressing);
                if !inline_memory {
                    l.simd_trap_access(address, a.bytes as u64, a.rt, None, post, pc, trap);
                } else {
                    let at_rt = offset_of!(Cpu, v) + a.rt as usize * 16;
                    let slow_block = l.new_block();
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
                    let starts: [Value; 2] = [address, l.imm(l.u64, B::Add, address, 8)];
                    let chunk_ty = l.uint((a.bytes.min(8) as u16) * 8);
                    let src: [usize; 2] = [at_rt, at_rt + 8];
                    let mut hosts: [Option<Value>; 2] = [None; 2];
                    for i in 0..chunk_count {
                        let next = l.new_block();
                        let (hit, host) = l.dtlb_probe(starts[i], size, store);
                        l.branch_if(hit, next, slow_block);
                        l.block.set(next);
                        hosts[i] = Some(host);
                    }
                    if store {
                        for i in 0..chunk_count {
                            let host = hosts[i].unwrap();
                            l.store_ptr(host, l.cv(chunk_ty, l.load(l.u64, src[i])));
                        }
                    } else {
                        for i in 0..chunk_count {
                            let host = hosts[i].unwrap();
                            l.store(src[i], l.cv(l.u64, l.load_ptr(chunk_ty, host)));
                        }
                        // Scalar loads zero the whole upper register, not just the
                        // untransferred half.
                        if a.bytes <= 8 {
                            l.store(at_rt + 8, l.k(l.u64, 0));
                            if a.bytes < 8 {
                                let kept = (1u64 << (a.bytes * 8)) - 1;
                                let lo = l.imm(l.u64, B::BitAnd, l.load(l.u64, at_rt), kept);
                                l.store(at_rt, lo);
                            }
                        }
                    }
                    if let Some((rn, value)) = post {
                        l.put_sp(rn, value);
                    }
                    let resume = l.block.get();
                    l.block.set(slow_block);
                    l.simd_trap_access(address, a.bytes as u64, a.rt, None, post, pc, trap);
                    l.block.set(resume);
                }
            }
            Instruction::Nop
            | Instruction::CacheOp
            | Instruction::Dsb
            | Instruction::Dmb
            | Instruction::Yield => {}
            Instruction::Wfe => l.trap(pc.wrapping_add(4), Trap::Wfe),
            Instruction::Brk(imm) => {
                l.store(offset_of!(Cpu, address), l.k(l.u64, u64::from(imm)));
                l.trap(pc, Trap::Brk)
            }
            Instruction::Isb => l.trap(pc.wrapping_add(4), Trap::Isb),
            Instruction::Tlbi => l.trap(pc.wrapping_add(4), Trap::Tlbi),
            Instruction::DcZva(rt) => {
                l.store(offset_of!(Cpu, address), l.reg(l.u64, rt, true));
                l.trap(pc.wrapping_add(4), Trap::DcZva)
            }
            Instruction::Eret => l.trap(pc.wrapping_add(4), Trap::Eret),
            Instruction::Svc => l.trap(pc.wrapping_add(4), Trap::Svc),
            Instruction::Psci => l.trap(pc.wrapping_add(4), Trap::Psci),
            Instruction::Wfi => l.trap(pc.wrapping_add(4), Trap::Wfi),
        }
        pc = pc.wrapping_add(4);
        if ended {
            break;
        }
    }
    if !ended {
        l.ret(l.k(l.u64, pc))
    }
    Ok(Block::new(native::compile_owned(l.f.into_inner())?))
}
/// Describe a GPR pair access to the machine: the first access in the usual
/// request fields, the second as `second_*`, completed after the first.
fn setup_pair(l: &Lower, address: Value, a: Pair) {
    setup_memory(l, address, a.size, a.rt, a.signed, a.op);
    l.store(
        offset_of!(Cpu, second_address),
        l.imm(l.u64, B::Add, address, a.size as u64),
    );
    l.byte(offset_of!(Cpu, second_width), a.size as u64);
    l.byte(offset_of!(Cpu, second_dest), a.rt2 as u64);
    if a.signed != SignExtend::None {
        l.byte(offset_of!(Cpu, second_signed), a.signed as u64)
    }
    if a.op == MemoryOp::Store {
        l.store(offset_of!(Cpu, second_value), l.reg(l.u64, a.rt2, true))
    }
    l.byte(offset_of!(Cpu, second_pending), 1);
}
fn setup_memory(l: &Lower, address: Value, size: Size, rt: u8, signed: SignExtend, op: MemoryOp) {
    l.store(offset_of!(Cpu, address), address);
    l.byte(offset_of!(Cpu, width), size as u64);
    l.byte(offset_of!(Cpu, dest), rt as u64);
    if signed != SignExtend::None {
        l.byte(offset_of!(Cpu, load_signed), signed as u64)
    }
    if op == MemoryOp::Store {
        l.store(offset_of!(Cpu, value), l.reg(l.u64, rt, true))
    }
}

fn lower_variable(l: &Lower, a: Variable) {
    let t = l.ty(a.width);
    let lhs = l.reg(t, a.rn, true);
    let rhs = l.reg(t, a.rm, true);
    let bits = a.width as u64;
    let r = match a.op {
        VariableOp::Lsl | VariableOp::Lsr | VariableOp::Asr | VariableOp::Ror => {
            let distance = l.imm(t, B::BitAnd, rhs, bits - 1);
            match a.op {
                VariableOp::Lsl => l.bin(t, B::Shl, lhs, distance),
                VariableOp::Lsr => l.bin(t, B::Shr, lhs, distance),
                VariableOp::Asr => {
                    let s = l.signed(a.width);
                    l.cv(t, l.bin(s, B::Shr, l.cv(s, lhs), l.cv(s, distance)))
                }
                VariableOp::Ror => {
                    let back = l.imm(
                        t,
                        B::BitAnd,
                        l.bin(t, B::Sub, l.k(t, bits), distance),
                        bits - 1,
                    );
                    l.bin(
                        t,
                        B::BitOr,
                        l.bin(t, B::Shr, lhs, distance),
                        l.bin(t, B::Shl, lhs, back),
                    )
                }
                _ => unreachable!(),
            }
        }
        VariableOp::Udiv | VariableOp::Sdiv => {
            let zero = l.k(t, 0);
            let one = l.k(t, 1);
            let by_zero = l.cmp(C::Eq, rhs, zero);
            if a.op == VariableOp::Udiv {
                let safe = l.sel(t, by_zero, one, rhs);
                l.sel(t, by_zero, zero, l.bin(t, B::Div, lhs, safe))
            } else {
                let minus_one = l.cmp(C::Eq, rhs, l.k(t, u64::MAX));
                let safe = l.sel(t, minus_one, one, l.sel(t, by_zero, one, rhs));
                let s = l.signed(a.width);
                let quotient = l.cv(t, l.bin(s, B::Div, l.cv(s, lhs), l.cv(s, safe)));
                let negated = l.bin(t, B::Sub, zero, lhs);
                l.sel(t, by_zero, zero, l.sel(t, minus_one, negated, quotient))
            }
        }
    };
    if a.rd != 31 {
        l.put(a.rd, r)
    }
}
fn lower_bitfield(l: &Lower, a: Bitfield) {
    let t = l.ty(a.width);
    let bits = a.width as u64;
    let len = a.len as u64;
    let lsb = a.lsb as u64;
    let source = l.reg(t, a.rn, true);
    let mask = low_mask(a.len);
    let width_mask = if a.width == Width::X64 {
        u64::MAX
    } else {
        u32::MAX as u64
    };
    let extract = |signed| {
        let raised = l.imm(t, B::Shl, source, bits - lsb - len);
        l.shift(
            a.width,
            raised,
            if signed { Shift::Asr } else { Shift::Lsr },
            bits - len,
        )
    };
    let r = match a.op {
        BitfieldOp::Unsigned => {
            if a.extract {
                extract(false)
            } else {
                l.imm(
                    t,
                    B::Shl,
                    l.imm(t, B::BitAnd, source, mask & width_mask),
                    lsb,
                )
            }
        }
        BitfieldOp::Signed => {
            if a.extract {
                extract(true)
            } else {
                let raised = l.imm(t, B::Shl, source, bits - len);
                let extended = l.shift(a.width, raised, Shift::Asr, bits - len);
                l.imm(t, B::Shl, extended, lsb)
            }
        }
        BitfieldOp::Insert => {
            let value = if a.extract {
                extract(false)
            } else {
                l.imm(
                    t,
                    B::Shl,
                    l.imm(t, B::BitAnd, source, mask & width_mask),
                    lsb,
                )
            };
            let place = if a.extract { 0 } else { lsb };
            let hole = (mask << place) & width_mask;
            let kept = l.imm(t, B::BitAnd, l.reg(t, a.rd, true), !hole & width_mask);
            l.bin(t, B::BitOr, kept, value)
        }
    };
    if a.rd != 31 {
        l.put(a.rd, r)
    }
}
fn repeated(pattern: u64, unit: u32, bits: u32) -> u64 {
    let mut r = 0;
    let mut at = 0;
    while at < bits {
        r |= pattern << at;
        at += unit
    }
    r
}
fn swap(l: &Lower, t: Type, v: Value, mask: u64, n: u64) -> Value {
    let low = l.imm(t, B::BitAnd, v, mask);
    let high = l.imm(t, B::BitAnd, l.imm(t, B::Shr, v, n), mask);
    l.bin(t, B::BitOr, l.imm(t, B::Shl, low, n), high)
}
fn lower_unary(l: &Lower, a: Unary) {
    let t = l.ty(a.width);
    let bits = a.width as u32;
    let mut v = l.reg(t, a.rn, true);
    match a.op {
        UnaryOp::Rev16 => v = swap(l, t, v, repeated(0xff, 16, bits), 8),
        UnaryOp::Rev32 => {
            v = swap(l, t, v, repeated(0xff, 16, bits), 8);
            v = swap(l, t, v, repeated(0xffff, 32, bits), 16);
        }
        UnaryOp::Rev | UnaryOp::Rbit => {
            if a.op == UnaryOp::Rbit {
                for (mask, unit, n) in [(1, 2, 1), (3, 4, 2), (15, 8, 4)] {
                    v = swap(l, t, v, repeated(mask, unit, bits), n)
                }
            }
            v = swap(l, t, v, repeated(0xff, 16, bits), 8);
            v = swap(l, t, v, repeated(0xffff, 32, bits), 16);
            if bits == 64 {
                v = l.bin(
                    t,
                    B::BitOr,
                    l.imm(t, B::Shl, v, 32),
                    l.imm(t, B::Shr, v, 32),
                )
            }
        }
        UnaryOp::Clz => {
            let original = v;
            let mut count = l.k(t, 0);
            let mut step = bits / 2;
            while step >= 1 {
                let top = l.imm(t, B::Shr, v, (bits - step) as u64);
                let empty = l.cmp(C::Eq, top, l.k(t, 0));
                v = l.sel(t, empty, l.imm(t, B::Shl, v, step as u64), v);
                count = l.sel(t, empty, l.imm(t, B::Add, count, step as u64), count);
                step /= 2;
            }
            v = l.sel(
                t,
                l.cmp(C::Eq, original, l.k(t, 0)),
                l.k(t, bits as u64),
                count,
            );
        }
    }
    if a.rd != 31 {
        l.put(a.rd, v)
    }
}
fn lower_high(l: &Lower, a: MultiplyHigh) {
    let t = l.u64;
    let x = l.reg(t, a.rn, true);
    let y = l.reg(t, a.rm, true);
    let xl = l.imm(t, B::BitAnd, x, 0xffffffff);
    let yl = l.imm(t, B::BitAnd, y, 0xffffffff);
    let xh = l.imm(t, B::Shr, x, 32);
    let yh = l.imm(t, B::Shr, y, 32);
    let ll = l.bin(t, B::Mul, xl, yl);
    let lh = l.bin(t, B::Mul, xl, yh);
    let hl = l.bin(t, B::Mul, xh, yl);
    let hh = l.bin(t, B::Mul, xh, yh);
    let middle = l.bin(
        t,
        B::Add,
        l.bin(
            t,
            B::Add,
            l.imm(t, B::Shr, ll, 32),
            l.imm(t, B::BitAnd, lh, 0xffffffff),
        ),
        l.imm(t, B::BitAnd, hl, 0xffffffff),
    );
    let mut high = l.bin(
        t,
        B::Add,
        l.bin(t, B::Add, hh, l.imm(t, B::Shr, lh, 32)),
        l.bin(
            t,
            B::Add,
            l.imm(t, B::Shr, hl, 32),
            l.imm(t, B::Shr, middle, 32),
        ),
    );
    if a.signed {
        let zero = l.k(t, 0);
        let xn = l.bin(t, B::Sub, zero, l.imm(t, B::Shr, x, 63));
        let yn = l.bin(t, B::Sub, zero, l.imm(t, B::Shr, y, 63));
        high = l.bin(t, B::Sub, high, l.bin(t, B::BitAnd, y, xn));
        high = l.bin(t, B::Sub, high, l.bin(t, B::BitAnd, x, yn));
    }
    if a.rd != 31 {
        l.put(a.rd, high)
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
        CntpCtlEl0 => offset_of!(CpuSystem, cntp_ctl_el0),
        ActlrEl1 => offset_of!(CpuSystem, actlr_el1),
        VbarEl1 => offset_of!(CpuSystem, vbar_el1),
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
        MidrEl1 => offset_of!(CpuSystem, midr_el1),
        IdAa64pfr0El1 => offset_of!(CpuSystem, id_aa64pfr0_el1),
        IdAa64mmfr0El1 => offset_of!(CpuSystem, id_aa64mmfr0_el1),
        _ => return None,
    })
}
fn lower_system(l: &Lower, a: System) {
    use SystemRegister::*;
    l.flush_counter();
    let at = offset_of!(Cpu, system);
    let level = match a.register {
        SpEl0 => Some(0u64),
        SpEl1 => Some(1),
        SpEl2 => Some(2),
        _ => None,
    };
    if let Some(level) = level {
        let current = l.cv(l.u64, l.load(l.u8, at + offset_of!(CpuSystem, el)));
        let at_level = l.cmp(C::Eq, current, l.k(l.u64, level));
        let live = if level <= 1 {
            let el1 = l.cmp(C::Eq, current, l.k(l.u64, 1));
            let selected = l.cmp(
                if level == 0 { C::Eq } else { C::Ne },
                l.load(l.u8, at + offset_of!(CpuSystem, spsel)),
                l.k(l.u8, 0),
            );
            let el1_selected = l.bin(l.boolean, B::BitAnd, el1, selected);
            if level == 0 {
                l.bin(l.boolean, B::BitOr, at_level, el1_selected)
            } else {
                el1_selected
            }
        } else {
            at_level
        };
        let p = l.sel(
            l.ptr,
            live,
            l.addr(Lower::reg_at(31)),
            l.addr(at + offset_of!(CpuSystem, sp_el) + level as usize * 8),
        );
        if a.read {
            l.put(a.rt, l.load_ptr(l.u64, p))
        } else {
            l.store_ptr(p, l.reg(l.u64, a.rt, true))
        }
        return;
    }
    match a.register {
        CurrentEl => l.put(
            a.rt,
            l.imm(
                l.u64,
                B::Shl,
                l.cv(l.u64, l.load(l.u8, at + offset_of!(CpuSystem, el))),
                2,
            ),
        ),
        Pan | PanSet => {
            // PSTATE.PAN lives in bit 22 of the register form.
            let pan = at + offset_of!(CpuSystem, pan);
            if a.read {
                l.put(a.rt, l.imm(l.u64, B::Shl, l.load(l.u64, pan), 22));
            } else if a.register == PanSet {
                l.store(pan, l.k(l.u64, u64::from(a.immediate & 1)));
            } else {
                let bit = l.imm(l.u64, B::Shr, l.reg(l.u64, a.rt, true), 22);
                l.store(pan, l.imm(l.u64, B::BitAnd, bit, 1));
            }
        }
        RazWi | AppleZero => {
            if a.read {
                l.put(a.rt, l.k(l.u64, 0));
            }
        }
        SpSel | SpSelSet => {
            let selected = l.load(l.u8, at + offset_of!(CpuSystem, spsel));
            if a.read {
                l.put(a.rt, l.cv(l.u64, selected));
                return;
            }
            let new = if a.register == SpSelSet {
                l.k(l.u8, u64::from(a.immediate & 1))
            } else {
                l.cv(l.u8, l.imm(l.u64, B::BitAnd, l.reg(l.u64, a.rt, true), 1))
            };
            // The live stack pointer belongs to bank `SP_EL1` when SPSel is 1
            // and `SP_EL0` when it is 0. Park it in its bank, then load the
            // other one. Selecting the bank already in use is a store and a
            // reload of the same slot.
            let banks = at + offset_of!(CpuSystem, sp_el);
            let bank = |select: Value| {
                l.sel(
                    l.ptr,
                    l.cmp(C::Ne, select, l.k(l.u8, 0)),
                    l.addr(banks + 8),
                    l.addr(banks),
                )
            };
            let live = l.load(l.u64, Lower::reg_at(31));
            l.store_ptr(bank(selected), live);
            let incoming = l.load_ptr(l.u64, bank(new));
            l.store(Lower::reg_at(31), incoming);
            l.store(at + offset_of!(CpuSystem, spsel), new);
        }
        DaifSet | DaifClear => {
            let p = at + offset_of!(CpuSystem, daif);
            let before = l.load(l.u64, p);
            let updated = l.imm(
                l.u64,
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
            l.store(p, updated);
        }
        Nzcv => {
            if a.read {
                l.put(a.rt, l.cv(l.u64, l.load(l.u32, offset_of!(Cpu, flags))))
            } else {
                l.store(
                    offset_of!(Cpu, flags),
                    l.imm(
                        l.u32,
                        B::BitAnd,
                        l.cv(l.u32, l.reg(l.u64, a.rt, true)),
                        0xf0000000,
                    ),
                )
            }
        }
        OslsrEl1 => {
            let either = l.bin(
                l.u64,
                B::BitOr,
                l.load(l.u64, at + offset_of!(CpuSystem, oslar_el1)),
                l.load(l.u64, at + offset_of!(CpuSystem, osdlr_el1)),
            );
            let locked = l.cmp(C::Ne, either, l.k(l.u64, 0));
            l.put(a.rt, l.sel(l.u64, locked, l.k(l.u64, 2), l.k(l.u64, 0)));
        }
        CcsidrEl1 => {
            // CSSELR bit 0 is InD, bits 3:1 the level - 1; L2 has no separate I-side.
            use super::identification::{CCSIDR_L1D, CCSIDR_L1I, CCSIDR_L2};
            let select = l.imm(
                l.u64,
                B::BitAnd,
                l.load(l.u64, at + offset_of!(CpuSystem, csselr_el1)),
                0xf,
            );
            let is = |n: u64| l.cmp(C::Eq, select, l.k(l.u64, n));
            let l1_instruction = l.sel(l.u64, is(1), l.k(l.u64, CCSIDR_L1I), l.k(l.u64, CCSIDR_L2));
            l.put(
                a.rt,
                l.sel(l.u64, is(0), l.k(l.u64, CCSIDR_L1D), l1_instruction),
            );
        }
        CntvTvalEl0 => {
            // The timer value is the signed 32-bit distance to the compare value:
            // reads give the low 32 bits of `CVAL - count`, writes `CVAL = count + sext(value)`.
            let counter = l.load(l.u64, at + offset_of!(CpuSystem, cntvct_el0));
            let compare = at + offset_of!(CpuSystem, cntv_cval_el0);
            if a.read {
                let distance = l.bin(l.u64, B::Sub, l.load(l.u64, compare), counter);
                l.put(a.rt, l.imm(l.u64, B::BitAnd, distance, 0xffff_ffff));
            } else {
                let delta = l.extend(l.reg(l.u64, a.rt, true), Extend::Sxtw);
                l.store(compare, l.bin(l.u64, B::Add, counter, delta));
            }
        }
        Daif => {
            let p = at + offset_of!(CpuSystem, daif);
            if a.read {
                l.put(a.rt, l.imm(l.u64, B::Shl, l.load(l.u64, p), 6))
            } else {
                l.store(
                    p,
                    l.imm(
                        l.u64,
                        B::BitAnd,
                        l.imm(l.u64, B::Shr, l.reg(l.u64, a.rt, true), 6),
                        0xf,
                    ),
                );
            }
        }
        _ => {
            if let Some(constant) = super::identification::value(a.register) {
                l.put(a.rt, l.k(l.u64, constant));
                return;
            }
            let offset = system_offset(a.register).expect("decoded system register has storage");
            let p = at + offset;
            if a.read {
                let v = if a.register == CntvCtlEl0 {
                    let counter = l.load(l.u64, at + offset_of!(CpuSystem, cntvct_el0));
                    let compare = l.load(l.u64, at + offset_of!(CpuSystem, cntv_cval_el0));
                    let reached = l.cmp(C::Ge, counter, compare);
                    l.bin(
                        l.u64,
                        B::BitOr,
                        l.load(l.u64, p),
                        l.sel(l.u64, reached, l.k(l.u64, 4), l.k(l.u64, 0)),
                    )
                } else {
                    l.load(l.u64, p)
                };
                l.put(a.rt, v);
            } else {
                let from = l.reg(l.u64, a.rt, true);
                l.store(
                    p,
                    if a.register == CntvCtlEl0 {
                        l.imm(l.u64, B::BitAnd, from, 3)
                    } else {
                        from
                    },
                );
            }
        }
    }
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
    fn zero_division_and_signed_overflow_do_not_trap_host() {
        // UDIV X2,X0,X1; SDIV X3,X0,X1.
        let block = compile(0, &bytes(&[0x9ac10802, 0x9ac10c03])).unwrap();
        let mut cpu = Cpu::default();
        cpu.x[0] = 1 << 63;
        cpu.x[1] = 0;
        block.run(&mut cpu);
        assert_eq!(cpu.x[2], 0);
        assert_eq!(cpu.x[3], 0);
        cpu.x[1] = u64::MAX;
        block.run(&mut cpu);
        assert_eq!(cpu.x[3], 1 << 63);
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
    #[test]
    fn simd_compare_zero_masks_each_lane() {
        // cmeq v1.16b, v0.16b, #0.
        let block = compile(0x1000, &bytes(&[0x4e209801])).unwrap();
        let mut cpu = Cpu::default();
        cpu.v[0][0] = 0x00ff_0000_0000_00ff;
        cpu.v[0][1] = 0xffff_ffff_ffff_ffff;
        block.run(&mut cpu);
        assert_eq!(cpu.v[1][0], 0xff00_ffff_ffff_ff00);
        assert_eq!(cpu.v[1][1], 0);
        // Half-width form operates on v0.8b and clears the top half.
        let block = compile(0x1000, &bytes(&[0x0e209801])).unwrap();
        cpu.v[0][0] = 0x00ff_0000_0000_00ff;
        cpu.v[0][1] = 0x1234_5678_9abc_def0;
        cpu.v[1] = [0xdead_beef_dead_beef, 0xdead_beef_dead_beef];
        block.run(&mut cpu);
        assert_eq!(cpu.v[1][0], 0xff00_ffff_ffff_ff00);
        assert_eq!(cpu.v[1][1], 0);
    }
    #[test]
    fn simd_fmov_moves_bits_between_gpr_and_fpr() {
        // fmov x2, d2 then fmov d3, x2 round-trips the low doubleword.
        let block = compile(0x1000, &bytes(&[0x9e660042, 0x9e670043])).unwrap();
        let mut cpu = Cpu::default();
        cpu.v[2] = [0x1122_3344_5566_7788, 0xdead_beef_dead_beef];
        block.run(&mut cpu);
        assert_eq!(cpu.x[2], 0x1122_3344_5566_7788);
        assert_eq!(cpu.v[3], [0x1122_3344_5566_7788, 0]);
        // 32-bit form truncates to the low word.
        let block = compile(0x1000, &bytes(&[0x1e260022])).unwrap();
        cpu.v[1] = [0xaaaa_bbbb_cccc_dddd, 0];
        block.run(&mut cpu);
        assert_eq!(cpu.x[2] as u32, 0xcccc_dddd);
    }
    #[test]
    fn simd_pairwise_reductions_match_lane_layout() {
        let setup = || {
            let mut cpu = Cpu::default();
            cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
            cpu.v[2] = [0x1112_1314_1516_1718, 0x191a_1b1c_1d1e_1f20];
            cpu
        };
        for (word, want) in [
            (
                0x6e22a420u32,
                [0x0a0c_0e10_0204_0608u64, 0x1a1c_1e20_1214_1618],
            ),
            (0x6e22ac20, [0x090b_0d0f_0103_0507, 0x191b_1d1f_1113_1517]),
            (0x4e22bc20, [0x1317_1b1f_0307_0b0f, 0x3337_3b3f_2327_2b2f]),
            (0x4ea2bc20, [0x1618_1a1c_0608_0a0c, 0x3638_3a3c_2628_2a2c]),
            (0x4ee2bc20, [0x0a0c_0e10_1214_1618, 0x2a2c_2e30_3234_3638]),
        ] {
            let mut cpu = setup();
            let block = compile(0x1000, &bytes(&[word])).unwrap();
            block.run(&mut cpu);
            assert_eq!(cpu.v[0], want, "{word:08x}");
        }
    }
    #[test]
    fn simd_horizontal_add_sums_lanes() {
        for (word, lanes, want) in [
            (0x4e31b820u32, 16usize, 0x88u64),
            (0x4e71b820, 8, 0x4048),
            (0x4eb1b820, 4, 0x1c20_2428),
        ] {
            let block = compile(0x1000, &bytes(&[word])).unwrap();
            let mut cpu = Cpu::default();
            cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
            block.run(&mut cpu);
            let _ = lanes;
            assert_eq!(cpu.v[0], [want, 0], "{word:08x}");
        }
        // The 64-bit form reads only the low half of the source.
        let block = compile(0x1000, &bytes(&[0x0e31b820])).unwrap();
        let mut cpu = Cpu::default();
        cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
        block.run(&mut cpu);
        assert_eq!(cpu.v[0], [0x24, 0]);
    }
    #[test]
    fn simd_dup_from_element_replicates_a_lane() {
        for (word, want) in [
            // dup v0.4h, v1.h[0] -> element 0x0708 in every halfword.
            (0x0e020420u32, [0x0708_0708_0708_0708u64, 0]),
            // dup v0.8h, v1.h[7] -> element 0x090a, both halves.
            (0x4e1e0420, [0x090a_090a_090a_090a, 0x090a_090a_090a_090a]),
            // dup v0.4s, v1.s[1] -> element 0x0102_0304.
            (0x4e0c0420, [0x0102_0304_0102_0304, 0x0102_0304_0102_0304]),
            // dup v0.2d, v1.d[1] -> element 0x090a_0b0c_0d0e_0f10.
            (0x4e180420, [0x090a_0b0c_0d0e_0f10, 0x090a_0b0c_0d0e_0f10]),
            // dup v0.16b, v1.b[3] -> element 0x05.
            (0x4e070420, [0x0505_0505_0505_0505, 0x0505_0505_0505_0505]),
        ] {
            let mut cpu = Cpu::default();
            cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
            let block = compile(0x1000, &bytes(&[word])).unwrap();
            block.run(&mut cpu);
            assert_eq!(cpu.v[0], want, "{word:08x}");
        }
    }
    #[test]
    fn simd_table_lookup_selects_bytes() {
        // tbl v0.16b, {v1.16b, v2.16b}, v3.16b with indices 0,2,4,...
        let block = compile(0x1000, &bytes(&[0x4e032020])).unwrap();
        let mut cpu = Cpu::default();
        cpu.v[1] = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
        cpu.v[2] = [0x1716_1514_1312_1110, 0x1f1e_1d1c_1b1a_1918];
        cpu.v[3] = [0x0604_0200_0e0c_0a08, 0x1614_1210_1e1c_1a18];
        block.run(&mut cpu);
        assert_eq!(cpu.v[0], [0x0604_0200_0e0c_0a08, 0x1614_1210_1e1c_1a18]);
        // Out-of-range indices yield zero for tbl.
        let block = compile(0x1000, &bytes(&[0x4e020020])).unwrap();
        let mut cpu = Cpu::default();
        cpu.v[1] = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
        cpu.v[2] = [0xff01_ff00_ff00_ff00, 0xffff_ffff_ffff_ffff];
        block.run(&mut cpu);
        assert_eq!(cpu.v[0], [0x0001_0000_0000_0000, 0]);
        // tbx keeps the destination byte where the index is out of range.
        let block = compile(0x1000, &bytes(&[0x4e021020])).unwrap();
        let mut cpu = Cpu::default();
        cpu.v[1] = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
        cpu.v[2] = [0xff01_ff00_ff00_ff00, 0xffff_ffff_ffff_ffff];
        cpu.v[0] = [0xdead_beef_dead_beef, 0xdead_beef_dead_beef];
        block.run(&mut cpu);
        assert_eq!(cpu.v[0], [0xde01_be00_de00_be00, 0xdead_beef_dead_beef]);
    }
    #[test]
    fn simd_alu_lane_semantics() {
        // Ground truth computed with the host's own byte arithmetic.
        struct Case {
            word: u32,
            a: [u64; 2],
            b: [u64; 2],
            want: [u64; 2],
        }
        let cases = [
            // add v0.16b, v1.16b, v2.16b wraps per byte
            Case {
                word: 0x4e228420,
                a: [0x0102_03ff_0405_0607, 0x0f10_1120_2122_2324],
                b: [0x0101_0101_0101_0101, 0x0102_0304_0506_0708],
                want: [0x0203_0400_0506_0708, 0x1012_1424_2628_2a2c],
            },
            // sub v0.16b, v1.16b, v2.16b
            Case {
                word: 0x6e228420,
                a: [0x0001_0203_0405_0607, 0],
                b: [0x0101_0101_0101_0101, 0],
                want: [0xff00_0102_0304_0506, 0],
            },
            // cmhs v0.16b, v1.16b, v2.16b (unsigned >=)
            Case {
                word: 0x6e223c20,
                a: [0x00ff_0080_7f7f_8000, 0],
                b: [0x0001_0180_7f80_8000, 0],
                want: [0xffff_00ff_ff00_ffff, 0xffff_ffff_ffff_ffff],
            },
            // umin v0.8h, v1.8h, v2.8h
            Case {
                word: 0x6e626c20,
                a: [0x0003_0002_0001_0000, 0],
                b: [0x0002_0003_0000_0001, 0],
                want: [0x0002_0002_0000_0000, 0],
            },
            // uqadd v0.8h, v1.8h, v2.8h saturates
            Case {
                word: 0x6e620c20,
                a: [0xffff_8000_0001_0000, 0],
                b: [0x0001_9000_0002_0001, 0],
                want: [0xffff_ffff_0003_0001, 0],
            },
            // shadd v0.16b, v1.16b, v2.16b rounds down (arithmetic)
            Case {
                word: 0x4e220420,
                a: [0x0303_fefe_0505_00ff, 0],
                b: [0x0303_fefe_0505_00ff, 0],
                want: [0x0303_fefe_0505_00ff, 0],
            },
            // sshl v0.8h by signed amounts in the low byte of each lane
            Case {
                word: 0x4e624420,
                a: [0x0004_0003_0002_0001, 0],
                b: [0x0000_0002_00ff_0001, 0],
                want: [0x0004_000c_0001_0002, 0],
            },
            // mla/mls fold in the destination *lane*, not the whole half
            Case {
                word: 0x4e229420,
                a: [0x0202_0202_0202_0202, 0],
                b: [0x0505_0505_0505_0505, 0],
                want: [0x0a0a_0a0a_0a0a_0a0a, 0],
            },
            Case {
                word: 0x6e229420,
                a: [0x0202_0202_0202_0202, 0],
                b: [0x0505_0505_0505_0505, 0],
                want: [0xf6f6_f6f6_f6f6_f6f6, 0],
            },
            // bsl: destination starts zero, so the result is v2 unchanged
            Case {
                word: 0x6e621c20,
                a: [0xaaaa_aaaa_aaaa_aaaa, 0],
                b: [0x5555_5555_5555_5555, 0],
                want: [0x5555_5555_5555_5555, 0],
            },
            // bit with a zero destination inserts the masked v1 bits
            Case {
                word: 0x6ea21c20,
                a: [0x0f0f_0f0f_0f0f_0f0f, 0],
                b: [0xaaaa_aaaa_aaaa_aaaa, 0],
                want: [0x0a0a_0a0a_0a0a_0a0a, 0],
            },
            // bif with a zero destination inserts the inverse-masked bits
            Case {
                word: 0x6ee21c20,
                a: [0x0f0f_0f0f_0f0f_0f0f, 0],
                b: [0xaaaa_aaaa_aaaa_aaaa, 0],
                want: [0x0505_0505_0505_0505, 0],
            },
            // shadd of two -1 bytes stays -1 (arithmetic halving)
            Case {
                word: 0x4e220420,
                a: [0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff],
                b: [0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff],
                want: [0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff],
            },
            // shsub of 0 - 1 halves to -1
            Case {
                word: 0x4e222420,
                a: [0, 0],
                b: [0x0101_0101_0101_0101, 0],
                want: [0xffff_ffff_ffff_ffff, 0],
            },
            // srhadd of 1 and 2 rounds to 2
            Case {
                word: 0x4e221420,
                a: [0x0101_0101_0101_0101, 0],
                b: [0x0202_0202_0202_0202, 0],
                want: [0x0202_0202_0202_0202, 0],
            },
            // sshl v0.8h: lane 0 shifts left by 1, lane 1 right by 1
            Case {
                word: 0x4e624420,
                a: [0x0008_0004_0002_0001, 0],
                b: [0x0000_00ff_0001_0001, 0],
                want: [0x0008_0002_0004_0002, 0],
            },
            // uqshl v0.8h saturates an overflow to the lane maximum
            Case {
                word: 0x6e624c20,
                a: [0x0000_0000_0000_8000, 0],
                b: [0x0000_0000_0000_0001, 0],
                want: [0x0000_0000_0000_ffff, 0],
            },
            // sqshl v0.8h saturates a positive overflow to the lane maximum
            Case {
                word: 0x4e624c20,
                a: [0x0000_0000_0000_4000, 0],
                b: [0x0000_0000_0000_0001, 0],
                want: [0x0000_0000_0000_7fff, 0],
            },
            // sqrshl v0.8h rounds before shifting right
            Case {
                word: 0x4e625c20,
                a: [0x0000_0000_0000_0003, 0],
                b: [0x0000_0000_0000_00ff, 0],
                want: [0x0000_0000_0000_0002, 0],
            },
        ];
        for case in cases {
            let block = compile(0x1000, &bytes(&[case.word])).unwrap();
            let mut cpu = Cpu::default();
            cpu.v[1] = case.a;
            cpu.v[2] = case.b;
            block.run(&mut cpu);
            assert_eq!(cpu.v[0], case.want, "{:08x}", case.word);
        }
    }
}
