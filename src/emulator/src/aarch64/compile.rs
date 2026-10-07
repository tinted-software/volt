use super::{
    cpu::{Cpu, DTLB_ENTRIES, Exclusive as CpuExclusive, System as CpuSystem, Trap},
    decode::*,
};
use crate::memory::MemoryError;
use std::{
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
    Decode(DecodeError),
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
            Self::Decode(e) => write!(f, "{e}"),
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
impl std::error::Error for Error {}
impl From<MemoryError> for Error {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}
impl From<DecodeError> for Error {
    fn from(e: DecodeError) -> Self {
        Self::Decode(e)
    }
}
impl From<native::Error> for Error {
    fn from(e: native::Error) -> Self {
        Self::Native(e)
    }
}
pub struct Block {
    image: native::JittedFunction,
}
impl Block {
    pub fn run(&self, cpu: &mut Cpu) -> u64 {
        cpu.trap = Trap::None;
        let entry: unsafe extern "C" fn(*mut Cpu) -> u64 = unsafe { self.image.entry() };
        let next = unsafe { entry(cpu) };
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
}
impl Lower {
    fn new() -> Self {
        let mut f = Function::new();
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
        self.bin(t, op, a, self.k(t, n))
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
        self.bin(self.ptr, B::Add, self.cpu, self.k(self.u64, at as u64))
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
    fn ret(&self, v: Value) {
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
    /// Like `address`, but a post-index writeback is returned instead of being
    /// scheduled for the machine, so each path can apply it its own way.
    fn address_split(&self, rn: u8, a: Addressing) -> (Value, Option<(u8, Value)>) {
        match a {
            Addressing::PostIndex(n) => {
                let base = self.reg(self.u64, rn, false);
                (base, Some((rn, self.imm(self.u64, B::Add, base, n as u64))))
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
        let entry = self.bin(
            self.ptr,
            B::Add,
            self.cpu,
            self.imm(self.u64, B::Add, index, offset_of!(Cpu, dtlb) as u64),
        );
        let field = |at: u64| {
            self.load_ptr(
                self.u64,
                self.bin(self.ptr, B::Add, entry, self.k(self.u64, at)),
            )
        };
        let tag = field(if write { 8 } else { 0 });
        let addend = field(16);
        let want = self.imm(self.u64, B::BitAnd, address, !0xfffu64 | (n - 1));
        let hit = self.cmp(C::Eq, tag, want);
        let host = self.cv(self.ptr, self.bin(self.u64, B::Add, address, addend));
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
    fn trap(&self, pc: u64, t: Trap) {
        self.byte(offset_of!(Cpu, trap), t as u64);
        self.ret(self.k(self.u64, pc))
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
        let counter_at = offset_of!(Cpu, system) + offset_of!(CpuSystem, cntvct_el0);
        l.store(
            counter_at,
            l.imm(l.u64, B::Add, l.load(l.u64, counter_at), 1),
        );
        let instruction = decode(u32::from_le_bytes(chunk.try_into().unwrap()))?;
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
            Instruction::Pair(a) => {
                let address = l.address(a.rn, a.addressing);
                setup_memory(&l, address, a.size, a.rt, a.signed, a.op);
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
                l.trap(
                    pc.wrapping_add(4),
                    if a.op == MemoryOp::Store {
                        Trap::Store
                    } else {
                        Trap::Load
                    },
                );
            }
            Instruction::Clrex => l.byte(offset_of!(Cpu, monitor_valid), 0),
            Instruction::Sev | Instruction::Sevl => l.byte(offset_of!(Cpu, event_set), 1),
            Instruction::Nop
            | Instruction::CacheOp
            | Instruction::Dsb
            | Instruction::Dmb
            | Instruction::Yield => {}
            Instruction::Wfe => l.trap(pc.wrapping_add(4), Trap::Wfe),
            Instruction::Trap => l.trap(pc, Trap::Brk),
            Instruction::Isb | Instruction::Tlbi => l.trap(pc.wrapping_add(4), Trap::Sync),
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
    Ok(Block {
        image: native::compile(&l.f.into_inner())?,
    })
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
        MdscrEl1 => offset_of!(CpuSystem, mdscr_el1),
        CntkctlEl1 => offset_of!(CpuSystem, cntkctl_el1),
        OslarEl1 => offset_of!(CpuSystem, oslar_el1),
        OsdlrEl1 => offset_of!(CpuSystem, osdlr_el1),
        Fpcr => offset_of!(CpuSystem, fpcr),
        Fpsr => offset_of!(CpuSystem, fpsr),
        Daif => offset_of!(CpuSystem, daif),
        _ => return None,
    })
}
fn lower_system(l: &Lower, a: System) {
    use SystemRegister::*;
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
