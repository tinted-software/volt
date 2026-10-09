//! AArch64 instruction semantics lifted to [`volt_ir`], independent of any
//! particular machine.
//!
//! [`Lifter`] turns decoded [`Instruction`]s into IR over *guest state held in
//! memory*. Every lifted function has the signature
//!
//! ```text
//! fn(state: *mut u8) -> u64
//! ```
//!
//! It reads and writes guest registers at `state + offset`, and **returns the
//! next guest pc**. Which offsets, and everything that is not pure
//! architecture, belongs to the [`Environment`] the lifter is generic over:
//!
//! * **Layout.** [`Environment::x_offset`], [`sp_offset`](Environment::sp_offset),
//!   [`flags_offset`](Environment::flags_offset) and
//!   [`v_offset`](Environment::v_offset) say where `X0..=X30`, `SP`, the NZCV
//!   word (`u32`, flags in bits 31..28) and the 128-bit vector registers (two
//!   little-endian `u64`s) live. They are read once, when the [`Lifter`] is
//!   created.
//! * **Memory.** Loads, stores, pairs, literals, exclusives, atomics and SIMD
//!   structure accesses are handed to [`Environment::memory`] with the
//!   effective address already computed. The environment decides how guest
//!   memory is reached: a plain host load, a TLB probe, a call out, a trap.
//! * **Events.** System registers, `SVC`, `BRK`, `WFI`/`WFE`/`SEV`, barriers
//!   that need machine action, TLB maintenance, `ERET`, `DC ZVA`, `AT`,
//!   `CLREX` and cache operations go to [`Environment::event`].
//! * **Exits.** Every path that leaves the function (branches, fall-through
//!   at the end of a block, anything the environment ends itself) goes through
//!   [`Environment::exit`], which by default returns the next pc.
//!
//! Everything else (integer ALU, flags, shifts, bitfields, multiplies,
//! conditional select/compare, `ADR`, branches and the register-only SIMD
//! families) is architecture and is lifted here.
//!
//! The lifter lifts instructions; the *caller* owns the block: it decodes,
//! decides where a block ends (see [`Instruction::terminates_with`]), and
//! calls [`Lifter::exit`] when a block runs off its end without a
//! terminating instruction. An instruction that falls through leaves the
//! [`Builder`] positioned for the next one. An instruction that does not (a
//! branch, or a trap the environment ends with [`Environment::exit`]) has
//! already terminated the current block. An environment must therefore end the
//! block for every forwarded instruction for which `terminates_with` is true.

#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

mod builder;
mod int;
mod simd;
#[cfg(test)]
mod tests;

pub use builder::Builder;
use core::fmt;
use volt_ir::function::{BinOp as B, CmpOp as C, Function, Value};
use volt_target::aarch64::decode::*;

/// A load, store or other guest memory instruction handed to
/// [`Environment::memory`].
#[derive(Clone, Copy)]
pub struct MemoryAccess<'a> {
    /// The decoded instruction: `Memory`, `Pair`, `Literal`, `SimdLiteral`,
    /// `SimdMemory`, `SimdPair`, `Exclusive`, `Atomic` or `SimdStruct`.
    pub insn: &'a Instruction,
    /// The effective address, for the forms whose address is computed from the
    /// addressing mode: `Memory`, `Pair`, `SimdMemory`, `SimdPair` (the base
    /// for post-indexing, the updated base for pre-indexing), `Literal` and
    /// `SimdLiteral` (a constant). `None` for `Exclusive`, `Atomic` and
    /// `SimdStruct`, which address through a plain base register the
    /// environment reads itself.
    pub address: Option<Value>,
    /// Base-register update of a pre- or post-index mode. The lifter does not
    /// apply it: an access that can fault must leave the base untouched until
    /// it has succeeded, so the environment writes it (with
    /// [`Builder::put_sp`]) once it knows how.
    pub writeback: Option<Writeback>,
}

/// The base-register update of a pre- or post-indexed access.
#[derive(Clone, Copy, Debug)]
pub struct Writeback {
    /// The base register; 31 is `SP`.
    pub register: u8,
    /// Its new value.
    pub value: Value,
    /// Pre-indexed: the access address is `value` itself. Otherwise the access
    /// uses the old base and `value` is computed after it.
    pub pre_index: bool,
}

/// Why an instruction could not be lifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiftError {
    /// The [`Environment`] does not implement this instruction.
    Unsupported,
}
impl fmt::Display for LiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("instruction is not supported by the environment"),
        }
    }
}
impl core::error::Error for LiftError {}

/// What a machine provides to the lifter: where guest state lives, how memory
/// is reached, and what instructions with machine effects do. See the
/// [crate documentation](crate).
pub trait Environment {
    // ---- guest state layout (byte offsets from the state pointer) ----

    /// Offset of `X<r>` (`u64`), `r` in `0..=30`.
    fn x_offset(&self, r: u8) -> usize;
    /// Offset of the stack pointer (`u64`).
    fn sp_offset(&self) -> usize;
    /// Offset of the NZCV word (`u32`), flags in bits 31..28.
    fn flags_offset(&self) -> usize;
    /// Offset of vector register `V<r>` (16 bytes: two little-endian `u64`s), `r` in `0..=31`.
    fn v_offset(&self, r: u8) -> usize;

    // ---- hooks ----

    /// Called once per guest instruction before it is lifted. An emulator
    /// counts retired instructions here.
    fn begin_instruction(&mut self, b: &Builder) {
        let _ = b;
    }
    /// Called on every path that leaves the function with the next guest pc.
    /// The default ends the current block with `Ret(next_pc)`; an environment
    /// overrides it to do work on every exit first (for example flushing a
    /// pending cycle counter) and then calls [`Builder::set_ret`].
    fn exit(&mut self, b: &Builder, next_pc: Value) {
        b.set_ret(next_pc)
    }
    /// A load, store, exclusive, atomic or SIMD structure access at guest `pc`.
    /// The environment emits whatever reaches memory; see [`MemoryAccess`]
    /// for what the lifter has already computed. It must end the block (with
    /// [`Self::exit`]) for instructions that terminate it.
    fn memory(&mut self, b: &Builder, pc: u64, access: MemoryAccess<'_>) -> Result<(), LiftError>;
    /// An instruction whose effect belongs to the machine: `System`
    /// (`MRS`/`MSR`/`SYS`), `Svc`, `Brk`, `Wfi`, `Wfe`, `Sev`, `Sevl`, `Isb`,
    /// `Tlbi`, `Eret`, `DcZva`, `AddressTranslate`, `Clrex`, `Psci` and
    /// `CacheOp`. It receives the decoded instruction.
    fn event(&mut self, b: &Builder, pc: u64, insn: &Instruction) -> Result<(), LiftError>;
}

/// Lifts decoded AArch64 instructions into one [`Function`].
pub struct Lifter<E: Environment> {
    b: Builder,
    env: E,
}
impl<E: Environment> Lifter<E> {
    /// Start a function with the layout `env` reports.
    pub fn new(env: E) -> Self {
        Self {
            b: Builder::new(&env),
            env,
        }
    }
    /// The builder and the environment, for callers that emit IR of their own
    /// between instructions (a block driver handling words it does not lift).
    pub fn parts(&mut self) -> (&Builder, &mut E) {
        (&self.b, &mut self.env)
    }
    /// The function ends at `next_pc`: the block fell off its end without a
    /// terminating instruction.
    pub fn exit(&mut self, next_pc: u64) {
        let next = self.b.k(self.b.i64, next_pc);
        self.env.exit(&self.b, next)
    }
    /// Finish: the function built so far, and the environment back.
    pub fn into_parts(self) -> (Function, E) {
        (self.b.into_function(), self.env)
    }
    /// Lift one decoded instruction at guest address `pc`.
    pub fn lift(&mut self, pc: u64, insn: Instruction) -> Result<(), LiftError> {
        let (b, env) = (&self.b, &mut self.env);
        env.begin_instruction(b);
        match insn {
            Instruction::Movz(a) | Instruction::Movn(a) => {
                let t = b.ty(a.width);
                let bits = (a.immediate as u64) << a.shift;
                let value = if matches!(insn, Instruction::Movn(_)) {
                    !bits
                } else {
                    bits
                };
                if a.rd != 31 {
                    b.put(a.rd, b.k(t, value));
                }
            }
            Instruction::Movk(a) => {
                if a.rd != 31 {
                    let t = b.ty(a.width);
                    let old = b.reg(t, a.rd, false);
                    let kept = b.imm(t, B::BitAnd, old, !(0xffffu64 << a.shift));
                    b.put(
                        a.rd,
                        b.imm(t, B::BitOr, kept, (a.immediate as u64) << a.shift),
                    );
                }
            }
            Instruction::ArithImm(a) => {
                let t = b.ty(a.width);
                let lhs = b.reg(t, a.rn, false);
                let rhs = b.k(t, a.immediate as u64);
                let r = b.bin(t, int::arithmetic_op(a.op), lhs, rhs);
                if a.flags {
                    b.put(a.rd, r);
                    b.store(b.flags_at(), b.flags(a.width, a.op, lhs, rhs, r, None));
                } else {
                    b.put_sp(a.rd, r)
                }
            }
            Instruction::ArithReg(a) => {
                let t = b.ty(a.width);
                let lhs = b.reg(t, a.rn, true);
                let rhs = b.shift(a.width, b.reg(t, a.rm, true), a.shift, a.amount as u64);
                let r = b.bin(t, int::arithmetic_op(a.op), lhs, rhs);
                if a.rd != 31 {
                    b.put(a.rd, r)
                }
                if a.flags {
                    b.store(b.flags_at(), b.flags(a.width, a.op, lhs, rhs, r, None));
                }
            }
            Instruction::ArithExt(a) => {
                let t = b.ty(a.width);
                let lhs = b.reg(t, a.rn, false);
                let extended = b.extend(b.reg(b.i64, a.rm, true), a.extend);
                let rhs = b.cv(t, b.imm(b.i64, B::Shl, extended, a.amount as u64));
                let r = b.bin(t, int::arithmetic_op(a.op), lhs, rhs);
                if a.flags {
                    b.put(a.rd, r);
                    b.store(b.flags_at(), b.flags(a.width, a.op, lhs, rhs, r, None));
                } else {
                    b.put_sp(a.rd, r)
                }
            }
            Instruction::LogicReg(a) => {
                let t = b.ty(a.width);
                let lhs = b.reg(t, a.rn, true);
                let shifted = b.shift(a.width, b.reg(t, a.rm, true), a.shift, a.amount as u64);
                let rhs = if a.invert {
                    b.imm(t, B::BitXor, shifted, u64::MAX)
                } else {
                    shifted
                };
                let r = b.bin(t, int::logic_op(a.op), lhs, rhs);
                b.put(a.rd, r);
                if a.flags {
                    b.logic_flags(a.width, r)
                }
            }
            Instruction::LogicImm(a) => {
                let t = b.ty(a.width);
                let r = b.imm(t, int::logic_op(a.op), b.reg(t, a.rn, true), a.immediate);
                if a.flags {
                    b.put(a.rd, r);
                    b.logic_flags(a.width, r)
                } else {
                    b.put_sp(a.rd, r)
                }
            }
            Instruction::ArithCarry(a) => {
                let t = b.ty(a.width);
                let lhs = b.reg(t, a.rn, true);
                let loaded = b.reg(t, a.rm, true);
                let rhs = if a.op == ArithOp::Sub {
                    b.imm(t, B::BitXor, loaded, u64::MAX)
                } else {
                    loaded
                };
                let carry = b.cv(
                    t,
                    b.imm(
                        b.i32,
                        B::BitAnd,
                        b.imm(b.i32, B::Shr, b.load(b.i32, b.flags_at()), 29),
                        1,
                    ),
                );
                let partial = b.bin(t, B::Add, lhs, rhs);
                let r = b.bin(t, B::Add, partial, carry);
                if a.rd != 31 {
                    b.put(a.rd, r)
                }
                if a.flags {
                    let out = b.bin(
                        b.boolean,
                        B::BitOr,
                        b.cmp(C::Ult, partial, lhs),
                        b.cmp(C::Ult, r, partial),
                    );
                    b.store(
                        b.flags_at(),
                        b.flags(a.width, ArithOp::Add, lhs, rhs, r, Some(out)),
                    );
                }
            }
            Instruction::CondCompare(a) => {
                let t = b.ty(a.width);
                let lhs = b.reg(t, a.rn, true);
                let rhs = match a.operand {
                    CompareOperand::Register(r) => b.reg(t, r, true),
                    CompareOperand::Immediate(n) => b.k(t, n as u64),
                };
                let r = b.bin(t, int::arithmetic_op(a.op), lhs, rhs);
                let flags = b.flags(a.width, a.op, lhs, rhs, r, None);
                b.store(
                    b.flags_at(),
                    b.sel(
                        b.i32,
                        b.condition(a.cond),
                        flags,
                        b.k(b.i32, (a.nzcv as u64) << 28),
                    ),
                );
            }
            Instruction::Variable(a) => int::variable(b, a),
            Instruction::Bitfield(a) => int::bitfield(b, a),
            Instruction::Unary(a) => int::unary(b, a),
            Instruction::Extract(a) => {
                let t = b.ty(a.width);
                let low = b.reg(t, a.rm, true);
                let r = if a.lsb == 0 {
                    low
                } else {
                    b.bin(
                        t,
                        B::BitOr,
                        b.imm(t, B::Shr, low, a.lsb as u64),
                        b.imm(
                            t,
                            B::Shl,
                            b.reg(t, a.rn, true),
                            a.width as u64 - a.lsb as u64,
                        ),
                    )
                };
                if a.rd != 31 {
                    b.put(a.rd, r)
                }
            }
            Instruction::Mul(a) => {
                let t = b.ty(a.width);
                let p = b.bin(t, B::Mul, b.reg(t, a.rn, true), b.reg(t, a.rm, true));
                let r = b.bin(t, int::arithmetic_op(a.op), b.reg(t, a.ra, true), p);
                if a.rd != 31 {
                    b.put(a.rd, r)
                }
            }
            Instruction::MulLong(a) => {
                let e = if a.signed { Extend::Sxtw } else { Extend::Uxtw };
                let p = b.bin(
                    b.i64,
                    B::Mul,
                    b.extend(b.reg(b.i64, a.rn, true), e),
                    b.extend(b.reg(b.i64, a.rm, true), e),
                );
                let r = b.bin(b.i64, int::arithmetic_op(a.op), b.reg(b.i64, a.ra, true), p);
                if a.rd != 31 {
                    b.put(a.rd, r)
                }
            }
            Instruction::MulHigh(a) => int::high(b, a),
            Instruction::Adr(a) => {
                let here = if a.page { pc & 0xfffffffffffff000 } else { pc };
                b.put(a.rd, b.k(b.i64, here.wrapping_add(a.offset as u64)));
            }
            Instruction::Csel(a) => {
                let t = b.ty(a.width);
                let lhs = b.reg(t, a.rn, true);
                let rhs = b.reg(t, a.rm, true);
                let alt = match a.opc {
                    0 => rhs,
                    1 => b.imm(t, B::Add, rhs, 1),
                    2 => b.imm(t, B::BitXor, rhs, u64::MAX),
                    3 => b.bin(t, B::Sub, b.k(t, 0), rhs),
                    _ => unreachable!(),
                };
                if a.rd != 31 {
                    b.put(a.rd, b.sel(t, b.condition(a.cond), lhs, alt));
                }
            }
            Instruction::TestBranch(a) => {
                let t = b.ty(a.width);
                let v = b.reg(t, a.rt, true);
                let subject = if let Some(bit) = a.bit_index {
                    b.imm(t, B::BitAnd, v, 1u64 << bit)
                } else {
                    v
                };
                let taken = b.cmp(if a.negate { C::Ne } else { C::Eq }, subject, b.k(t, 0));
                branch(env, b, pc, a.offset, taken);
            }
            Instruction::BCond(a) => {
                let taken = b.condition(a.cond);
                branch(env, b, pc, a.offset, taken)
            }
            Instruction::B(offset) => {
                let target = b.k(b.i64, pc.wrapping_add(offset as u64));
                env.exit(b, target)
            }
            Instruction::Call(a) => {
                if a.link {
                    b.put(30, b.k(b.i64, pc.wrapping_add(4)))
                }
                let target = b.k(b.i64, pc.wrapping_add(a.target as u64));
                env.exit(b, target);
            }
            Instruction::Indirect(a) => {
                let target = b.reg(b.i64, a.rn, false);
                if a.link {
                    b.put(30, b.k(b.i64, pc.wrapping_add(4)))
                }
                env.exit(b, target);
            }

            Instruction::Memory(a) => {
                let (address, writeback) = b.address(a.rn, a.addressing);
                memory(env, b, pc, &insn, Some(address), writeback)?
            }
            Instruction::Pair(a) => {
                let (address, writeback) = b.address(a.rn, a.addressing);
                memory(env, b, pc, &insn, Some(address), writeback)?
            }
            Instruction::SimdMemory(a) => {
                let (address, writeback) = b.address(a.rn, a.addressing);
                memory(env, b, pc, &insn, Some(address), writeback)?
            }
            Instruction::SimdPair(a) => {
                let (address, writeback) = b.address(a.rn, a.addressing);
                memory(env, b, pc, &insn, Some(address), writeback)?
            }
            Instruction::Literal(a) => {
                let address = b.k(b.i64, pc.wrapping_add(a.offset as u64));
                memory(env, b, pc, &insn, Some(address), None)?
            }
            Instruction::SimdLiteral(a) => {
                let address = b.k(b.i64, pc.wrapping_add(a.offset as u64));
                memory(env, b, pc, &insn, Some(address), None)?
            }
            Instruction::Exclusive(_) | Instruction::Atomic(_) | Instruction::SimdStruct(_) => {
                memory(env, b, pc, &insn, None, None)?
            }
            Instruction::SimdImm(a) => simd::imm(b, a),
            Instruction::SimdMove { rd, rn } => simd::mov(b, rd, rn),
            Instruction::SimdInsert(a) => simd::insert(b, a),
            Instruction::SimdInsertElement(a) => simd::insert_element(b, a),
            Instruction::SimdExtract(a) => simd::extract(b, a),
            Instruction::SimdDupElement(a) => simd::dup_element(b, a),
            Instruction::SimdDup { rd, rn, esize, q } => simd::dup(b, rd, rn, esize, q),
            Instruction::SimdExt(a) => simd::ext(b, a),
            Instruction::SimdCompareZero(a) => simd::compare_zero(b, a),
            Instruction::SimdTable(a) => simd::table(b, a),
            Instruction::SimdFmov(a) => simd::fmov(b, a),
            Instruction::SimdAlu(a) => simd::alu(b, a),
            // Architecturally no-ops: nothing to emit.
            Instruction::Nop
            | Instruction::Hint(_)
            | Instruction::Prefetch(_)
            | Instruction::PrefetchLiteral(_)
            | Instruction::Dsb(_)
            | Instruction::Dmb(_)
            | Instruction::Yield => {}
            Instruction::System(_)
            | Instruction::Svc(_)
            | Instruction::Brk(_)
            | Instruction::Wfi
            | Instruction::Wfe
            | Instruction::Sev
            | Instruction::Sevl
            | Instruction::Isb(_)
            | Instruction::Tlbi(_)
            | Instruction::Eret
            | Instruction::DcZva(_)
            | Instruction::AddressTranslate(_)
            | Instruction::Clrex(_)
            | Instruction::Psci
            | Instruction::CacheOp(_) => env.event(b, pc, &insn)?,
        }
        Ok(())
    }
}
fn memory<E: Environment>(
    env: &mut E,
    b: &Builder,
    pc: u64,
    insn: &Instruction,
    address: Option<Value>,
    writeback: Option<Writeback>,
) -> Result<(), LiftError> {
    env.memory(
        b,
        pc,
        MemoryAccess {
            insn,
            address,
            writeback,
        },
    )
}
/// End the function at `pc + offset` when `taken`, otherwise at `pc + 4`.
fn branch<E: Environment>(env: &mut E, b: &Builder, pc: u64, offset: i64, taken: Value) {
    let next = b.sel(
        b.i64,
        taken,
        b.k(b.i64, pc.wrapping_add(offset as u64)),
        b.k(b.i64, pc.wrapping_add(4)),
    );
    env.exit(b, next)
}
