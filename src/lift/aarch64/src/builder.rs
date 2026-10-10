use crate::{Environment, Writeback};
use core::cell::{Cell, RefCell};
use volt_ir::{
    function::{self as ir, BinOp as B, CmpOp as C, Function, Opcode, Value},
    types::{IntDesc, Type, TypeKind},
};
use volt_isa_aarch64::decode::{Addressing, Width};

/// Emits IR for one lifted function and knows where guest state lives.
///
/// The builder owns the [`Function`] under construction, the block currently
/// being filled, the interned scalar types, and the state-pointer parameter
/// (`fn(state: *mut u8) -> u64`). Guest registers are addressed through the
/// offsets the [`Environment`] reported when the builder was created.
///
/// Every method takes `&self`: emission order is the call order, and
/// [`Builder::set_block`] redirects subsequent emission, which is how an
/// environment builds fast and slow memory paths.
pub struct Builder {
    f: RefCell<Function>,
    /// Block currently being emitted into.
    block: Cell<ir::Block>,
    state: Value,
    /// 64-bit integer.
    pub i64: Type,
    /// 32-bit integer.
    pub i32: Type,
    /// 8-bit integer.
    pub i8: Type,
    /// One-bit comparison result.
    pub boolean: Type,
    /// Pointer; the type of the state pointer and of host addresses.
    pub ptr: Type,
    x: [usize; 31],
    sp: usize,
    flags: usize,
    v: [usize; 32],
}
impl Builder {
    pub(crate) fn new<E: Environment + ?Sized>(env: &E) -> Self {
        let mut f = Function::new();
        // A typical block is a few blocks and about a hundred instructions.
        f.reserve(8, 128, 128);
        let i64 = f.types.intern(TypeKind::Int(IntDesc { bits: 64 }));
        let i32 = f.types.intern(TypeKind::Int(IntDesc { bits: 32 }));
        let i8 = f.types.intern(TypeKind::Int(IntDesc { bits: 8 }));
        let boolean = f.types.intern(TypeKind::Bool);
        let ptr = f.types.ptr_global();
        let block = f.append_block();
        let state = f.append_block_param(block, ptr);
        Self {
            f: RefCell::new(f),
            block: Cell::new(block),
            state,
            i64,
            i32,
            i8,
            boolean,
            ptr,
            x: core::array::from_fn(|r| env.x_offset(r as u8)),
            sp: env.sp_offset(),
            flags: env.flags_offset(),
            v: core::array::from_fn(|r| env.v_offset(r as u8)),
        }
    }
    pub(crate) fn into_function(self) -> Function {
        self.f.into_inner()
    }

    // ---- guest state layout ----

    /// The state pointer parameter of the function being built.
    pub fn state(&self) -> Value {
        self.state
    }
    /// Offset of general register `r`: `X0..=X30` for `r < 31`, `SP` for `r == 31`.
    pub fn reg_at(&self, r: u8) -> usize {
        if r == 31 { self.sp } else { self.x[r as usize] }
    }
    /// Offset of the 32-bit NZCV word.
    pub fn flags_at(&self) -> usize {
        self.flags
    }
    /// Offset of the low doubleword of vector register `r`.
    pub fn v_at(&self, r: u8) -> usize {
        self.v[r as usize]
    }

    // ---- emission ----

    /// Append `op` producing a value of type `t` to the current block.
    pub fn emit(&self, t: Type, op: Opcode) -> Value {
        self.f.borrow_mut().append_inst(self.block.get(), t, op)
    }
    /// An integer constant of type `t`.
    pub fn k(&self, t: Type, n: u64) -> Value {
        self.emit(t, Opcode::Iconst(n as i64))
    }
    pub fn bin(&self, t: Type, op: B, a: Value, b: Value) -> Value {
        self.emit(t, Opcode::Arith(ir::Arith { op, lhs: a, rhs: b }))
    }
    pub fn imm(&self, t: Type, op: B, a: Value, n: u64) -> Value {
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
    /// Zero-extend or truncate `v` to `t`; `v` unchanged when it already has that type.
    pub fn cv(&self, t: Type, v: Value) -> Value {
        let from = self.f.borrow().value_type(v);
        if from == t {
            return v;
        }
        let kind = {
            let f = self.f.borrow();
            let bits = |ty| match f.types.type_kind(ty) {
                TypeKind::Int(d) => d.bits,
                _ => 1,
            };
            if bits(t) > bits(from) {
                ir::ConvertKind::Zext
            } else {
                ir::ConvertKind::Trunc
            }
        };
        self.emit(t, Opcode::Convert(ir::Convert { value: v, kind }))
    }
    pub fn cmp(&self, op: C, a: Value, b: Value) -> Value {
        self.emit(
            self.boolean,
            Opcode::Icmp(ir::Compare { op, lhs: a, rhs: b }),
        )
    }
    pub fn sel(&self, t: Type, c: Value, a: Value, b: Value) -> Value {
        self.emit(
            t,
            Opcode::Select(ir::Select {
                cond: c,
                then: a,
                else_: b,
            }),
        )
    }
    /// The interned integer type of `bits` bits.
    pub fn int(&self, bits: u16) -> Type {
        self.f
            .borrow_mut()
            .types
            .intern(TypeKind::Int(IntDesc { bits }))
    }
    pub(crate) fn ty(&self, w: Width) -> Type {
        if w == Width::X64 { self.i64 } else { self.i32 }
    }

    // ---- guest state access ----

    /// Pointer to `state + at`.
    pub fn addr(&self, at: usize) -> Value {
        self.imm(self.ptr, B::Add, self.state, at as u64)
    }
    pub fn load_ptr(&self, t: Type, p: Value) -> Value {
        self.emit(
            t,
            Opcode::Load(ir::Load {
                ptr: p,
                mem: ir::MemFlags::default(),
            }),
        )
    }
    /// Load a `t` from `state + at`.
    pub fn load(&self, t: Type, at: usize) -> Value {
        self.load_ptr(t, self.addr(at))
    }
    pub fn store_ptr(&self, p: Value, v: Value) {
        self.f.borrow_mut().append_store(self.block.get(), v, p)
    }
    /// Store `v` to `state + at`.
    pub fn store(&self, at: usize, v: Value) {
        self.store_ptr(self.addr(at), v)
    }
    /// Store the 8-bit constant `v` to `state + at`.
    pub fn byte(&self, at: usize, v: u64) {
        self.store(at, self.k(self.i8, v))
    }
    /// Read register `r` as a `t`. With `zero`, register 31 is `XZR` and reads
    /// as zero; otherwise it is `SP`.
    pub fn reg(&self, t: Type, r: u8, zero: bool) -> Value {
        if zero && r >= 31 {
            self.k(t, 0)
        } else {
            self.cv(t, self.load(self.i64, self.reg_at(r)))
        }
    }
    /// Write `X<r>`; a write to register 31 (`XZR`) is dropped.
    pub fn put(&self, r: u8, v: Value) {
        if r != 31 {
            self.put_sp(r, v)
        }
    }
    /// Write `X<r>`, or `SP` for register 31.
    pub fn put_sp(&self, r: u8, v: Value) {
        self.store(self.reg_at(r), self.cv(self.i64, v))
    }

    // ---- control flow ----

    /// Block currently being emitted into.
    pub fn block(&self) -> ir::Block {
        self.block.get()
    }
    /// Direct subsequent emission into `block`.
    pub fn set_block(&self, block: ir::Block) {
        self.block.set(block)
    }
    pub fn new_block(&self) -> ir::Block {
        self.f.borrow_mut().append_block()
    }
    /// End the current block with a conditional branch.
    pub fn branch_if(&self, c: Value, then: ir::Block, otherwise: ir::Block) {
        self.f.borrow_mut().append_if(
            self.block.get(),
            c,
            ir::EdgeDesc::bare(then),
            ir::EdgeDesc::bare(otherwise),
        )
    }
    /// End the current block by returning `next_pc` from the function. An
    /// environment's [`Environment::exit`] ends here; guest code never calls
    /// this directly.
    pub fn set_ret(&self, next_pc: Value) {
        self.f
            .borrow_mut()
            .set_terminator(self.block.get(), ir::Terminator::Ret(ir::Ret::one(next_pc)))
    }

    // ---- addressing ----

    /// Effective address of a load/store with base `rn`, plus the base-register
    /// update of a pre- or post-index mode.
    ///
    /// The update is returned instead of applied so each path can apply it its
    /// own way once the access has succeeded: an access that faults must leave
    /// the base register untouched so the retry sees it.
    pub fn address(&self, rn: u8, a: Addressing) -> (Value, Option<Writeback>) {
        let base = self.reg(self.i64, rn, false);
        match a {
            // The machines do not tell the unprivileged `ldtr`/`sttr` from a plain access.
            Addressing::Offset(n) | Addressing::Unscaled(n) | Addressing::Unprivileged(n) => {
                (self.imm(self.i64, B::Add, base, n as u64), None)
            }
            Addressing::PreIndex(n) => {
                let address = self.imm(self.i64, B::Add, base, n as u64);
                let writeback = Writeback {
                    register: rn,
                    value: address,
                    pre_index: true,
                };
                (address, Some(writeback))
            }
            Addressing::PostIndex(n) => {
                let writeback = Writeback {
                    register: rn,
                    value: self.imm(self.i64, B::Add, base, n as u64),
                    pre_index: false,
                };
                (base, Some(writeback))
            }
            Addressing::Register { rm, extend, amount } => {
                let v = self.extend(self.reg(self.i64, rm, true), extend);
                let address = self.bin(
                    self.i64,
                    B::Add,
                    base,
                    self.imm(self.i64, B::Shl, v, u64::from(amount.unwrap_or(0))),
                );
                (address, None)
            }
        }
    }
}
