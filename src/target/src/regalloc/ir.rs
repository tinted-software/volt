//! Adapter over the `volt-ir` function query API.
//!
//! The allocator is written against the [`Func`] trait so unit tests can use a
//! lightweight fake. The real implementation forwards to
//! `volt_ir::function::Function`.

pub use volt_ir::function::{
    Block, Function, Inst, Jump, Opcode, Ret, Terminator, Value, ValueList,
};
pub use volt_ir::types::Type;

pub trait Func {
    fn block_count(&self) -> usize;
    fn value_count(&self) -> usize;
    fn block_insts(&self, block: Block) -> &[Inst];
    fn block_params(&self, block: Block) -> &[Value];
    fn terminator(&self, block: Block) -> Option<Terminator>;
    fn opcode(&self, inst: Inst) -> Opcode;
    fn defining_inst(&self, value: Value) -> Option<Inst>;
    fn inst_result(&self, inst: Inst) -> Option<Value>;
    fn value_list(&self, list: ValueList) -> &[Value];
    fn block_args(&self, jump: Jump) -> &[Value];
    fn value_type(&self, value: Value) -> Type;
}

impl Func for Function {
    fn block_count(&self) -> usize {
        self.block_count()
    }
    fn value_count(&self) -> usize {
        self.value_count()
    }
    fn block_insts(&self, block: Block) -> &[Inst] {
        self.block_insts(block)
    }
    fn block_params(&self, block: Block) -> &[Value] {
        self.block_params(block)
    }
    fn terminator(&self, block: Block) -> Option<Terminator> {
        self.terminator(block)
    }
    fn opcode(&self, inst: Inst) -> Opcode {
        self.opcode(inst)
    }
    fn defining_inst(&self, value: Value) -> Option<Inst> {
        self.defining_inst(value)
    }
    fn inst_result(&self, inst: Inst) -> Option<Value> {
        self.inst_result(inst)
    }
    fn value_list(&self, list: ValueList) -> &[Value] {
        self.value_list(list)
    }
    fn block_args(&self, jump: Jump) -> &[Value] {
        self.block_args(jump)
    }
    fn value_type(&self, value: Value) -> Type {
        self.value_type(value)
    }
}

/// Visit every operand value of `inst`, calling `f(value, is_edge_arg)`.
/// Edge arguments (values passed to a successor's block parameters by an
/// `If`) are flagged so callers can treat them as uses in the predecessor.
pub fn visit_inst_operands<F: Func>(func: &F, inst: Inst, f: &mut impl FnMut(Value, bool)) {
    match func.opcode(inst) {
        Opcode::AtomicRmw(a) => {
            f(a.ptr, false);
            f(a.value, false);
            if let Some(c) = a.compare {
                f(c, false);
            }
        }
        Opcode::Iconst(_)
        | Opcode::Fconst(_)
        | Opcode::Fconst128(_)
        | Opcode::Alloca(_)
        | Opcode::GlobalAddr(_)
        | Opcode::Barrier(_) => {}
        Opcode::Arith(a) => {
            f(a.lhs, false);
            f(a.rhs, false);
        }
        Opcode::ArithImm(a) => f(a.lhs, false),
        Opcode::Icmp(c) => {
            f(c.lhs, false);
            f(c.rhs, false);
        }
        Opcode::Select(s) => {
            f(s.cond, false);
            f(s.then, false);
            f(s.else_, false);
        }
        Opcode::Extract(e) => f(e.aggregate, false),
        Opcode::Convert(c) => f(c.value, false),
        Opcode::DecodeLowFloat(c) | Opcode::EncodeLowFloat(c) => f(c.value, false),
        Opcode::DequantizeNvfp4(c) | Opcode::QuantizeNvfp4(c) => {
            f(c.value, false);
            f(c.block_scale, false);
            f(c.global_scale, false);
        }
        Opcode::Unary(u) => f(u.value, false),
        Opcode::Load(l) => f(l.ptr, false),
        Opcode::Store(s) => {
            f(s.value, false);
            f(s.ptr, false);
        }
        Opcode::Prefetch(p) => f(p.ptr, false),
        Opcode::VaStart(v) => f(v.list, false),
        Opcode::VaArg(v) => f(v.list, false),
        Opcode::VaEnd(v) => f(v.list, false),
        Opcode::Dot(d) => {
            f(d.acc, false);
            f(d.a, false);
            f(d.b, false);
        }
        Opcode::Reduce(r) => f(r.vector, false),
        Opcode::Splat(s) => f(s.scalar, false),
        Opcode::Matmul(m) => {
            f(m.a, false);
            f(m.b, false);
            f(m.c, false);
        }
        Opcode::StructNew(s) => {
            for v in func.value_list(s.fields) {
                f(*v, false);
            }
        }
        Opcode::Call(c) => {
            for a in func.value_list(c.args) {
                f(*a, false);
            }
            if let Some(rd) = c.ret_dest {
                f(rd, false);
            }
        }
        Opcode::CallIndirect(c) => {
            f(c.target, false);
            for a in func.value_list(c.args) {
                f(*a, false);
            }
            if let Some(rd) = c.ret_dest {
                f(rd, false);
            }
        }
        Opcode::If(cf) => {
            f(cf.cond, false);
            for a in func.block_args(cf.then) {
                f(*a, true);
            }
            for a in func.block_args(cf.else_) {
                f(*a, true);
            }
        }
    }
}

/// Visit every operand value of a terminator. A `Ret` value is an ordinary
/// operand; a `Jump`'s arguments are edge arguments.
pub fn visit_term_operands<F: Func>(func: &F, term: &Terminator, f: &mut impl FnMut(Value, bool)) {
    match term {
        Terminator::Ret(r) => {
            for v in r.slice() {
                f(*v, false);
            }
        }
        Terminator::Jump(j) => {
            for a in func.block_args(*j) {
                f(*a, true);
            }
        }
    }
}
