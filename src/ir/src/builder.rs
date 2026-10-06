extern crate alloc;

use super::function::{BinOp, Block, CmpOp, EdgeDesc, Function, Value};
use super::types::Type;

pub struct Builder<'a> {
    func: &'a mut Function,
    block: Block,
}

impl<'a> Builder<'a> {
    pub fn new(func: &'a mut Function, block: Block) -> Self {
        Self { func, block }
    }

    pub fn switch_to(&mut self, block: Block) {
        self.block = block;
    }

    pub fn iconst(&mut self, ty: Type, value: i64) -> Value {
        self.func
            .append_inst(self.block, ty, super::function::Opcode::Iconst(value))
    }

    pub fn arith(&mut self, op: BinOp, lhs: Value, rhs: Value) -> Value {
        let ty = self.func.value_type(lhs);
        self.func.append_inst(
            self.block,
            ty,
            super::function::Opcode::Arith(super::function::Arith { op, lhs, rhs }),
        )
    }

    pub fn iadd(&mut self, lhs: Value, rhs: Value) -> Value {
        self.arith(BinOp::Add, lhs, rhs)
    }

    pub fn icmp(&mut self, op: CmpOp, lhs: Value, rhs: Value) -> Value {
        let bool_t = self.func.types.intern(super::types::TypeKind::Bool);
        self.func.append_inst(
            self.block,
            bool_t,
            super::function::Opcode::Icmp(super::function::Compare { op, lhs, rhs }),
        )
    }

    pub fn select(&mut self, cond: Value, then_value: Value, else_value: Value) -> Value {
        let ty = self.func.value_type(then_value);
        self.func.append_inst(
            self.block,
            ty,
            super::function::Opcode::Select(super::function::Select {
                cond,
                then: then_value,
                else_: else_value,
            }),
        )
    }

    pub fn struct_new(&mut self, fields: &[Value]) -> Value {
        let tys: alloc::vec::Vec<Type> = fields.iter().map(|f| self.func.value_type(*f)).collect();
        let st = self.func.types.intern(super::types::TypeKind::Struct(tys));
        self.func.append_struct_new(self.block, st, fields)
    }

    pub fn extract(&mut self, aggregate: Value, index: u32) -> Value {
        let field_ty = match self.func.types.type_kind(self.func.value_type(aggregate)) {
            super::types::TypeKind::Struct(fields) => fields[index as usize],
            _ => panic!("builder.extract: aggregate must be a struct"),
        };
        self.func.append_inst(
            self.block,
            field_ty,
            super::function::Opcode::Extract(super::function::Extract { aggregate, index }),
        )
    }

    pub fn load(&mut self, ty: Type, ptr: Value) -> Value {
        self.func.append_inst(
            self.block,
            ty,
            super::function::Opcode::Load(super::function::Load {
                ptr,
                volatile: false,
            }),
        )
    }

    pub fn store(&mut self, value: Value, ptr: Value) {
        self.func.append_store(self.block, value, ptr);
    }

    pub fn ret(&mut self, value: Option<Value>) {
        let r = match value {
            Some(v) => super::function::Ret::one(v),
            None => super::function::Ret::none(),
        };
        self.func
            .set_terminator(self.block, super::function::Terminator::Ret(r));
    }

    pub fn ret_values(&mut self, vals: &[Value]) {
        self.func.set_terminator(
            self.block,
            super::function::Terminator::Ret(super::function::Ret::many(vals)),
        );
    }

    pub fn jump(&mut self, target: Block, args: &[Value]) {
        self.func.set_jump(self.block, target, args);
    }

    pub fn if_(&mut self, cond: Value, then_edge: EdgeDesc, else_edge: EdgeDesc) {
        self.func.append_if(self.block, cond, then_edge, else_edge);
    }
}

#[cfg(test)]
mod tests {
    use super::super::function::{BinOp, CmpOp, Function};
    use super::super::types::{IntDesc, TypeKind};
    use super::*;

    fn i32t(f: &mut Function) -> Type {
        f.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }))
    }

    #[test]
    fn appends_to_current_block() {
        let mut func = Function::new();
        let i32_t = i32t(&mut func);
        let entry = func.append_block();
        let mut b = Builder::new(&mut func, entry);
        let x = b.iconst(i32_t, 10);
        let y = b.iconst(i32_t, 20);
        let sum = b.iadd(x, y);
        assert_eq!(i32_t, func.value_type(sum));
        let op = func.opcode(func.defining_inst(sum).unwrap());
        match op {
            super::super::function::Opcode::Arith(a) => {
                assert_eq!(x, a.lhs);
                assert_eq!(y, a.rhs);
            }
            _ => panic!("expected arith"),
        }
    }

    #[test]
    fn arith_any_operator() {
        let mut func = Function::new();
        let i32_t = i32t(&mut func);
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        let mut bld = Builder::new(&mut func, entry);
        let d = bld.arith(BinOp::Mul, a, b);
        assert_eq!(i32_t, func.value_type(d));
        match func.opcode(func.defining_inst(d).unwrap()) {
            super::super::function::Opcode::Arith(x) => assert_eq!(BinOp::Mul, x.op),
            _ => panic!("expected arith"),
        }
    }

    #[test]
    fn comparison_produces_bool() {
        let mut func = Function::new();
        let i32_t = i32t(&mut func);
        let bool_t = func.types.intern(TypeKind::Bool);
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        let mut bld = Builder::new(&mut func, entry);
        let c = bld.icmp(CmpOp::Gt, a, b);
        assert_eq!(bool_t, func.value_type(c));
    }

    #[test]
    fn struct_construction_and_extraction() {
        let mut func = Function::new();
        let i32_t = i32t(&mut func);
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        let mut bld = Builder::new(&mut func, entry);
        let s = bld.struct_new(&[a, b]);
        let f0 = bld.extract(s, 0);
        assert_eq!(i32_t, func.value_type(f0));
    }

    #[test]
    fn loads_and_stores() {
        let mut func = Function::new();
        let i32_t = i32t(&mut func);
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let mut bld = Builder::new(&mut func, entry);
        let v = bld.load(i32_t, p);
        bld.store(v, p);
        assert_eq!(i32_t, func.value_type(v));
        let insts = func.block_insts(entry).to_vec();
        match func.opcode(*insts.last().unwrap()) {
            super::super::function::Opcode::Store(st) => assert_eq!(p, st.ptr),
            _ => panic!("expected store"),
        }
    }

    #[test]
    fn select_build() {
        let mut func = Function::new();
        let i32_t = i32t(&mut func);
        let bool_t = func.types.intern(TypeKind::Bool);
        let entry = func.append_block();
        let cond = func.append_block_param(entry, bool_t);
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        let mut bld = Builder::new(&mut func, entry);
        let c = bld.select(cond, a, b);
        assert_eq!(i32_t, func.value_type(c));
    }

    #[test]
    fn branching_function() {
        let mut func = Function::new();
        let i32_t = i32t(&mut func);
        let bool_t = func.types.intern(TypeKind::Bool);
        let entry = func.append_block();
        let cond = func.append_block_param(entry, bool_t);
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        let then_b = func.append_block();
        let else_b = func.append_block();
        let mut b_ = Builder::new(&mut func, entry);
        b_.if_(
            cond,
            EdgeDesc {
                target: then_b,
                args: alloc::vec::Vec::new(),
            },
            EdgeDesc {
                target: else_b,
                args: alloc::vec::Vec::new(),
            },
        );
        b_.switch_to(then_b);
        b_.ret(Some(a));
        b_.switch_to(else_b);
        b_.ret(Some(b));
        let insts = func.block_insts(entry).to_vec();
        match func.opcode(*insts.last().unwrap()) {
            super::super::function::Opcode::If(cf) => assert_eq!(cond, cf.cond),
            _ => panic!("expected if"),
        }
        match func.terminator(then_b).unwrap() {
            super::super::function::Terminator::Ret(r) => {
                assert_eq!(1, r.count);
                assert_eq!(a, r.values[0]);
            }
            _ => panic!("expected ret"),
        }
    }
}
