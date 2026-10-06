//! Function-level container, builder, and legacy IR bridge over the `volt` dialect.

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use std::collections::HashMap;

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{OneRegionInterface, SymbolOpInterface},
        ops::FuncOp,
        types::FunctionType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    printable::Printable,
    region::Region,
    r#type::{TypeHandle, Typed},
    value::Value,
};

use super::attrs::{CallInfo, FuncInfo, FuncInfoAttr, MatMulDesc, QuantDesc};
use super::ops::{
    AllocaOp, ArithImmOp, ArithOp, AtomicRmwOp, BarrierOp, CallIndirectOp, CallOp, CondBrOp,
    ConstF128Op, ConstFloatOp, ConstIntOp, ConvertOp, DecodeLowFloatOp, DequantizeNvfp4Op, DotOp,
    EncodeLowFloatOp, ExtractOp, GlobalAddrOp, IcmpOp, JumpOp, LoadOp, MatMulOp, PrefetchOp,
    QuantizeNvfp4Op, ReduceOp, RetOp, SelectOp, SplatOp, StoreOp, StructNewOp, UnaryOp, VaArgOp,
    VaEndOp, VaStartOp,
};
use super::types::{
    ArrayType, BoolType, FloatType, IntType, PtrType, SliceType, StructType, VectorType, VoltKind,
    describe,
};
use crate::function::{
    AtomicOp, AtomicOrdering, AtomicScope, BarrierScope, BinOp, CmpOp, RetPiece,
};
use crate::low_float::Format as LowFloatFormat;
use crate::nvfp4::ScaleApplication;

/// Error encountered during conversion between pliron IR and legacy IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConvertError {
    UnsupportedOp(String),
    UnsupportedType(String),
    MissingValue,
    MissingBlock,
    InvalidSignature,
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOp(op) => write!(f, "unsupported operation: {op}"),
            Self::UnsupportedType(ty) => write!(f, "unsupported type: {ty}"),
            Self::MissingValue => f.write_str("missing SSA value during lowering"),
            Self::MissingBlock => f.write_str("missing block target during lowering"),
            Self::InvalidSignature => f.write_str("invalid function signature"),
        }
    }
}

impl core::error::Error for ConvertError {}

/// A function container over the Volt dialect.
pub struct Function {
    pub ctx: Context,
    pub func: FuncOp,
}

impl Function {
    /// Create a new function with an owned context, entry block, and typed arguments.
    pub fn new(name: &str, param_types: Vec<TypeHandle>, return_types: Vec<TypeHandle>) -> Self {
        let mut ctx = Context::new();
        let func_ty = FunctionType::get(&mut ctx, param_types, return_types);
        let id = Identifier::try_from(name).expect("valid function name");
        let func = FuncOp::new(&mut ctx, id, func_ty);
        Self { ctx, func }
    }

    /// Create a new function within an existing context.
    pub fn with_context(
        mut ctx: Context,
        name: &str,
        param_types: Vec<TypeHandle>,
        return_types: Vec<TypeHandle>,
    ) -> Self {
        let func_ty = FunctionType::get(&mut ctx, param_types, return_types);
        let id = Identifier::try_from(name).expect("valid function name");
        let func = FuncOp::new(&mut ctx, id, func_ty);
        Self { ctx, func }
    }

    /// Wrap an existing [`FuncOp`] and its context.
    pub fn from_func_op(ctx: Context, func: FuncOp) -> Self {
        Self { ctx, func }
    }

    /// Reference to the underlying context.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Mutable reference to the underlying context.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }

    /// The function's [`FuncOp`].
    pub fn func_op(&self) -> FuncOp {
        self.func
    }

    /// The function symbol name.
    pub fn name(&self) -> Identifier {
        self.func.get_symbol_name(&self.ctx)
    }

    /// The entry basic block.
    pub fn entry_block(&self) -> Ptr<BasicBlock> {
        self.func.get_entry_block(&self.ctx)
    }

    /// The single region of the function.
    pub fn region(&self) -> Ptr<Region> {
        self.func.get_region(&self.ctx)
    }

    /// All basic blocks in the function in order.
    pub fn blocks(&self) -> Vec<Ptr<BasicBlock>> {
        self.region().deref(&self.ctx).iter(&self.ctx).collect()
    }

    /// Append a new basic block to the function with optional label and arguments.
    pub fn append_block(
        &mut self,
        label: Option<&str>,
        arg_types: Vec<TypeHandle>,
    ) -> Ptr<BasicBlock> {
        let id = label.map(|s| Identifier::try_from(s).expect("valid block label"));
        let bb = BasicBlock::new(&mut self.ctx, id, arg_types);
        let r = self.region();
        bb.insert_at_back(r, &mut self.ctx);
        bb
    }

    /// Obtain an instruction builder positioned at the end of the given block.
    pub fn builder(&mut self, block: Ptr<BasicBlock>) -> Builder<'_> {
        Builder::new(&mut self.ctx, block)
    }

    /// Set function-level information flags.
    pub fn set_func_info(&mut self, info: FuncInfo) {
        let key = pliron::ident!("volt_func_info");
        self.func
            .get_operation()
            .deref_mut(&mut self.ctx)
            .attributes
            .set(key, FuncInfoAttr::new(info));
    }

    /// Retrieve function-level information flags, if present.
    pub fn func_info(&self) -> Option<FuncInfo> {
        let key = pliron::ident!("volt_func_info");
        self.func
            .get_operation()
            .deref(&self.ctx)
            .attributes
            .get::<FuncInfoAttr>(&key)
            .map(|a| a.info())
    }

    /// Verify the function.
    pub fn verify(&self) -> Result<(), pliron::result::Error> {
        self.func.get_operation().deref(&self.ctx).verify(&self.ctx)
    }

    /// Print the function to a string.
    pub fn print(&self) -> String {
        alloc::format!(
            "{}",
            self.func
                .get_operation()
                .deref(&self.ctx)
                .print(&self.ctx, &Default::default())
        )
    }

    /// Convert a legacy [`crate::function::Function`] into a pliron [`Function`].
    pub fn from_legacy(legacy: &crate::function::Function) -> Self {
        let mut ctx = Context::new();

        // Convert all types up-front.
        let type_map: Vec<TypeHandle> = (0..legacy.types.count())
            .map(|i| {
                let ty = crate::types::Type(i as u32);
                convert_legacy_type_to_pliron(&mut ctx, &legacy.types, ty)
            })
            .collect();

        // Determine input types and return types from the entry block and return statements.
        let (param_types, ret_types) = if legacy.block_count() > 0 {
            let entry_params = legacy.block_params(crate::function::Block(0));
            let params = entry_params
                .iter()
                .map(|&p| type_map[legacy.value_type(p).0 as usize])
                .collect();

            // Find first return terminator to determine return types.
            let mut rets = Vec::new();
            for b in 0..legacy.block_count() {
                if let Some(crate::function::Terminator::Ret(r)) =
                    legacy.terminator(crate::function::Block(b as u32))
                {
                    rets = r
                        .slice()
                        .iter()
                        .map(|&v| type_map[legacy.value_type(v).0 as usize])
                        .collect();
                    break;
                }
            }
            (params, rets)
        } else {
            (Vec::new(), Vec::new())
        };

        let func_ty = FunctionType::get(&mut ctx, param_types, ret_types);
        let func = FuncOp::new(&mut ctx, pliron::ident!("func"), func_ty);
        let region = func.get_region(&ctx);

        if legacy.block_count() == 0 {
            let mut res = Self { ctx, func };
            res.set_func_info(FuncInfo {
                is_variadic: legacy.is_variadic,
                num_fixed_params: legacy.num_fixed_params,
                sret: legacy.sret,
                is_local: legacy.is_local,
            });
            return res;
        }

        // Map blocks: block 0 is the entry block, blocks 1..N are created.
        let mut blocks = Vec::with_capacity(legacy.block_count());
        let entry_bb = func.get_entry_block(&ctx);
        blocks.push(entry_bb);

        for b in 1..legacy.block_count() {
            let param_tys: Vec<TypeHandle> = legacy
                .block_params(crate::function::Block(b as u32))
                .iter()
                .map(|&p| type_map[legacy.value_type(p).0 as usize])
                .collect();
            let label = Identifier::try_from(alloc::format!("block{b}")).expect("valid block name");
            let bb = BasicBlock::new(&mut ctx, Some(label), param_tys);
            bb.insert_at_back(region, &mut ctx);
            blocks.push(bb);
        }

        // Map values (indexed by legacy Value index).
        let mut val_map: Vec<Option<Value>> = alloc::vec![None; legacy.value_count()];

        // Map entry block parameters.
        for (i, &p) in legacy
            .block_params(crate::function::Block(0))
            .iter()
            .enumerate()
        {
            val_map[p.index()] = Some(entry_bb.deref(&ctx).get_argument(i));
        }

        // Map other block parameters.
        for b in 1..legacy.block_count() {
            let bb = blocks[b];
            for (i, &p) in legacy
                .block_params(crate::function::Block(b as u32))
                .iter()
                .enumerate()
            {
                val_map[p.index()] = Some(bb.deref(&ctx).get_argument(i));
            }
        }

        // Translate instructions per block.
        for b in 0..legacy.block_count() {
            let bb = blocks[b];
            let legacy_b = crate::function::Block(b as u32);

            for inst in legacy.block_insts(legacy_b) {
                let op = legacy.opcode(*inst);
                let pliron_ty = legacy
                    .inst_result(*inst)
                    .map(|r| type_map[legacy.value_type(r).0 as usize])
                    .unwrap_or_else(|| BoolType::get_type(&ctx));

                match op {
                    crate::function::Opcode::Iconst(val) => {
                        let op = ConstIntOp::new(&mut ctx, val, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Fconst(val) => {
                        let op = ConstFloatOp::new(&mut ctx, val, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Fconst128(val) => {
                        let op = ConstF128Op::new(&mut ctx, val, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Arith(a) => {
                        let lhs = val_map[a.lhs.index()].expect("operand");
                        let rhs = val_map[a.rhs.index()].expect("operand");
                        let op = ArithOp::new(&mut ctx, a.op, lhs, rhs, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::ArithImm(a) => {
                        let lhs = val_map[a.lhs.index()].expect("operand");
                        let op = ArithImmOp::new(&mut ctx, a.op, lhs, a.imm, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Icmp(c) => {
                        let lhs = val_map[c.lhs.index()].expect("operand");
                        let rhs = val_map[c.rhs.index()].expect("operand");
                        let op = IcmpOp::new(&mut ctx, c.op, lhs, rhs, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Select(s) => {
                        let cond = val_map[s.cond.index()].expect("operand");
                        let t = val_map[s.then.index()].expect("operand");
                        let e = val_map[s.else_.index()].expect("operand");
                        let op = SelectOp::new(&mut ctx, cond, t, e, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::StructNew(s) => {
                        let fields: Vec<_> = legacy
                            .value_list(s.fields)
                            .iter()
                            .map(|&v| val_map[v.index()].expect("operand"))
                            .collect();
                        let op = StructNewOp::new(&mut ctx, fields, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Extract(e) => {
                        let agg = val_map[e.aggregate.index()].expect("operand");
                        let op = ExtractOp::new(&mut ctx, agg, e.index, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Convert(c) => {
                        let val = val_map[c.value.index()].expect("operand");
                        let op = ConvertOp::new(&mut ctx, val, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::DecodeLowFloat(d) => {
                        let val = val_map[d.value.index()].expect("operand");
                        let op = DecodeLowFloatOp::new(&mut ctx, d.format, val, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::EncodeLowFloat(e) => {
                        let val = val_map[e.value.index()].expect("operand");
                        let op = EncodeLowFloatOp::new(&mut ctx, e.format, val, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::DequantizeNvfp4(q) => {
                        let val = val_map[q.value.index()].expect("operand");
                        let b_scale = val_map[q.block_scale.index()].expect("operand");
                        let g_scale = val_map[q.global_scale.index()].expect("operand");
                        let op = DequantizeNvfp4Op::new(
                            &mut ctx,
                            LowFloatFormat::F8E4M3,
                            val,
                            b_scale,
                            g_scale,
                            q.block_application,
                            q.global_application,
                            pliron_ty,
                        );
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::QuantizeNvfp4(q) => {
                        let val = val_map[q.value.index()].expect("operand");
                        let b_scale = val_map[q.block_scale.index()].expect("operand");
                        let g_scale = val_map[q.global_scale.index()].expect("operand");
                        let op = QuantizeNvfp4Op::new(
                            &mut ctx,
                            LowFloatFormat::F8E4M3,
                            val,
                            b_scale,
                            g_scale,
                            q.block_application,
                            q.global_application,
                            pliron_ty,
                        );
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Unary(u) => {
                        let val = val_map[u.value.index()].expect("operand");
                        let op = UnaryOp::new(&mut ctx, u.op, val, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Alloca(a) => {
                        let elem_ty = type_map[a.elem.0 as usize];
                        let op = AllocaOp::new(&mut ctx, elem_ty, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Call(c) => {
                        let callee = legacy.symbol_name(c.symbol);
                        let args: Vec<_> = legacy
                            .value_list(c.args)
                            .iter()
                            .map(|&v| val_map[v.index()].expect("operand"))
                            .collect();
                        let res_tys = legacy
                            .inst_result(*inst)
                            .map(|_| alloc::vec![pliron_ty])
                            .unwrap_or_default();
                        let call_info = CallInfo {
                            is_variadic: c.is_variadic,
                            num_fixed: c.num_fixed,
                            sret: c.sret,
                            ret_regs: c.ret_regs,
                            ret_pieces: c.ret_pieces.to_vec(),
                        };
                        let op = CallOp::new(&mut ctx, callee, args, res_tys, call_info);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::CallIndirect(c) => {
                        let target = val_map[c.target.index()].expect("operand");
                        let args: Vec<_> = legacy
                            .value_list(c.args)
                            .iter()
                            .map(|&v| val_map[v.index()].expect("operand"))
                            .collect();
                        let res_tys = legacy
                            .inst_result(*inst)
                            .map(|_| alloc::vec![pliron_ty])
                            .unwrap_or_default();
                        let call_info = CallInfo {
                            is_variadic: c.is_variadic,
                            num_fixed: c.num_fixed,
                            sret: c.sret,
                            ret_regs: c.ret_regs,
                            ret_pieces: c.ret_pieces.to_vec(),
                        };
                        let op = CallIndirectOp::new(&mut ctx, target, args, res_tys, call_info);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::GlobalAddr(g) => {
                        let symbol = legacy.symbol_name(g.symbol);
                        let op = GlobalAddrOp::new(&mut ctx, symbol, g.via_got, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Load(l) => {
                        let ptr = val_map[l.ptr.index()].expect("operand");
                        let op = LoadOp::new(&mut ctx, ptr, l.volatile, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Store(s) => {
                        let ptr = val_map[s.ptr.index()].expect("operand");
                        let val = val_map[s.value.index()].expect("operand");
                        let op = StoreOp::new(&mut ctx, val, ptr, s.volatile);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                    crate::function::Opcode::Prefetch(p) => {
                        let ptr = val_map[p.ptr.index()].expect("operand");
                        let op = PrefetchOp::new(&mut ctx, ptr);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                    crate::function::Opcode::VaStart(v) => {
                        let list = val_map[v.list.index()].expect("operand");
                        let op = VaStartOp::new(&mut ctx, list);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                    crate::function::Opcode::VaArg(v) => {
                        let list = val_map[v.list.index()].expect("operand");
                        let op = VaArgOp::new(&mut ctx, list, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::VaEnd(v) => {
                        let list = val_map[v.list.index()].expect("operand");
                        let op = VaEndOp::new(&mut ctx, list);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                    crate::function::Opcode::Dot(d) => {
                        let acc = val_map[d.acc.index()].expect("operand");
                        let a = val_map[d.a.index()].expect("operand");
                        let b = val_map[d.b.index()].expect("operand");
                        let op = DotOp::new(&mut ctx, acc, a, b, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Reduce(r) => {
                        let v = val_map[r.vector.index()].expect("operand");
                        let op = ReduceOp::new(&mut ctx, r.op, v, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Splat(s) => {
                        let v = val_map[s.scalar.index()].expect("operand");
                        let op = SplatOp::new(&mut ctx, v, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Matmul(m) => {
                        let a = val_map[m.a.index()].expect("operand");
                        let b = val_map[m.b.index()].expect("operand");
                        let c = val_map[m.c.index()].expect("operand");
                        let desc = convert_matmul_desc_to_pliron(legacy, &m);
                        let op = MatMulOp::new(&mut ctx, a, b, c, desc, pliron_ty);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::Barrier(b_op) => {
                        let op = BarrierOp::new(&mut ctx, b_op.scope);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                    crate::function::Opcode::AtomicRmw(a) => {
                        let ptr = val_map[a.ptr.index()].expect("operand");
                        let val = val_map[a.value.index()].expect("operand");
                        let cmp = a.compare.map(|c| val_map[c.index()].expect("operand"));
                        let op = AtomicRmwOp::new(
                            &mut ctx, a.op, ptr, val, cmp, a.ordering, a.scope, pliron_ty,
                        );
                        op.get_operation().insert_at_back(bb, &mut ctx);
                        if let Some(r) = legacy.inst_result(*inst) {
                            val_map[r.index()] = Some(op.get_operation().deref(&ctx).get_result(0));
                        }
                    }
                    crate::function::Opcode::If(cf) => {
                        let cond = val_map[cf.cond.index()].expect("operand");
                        let then_dest = blocks[cf.then.target.index()];
                        let then_args: Vec<_> = legacy
                            .value_list(cf.then.args)
                            .iter()
                            .map(|&v| val_map[v.index()].expect("operand"))
                            .collect();
                        let else_dest = blocks[cf.else_.target.index()];
                        let else_args: Vec<_> = legacy
                            .value_list(cf.else_.args)
                            .iter()
                            .map(|&v| val_map[v.index()].expect("operand"))
                            .collect();
                        let op = CondBrOp::new(
                            &mut ctx, cond, then_dest, then_args, else_dest, else_args,
                        );
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                }
            }

            // Emit terminator if present.
            if let Some(term) = legacy.terminator(legacy_b) {
                match term {
                    crate::function::Terminator::Ret(r) => {
                        let vals: Vec<_> = r
                            .slice()
                            .iter()
                            .map(|&v| val_map[v.index()].expect("operand"))
                            .collect();
                        let op = RetOp::new(&mut ctx, vals);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                    crate::function::Terminator::Jump(j) => {
                        let dest = blocks[j.target.index()];
                        let args: Vec<_> = legacy
                            .value_list(j.args)
                            .iter()
                            .map(|&v| val_map[v.index()].expect("operand"))
                            .collect();
                        let op = JumpOp::new(&mut ctx, dest, args);
                        op.get_operation().insert_at_back(bb, &mut ctx);
                    }
                }
            }
        }

        let mut res = Self { ctx, func };
        res.set_func_info(FuncInfo {
            is_variadic: legacy.is_variadic,
            num_fixed_params: legacy.num_fixed_params,
            sret: legacy.sret,
            is_local: legacy.is_local,
        });

        res
    }

    /// Convert this pliron [`Function`] into a legacy [`crate::function::Function`].
    pub fn to_legacy(&self) -> Result<crate::function::Function, ConvertError> {
        let mut legacy = crate::function::Function::new();

        if let Some(info) = self.func_info() {
            legacy.is_variadic = info.is_variadic;
            legacy.num_fixed_params = info.num_fixed_params;
            legacy.sret = info.sret;
            legacy.is_local = info.is_local;
        }

        let region = self.func.get_region(&self.ctx);
        let blocks: Vec<_> = region.deref(&self.ctx).iter(&self.ctx).collect();
        if blocks.is_empty() {
            return Ok(legacy);
        }

        // Map pliron basic blocks to legacy Block indices.
        let mut block_map: HashMap<Ptr<BasicBlock>, crate::function::Block> = HashMap::new();
        let mut val_map: HashMap<Value, crate::function::Value> = HashMap::new();

        for (i, &bb) in blocks.iter().enumerate() {
            let legacy_b = legacy.append_block();
            assert_eq!(legacy_b.index(), i);
            block_map.insert(bb, legacy_b);
        }

        // Create block parameters.
        for &bb in &blocks {
            let legacy_b = block_map[&bb];
            let bb_ref = bb.deref(&self.ctx);
            for arg_idx in 0..bb_ref.get_num_arguments() {
                let arg = bb_ref.get_argument(arg_idx);
                let ty = arg.get_type(&self.ctx);
                let legacy_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                let legacy_v = legacy.append_block_param(legacy_b, legacy_ty);
                val_map.insert(arg, legacy_v);
            }
        }

        // Walk instructions in each block.
        for &bb in &blocks {
            let legacy_b = block_map[&bb];
            let bb_ref = bb.deref(&self.ctx);

            for op_ptr in bb_ref.iter(&self.ctx) {
                let op = op_ptr.deref(&self.ctx);
                let op_id = Operation::get_opid(op_ptr, &self.ctx);
                let op_name = op_id.to_string();

                match op_name.as_str() {
                    "volt.iconst" => {
                        let c = ConstIntOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Iconst(c.value(&self.ctx)),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.fconst" => {
                        let c = ConstFloatOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Fconst(c.value(&self.ctx)),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.fconst128" => {
                        let c = ConstF128Op::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Fconst128(c.value(&self.ctx)),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.arith" => {
                        let a = ArithOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let lhs = *val_map
                            .get(&a.lhs(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let rhs = *val_map
                            .get(&a.rhs(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Arith(crate::function::Arith {
                                op: a.kind(&self.ctx),
                                lhs,
                                rhs,
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.arith_imm" => {
                        let a = ArithImmOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let lhs = *val_map
                            .get(&a.lhs(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::ArithImm(crate::function::ArithImm {
                                op: a.kind(&self.ctx),
                                lhs,
                                imm: a.imm(&self.ctx),
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.icmp" => {
                        let c = IcmpOp::from_operation(op_ptr);
                        let bool_ty = legacy.types.intern(crate::types::TypeKind::Bool);
                        let lhs = *val_map
                            .get(&c.lhs(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let rhs = *val_map
                            .get(&c.rhs(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            bool_ty,
                            crate::function::Opcode::Icmp(crate::function::Compare {
                                op: c.kind(&self.ctx),
                                lhs,
                                rhs,
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.select" => {
                        let s = SelectOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let (cond_v, then_raw, else_raw) = s.operands3(&self.ctx);
                        let cond = *val_map.get(&cond_v).ok_or(ConvertError::MissingValue)?;
                        let then_v = *val_map.get(&then_raw).ok_or(ConvertError::MissingValue)?;
                        let else_v = *val_map.get(&else_raw).ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Select(crate::function::Select {
                                cond,
                                then: then_v,
                                else_: else_v,
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.struct_new" => {
                        let sn = StructNewOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let fields: Result<Vec<_>, _> = sn
                            .fields(&self.ctx)
                            .iter()
                            .map(|f| val_map.get(f).copied().ok_or(ConvertError::MissingValue))
                            .collect();
                        let v = legacy.append_struct_new(legacy_b, l_ty, &fields?);
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.extract" => {
                        let e = ExtractOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let agg = *val_map
                            .get(&e.aggregate(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Extract(crate::function::Extract {
                                aggregate: agg,
                                index: e.index(&self.ctx),
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.convert" => {
                        let c = ConvertOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let val = *val_map
                            .get(&c.value(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Convert(crate::function::Convert {
                                value: val,
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.decode_low_float" => {
                        let d = DecodeLowFloatOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let val = *val_map
                            .get(&d.value(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::DecodeLowFloat(
                                crate::function::LowFloatConvert {
                                    format: d.float_format(&self.ctx),
                                    value: val,
                                },
                            ),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.encode_low_float" => {
                        let e = EncodeLowFloatOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let val = *val_map
                            .get(&e.value(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::EncodeLowFloat(
                                crate::function::LowFloatConvert {
                                    format: e.float_format(&self.ctx),
                                    value: val,
                                },
                            ),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.dequantize_nvfp4" => {
                        let q = DequantizeNvfp4Op::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let (_fmt, b_app, g_app, v_raw, bs_raw, gs_raw) = q.parts(&self.ctx);
                        let val = *val_map.get(&v_raw).ok_or(ConvertError::MissingValue)?;
                        let b_scale = *val_map.get(&bs_raw).ok_or(ConvertError::MissingValue)?;
                        let g_scale = *val_map.get(&gs_raw).ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::DequantizeNvfp4(
                                crate::function::NvFp4Convert {
                                    block_application: b_app,
                                    global_application: g_app,
                                    value: val,
                                    block_scale: b_scale,
                                    global_scale: g_scale,
                                },
                            ),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.quantize_nvfp4" => {
                        let q = QuantizeNvfp4Op::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let (_fmt, b_app, g_app, v_raw, bs_raw, gs_raw) = q.parts(&self.ctx);
                        let val = *val_map.get(&v_raw).ok_or(ConvertError::MissingValue)?;
                        let b_scale = *val_map.get(&bs_raw).ok_or(ConvertError::MissingValue)?;
                        let g_scale = *val_map.get(&gs_raw).ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::QuantizeNvfp4(crate::function::NvFp4Convert {
                                block_application: b_app,
                                global_application: g_app,
                                value: val,
                                block_scale: b_scale,
                                global_scale: g_scale,
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.unary" => {
                        let u = UnaryOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let val = *val_map
                            .get(&u.value(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Unary(crate::function::Unary {
                                op: u.kind(&self.ctx),
                                value: val,
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.alloca" => {
                        let a = AllocaOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let elem = convert_pliron_type_to_legacy(
                            &self.ctx,
                            &mut legacy.types,
                            a.elem_type(&self.ctx),
                        )?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Alloca(crate::function::Alloca { elem }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.call" => {
                        let c = CallOp::from_operation(op_ptr);
                        let callee = c.callee(&self.ctx);
                        let info = c.info(&self.ctx);
                        let args: Result<Vec<_>, _> = c
                            .args(&self.ctx)
                            .iter()
                            .map(|a| val_map.get(a).copied().ok_or(ConvertError::MissingValue))
                            .collect();
                        let args_list = legacy.intern_values(&args?);
                        let symbol = legacy.intern_symbol(&callee);
                        let mut ret_pieces = [RetPiece::default(); 4];
                        for (dest, src) in ret_pieces.iter_mut().zip(&info.ret_pieces) {
                            *dest = *src;
                        }
                        let call = crate::function::Call {
                            symbol,
                            args: args_list,
                            is_variadic: info.is_variadic,
                            num_fixed: info.num_fixed,
                            ret_dest: None,
                            ret_regs: info.ret_regs,
                            ret_pieces,
                            sret: info.sret,
                        };
                        if op.get_num_results() > 0 {
                            let ty = op.get_result(0).get_type(&self.ctx);
                            let l_ty =
                                convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                            let v = legacy.append_inst(
                                legacy_b,
                                l_ty,
                                crate::function::Opcode::Call(call),
                            );
                            val_map.insert(op.get_result(0), v);
                        } else {
                            legacy.append_stmt_raw(legacy_b, crate::function::Opcode::Call(call));
                        }
                    }
                    "volt.call_indirect" => {
                        let c = CallIndirectOp::from_operation(op_ptr);
                        let target = *val_map
                            .get(&c.target(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let info = c.info(&self.ctx);
                        let args: Result<Vec<_>, _> = c
                            .args(&self.ctx)
                            .iter()
                            .map(|a| val_map.get(a).copied().ok_or(ConvertError::MissingValue))
                            .collect();
                        let args_list = legacy.intern_values(&args?);
                        let mut ret_pieces = [RetPiece::default(); 4];
                        for (dest, src) in ret_pieces.iter_mut().zip(&info.ret_pieces) {
                            *dest = *src;
                        }
                        let call = crate::function::CallIndirect {
                            target,
                            args: args_list,
                            is_variadic: info.is_variadic,
                            num_fixed: info.num_fixed,
                            ret_dest: None,
                            ret_regs: info.ret_regs,
                            ret_pieces,
                            sret: info.sret,
                        };
                        if op.get_num_results() > 0 {
                            let ty = op.get_result(0).get_type(&self.ctx);
                            let l_ty =
                                convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                            let v = legacy.append_inst(
                                legacy_b,
                                l_ty,
                                crate::function::Opcode::CallIndirect(call),
                            );
                            val_map.insert(op.get_result(0), v);
                        } else {
                            legacy.append_stmt_raw(
                                legacy_b,
                                crate::function::Opcode::CallIndirect(call),
                            );
                        }
                    }
                    "volt.global_addr" => {
                        let g = GlobalAddrOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let symbol = legacy.intern_symbol(&g.symbol(&self.ctx));
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::GlobalAddr(crate::function::GlobalAddr {
                                symbol,
                                via_got: g.via_got(&self.ctx),
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.load" => {
                        let l = LoadOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let ptr = *val_map
                            .get(&l.ptr(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Load(crate::function::Load {
                                ptr,
                                volatile: l.volatile(&self.ctx),
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.store" => {
                        let s = StoreOp::from_operation(op_ptr);
                        let (v_raw, p_raw) = s.parts(&self.ctx);
                        let val = *val_map.get(&v_raw).ok_or(ConvertError::MissingValue)?;
                        let ptr = *val_map.get(&p_raw).ok_or(ConvertError::MissingValue)?;
                        legacy.append_stmt_raw(
                            legacy_b,
                            crate::function::Opcode::Store(crate::function::Store {
                                ptr,
                                value: val,
                                volatile: s.volatile(&self.ctx),
                            }),
                        );
                    }
                    "volt.prefetch" => {
                        let p = PrefetchOp::from_operation(op_ptr);
                        let ptr = *val_map
                            .get(&p.ptr(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        legacy.append_stmt_raw(
                            legacy_b,
                            crate::function::Opcode::Prefetch(crate::function::Prefetch { ptr }),
                        );
                    }
                    "volt.va_start" => {
                        let v = VaStartOp::from_operation(op_ptr);
                        let list = *val_map
                            .get(&v.list(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        legacy.append_stmt_raw(
                            legacy_b,
                            crate::function::Opcode::VaStart(crate::function::VaStart { list }),
                        );
                    }
                    "volt.va_arg" => {
                        let v = VaArgOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let list = *val_map
                            .get(&v.list(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let val = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::VaArg(crate::function::VaArg {
                                list,
                                ty: l_ty,
                            }),
                        );
                        val_map.insert(op.get_result(0), val);
                    }
                    "volt.va_end" => {
                        let v = VaEndOp::from_operation(op_ptr);
                        let list = *val_map
                            .get(&v.list(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        legacy.append_stmt_raw(
                            legacy_b,
                            crate::function::Opcode::VaEnd(crate::function::VaEnd { list }),
                        );
                    }
                    "volt.dot" => {
                        let d = DotOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let (acc_raw, a_raw, b_raw) = d.operands3(&self.ctx);
                        let acc = *val_map.get(&acc_raw).ok_or(ConvertError::MissingValue)?;
                        let a = *val_map.get(&a_raw).ok_or(ConvertError::MissingValue)?;
                        let b = *val_map.get(&b_raw).ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Dot(crate::function::Dot { acc, a, b }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.reduce" => {
                        let r = ReduceOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let vector = *val_map
                            .get(&r.vector(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Reduce(crate::function::Reduce {
                                op: r.kind(&self.ctx),
                                vector,
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.splat" => {
                        let s = SplatOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let scalar = *val_map
                            .get(&s.scalar(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::Splat(crate::function::Splat { scalar }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.matmul" => {
                        let m = MatMulOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let (a_raw, b_raw, c_raw) = m.operands3(&self.ctx);
                        let a = *val_map.get(&a_raw).ok_or(ConvertError::MissingValue)?;
                        let b = *val_map.get(&b_raw).ok_or(ConvertError::MissingValue)?;
                        let c = *val_map.get(&c_raw).ok_or(ConvertError::MissingValue)?;
                        let desc = m.config(&self.ctx);
                        let mm = convert_matmul_desc_to_legacy(&mut legacy, &desc, a, b, c);
                        let v =
                            legacy.append_inst(legacy_b, l_ty, crate::function::Opcode::Matmul(mm));
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.barrier" => {
                        let b_op = BarrierOp::from_operation(op_ptr);
                        legacy.append_stmt_raw(
                            legacy_b,
                            crate::function::Opcode::Barrier(crate::function::Barrier {
                                scope: b_op.scope(&self.ctx),
                            }),
                        );
                    }
                    "volt.atomic_rmw" => {
                        let a = AtomicRmwOp::from_operation(op_ptr);
                        let ty = op.get_result(0).get_type(&self.ctx);
                        let l_ty = convert_pliron_type_to_legacy(&self.ctx, &mut legacy.types, ty)?;
                        let (p_raw, v_raw, cmp_raw) = a.operands_rmw(&self.ctx);
                        let ptr = *val_map.get(&p_raw).ok_or(ConvertError::MissingValue)?;
                        let value = *val_map.get(&v_raw).ok_or(ConvertError::MissingValue)?;
                        let compare = cmp_raw
                            .map(|c| val_map.get(&c).copied().ok_or(ConvertError::MissingValue))
                            .transpose()?;
                        let v = legacy.append_inst(
                            legacy_b,
                            l_ty,
                            crate::function::Opcode::AtomicRmw(crate::function::AtomicRmw {
                                op: a.kind(&self.ctx),
                                ptr,
                                value,
                                compare,
                                ordering: a.ordering(&self.ctx),
                                scope: a.scope(&self.ctx),
                            }),
                        );
                        val_map.insert(op.get_result(0), v);
                    }
                    "volt.cond_br" => {
                        let br = CondBrOp::from_operation(op_ptr);
                        let cond = *val_map
                            .get(&br.cond(&self.ctx))
                            .ok_or(ConvertError::MissingValue)?;
                        let then_target = *block_map
                            .get(&br.then_dest(&self.ctx))
                            .ok_or(ConvertError::MissingBlock)?;
                        let then_args: Result<Vec<_>, _> = br
                            .then_args(&self.ctx)
                            .iter()
                            .map(|a| val_map.get(a).copied().ok_or(ConvertError::MissingValue))
                            .collect();
                        let else_target = *block_map
                            .get(&br.else_dest(&self.ctx))
                            .ok_or(ConvertError::MissingBlock)?;
                        let else_args: Result<Vec<_>, _> = br
                            .else_args(&self.ctx)
                            .iter()
                            .map(|a| val_map.get(a).copied().ok_or(ConvertError::MissingValue))
                            .collect();
                        let then_edge = crate::function::EdgeDesc {
                            target: then_target,
                            args: then_args?,
                        };
                        let else_edge = crate::function::EdgeDesc {
                            target: else_target,
                            args: else_args?,
                        };
                        legacy.append_if(legacy_b, cond, then_edge, else_edge);
                    }
                    "volt.jump" => {
                        let j = JumpOp::from_operation(op_ptr);
                        let target = *block_map
                            .get(&j.target(&self.ctx))
                            .ok_or(ConvertError::MissingBlock)?;
                        let args: Vec<_> = j
                            .args(&self.ctx)
                            .iter()
                            .map(|a| val_map.get(a).copied().ok_or(ConvertError::MissingValue))
                            .collect::<Result<Vec<_>, _>>()?;
                        let interned = legacy.intern_values(&args);
                        legacy.set_terminator(
                            legacy_b,
                            crate::function::Terminator::Jump(crate::function::Jump {
                                target,
                                args: interned,
                            }),
                        );
                    }
                    "volt.ret" => {
                        let r = RetOp::from_operation(op_ptr);
                        let vals: Result<Vec<_>, _> = r
                            .values(&self.ctx)
                            .iter()
                            .map(|a| val_map.get(a).copied().ok_or(ConvertError::MissingValue))
                            .collect();
                        let vals = vals?;
                        legacy.set_terminator(
                            legacy_b,
                            crate::function::Terminator::Ret(crate::function::Ret::many(&vals)),
                        );
                    }
                    _ => return Err(ConvertError::UnsupportedOp(op_name)),
                }
            }
        }

        Ok(legacy)
    }
}

/// An instruction builder for pliron functions.
pub struct Builder<'a> {
    pub ctx: &'a mut Context,
    pub block: Ptr<BasicBlock>,
}

impl<'a> Builder<'a> {
    /// Create a new builder in the given block.
    pub fn new(ctx: &'a mut Context, block: Ptr<BasicBlock>) -> Self {
        Self { ctx, block }
    }

    /// Reposition the builder at the end of another block.
    pub fn switch_to(&mut self, block: Ptr<BasicBlock>) {
        self.block = block;
    }

    /// The current block being appended to.
    pub fn block(&self) -> Ptr<BasicBlock> {
        self.block
    }

    /// Emit an integer constant.
    pub fn iconst(&mut self, val: i64, ty: TypeHandle) -> Value {
        let op = ConstIntOp::new(self.ctx, val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit a double-precision float constant.
    pub fn fconst(&mut self, val: f64, ty: TypeHandle) -> Value {
        let op = ConstFloatOp::new(self.ctx, val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit an f128 constant.
    pub fn fconst128(&mut self, bits: u128, ty: TypeHandle) -> Value {
        let op = ConstF128Op::new(self.ctx, bits, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit binary arithmetic.
    pub fn arith(&mut self, kind: BinOp, lhs: Value, rhs: Value, ty: TypeHandle) -> Value {
        let op = ArithOp::new(self.ctx, kind, lhs, rhs, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit binary arithmetic with an immediate.
    pub fn arith_imm(&mut self, kind: BinOp, lhs: Value, imm: i64, ty: TypeHandle) -> Value {
        let op = ArithImmOp::new(self.ctx, kind, lhs, imm, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit an integer comparison.
    pub fn icmp(&mut self, kind: CmpOp, lhs: Value, rhs: Value) -> Value {
        let bool_ty = BoolType::get_type(self.ctx);
        let op = IcmpOp::new(self.ctx, kind, lhs, rhs, bool_ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit a select.
    pub fn select(
        &mut self,
        cond: Value,
        then_val: Value,
        else_val: Value,
        ty: TypeHandle,
    ) -> Value {
        let op = SelectOp::new(self.ctx, cond, then_val, else_val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit struct construction.
    pub fn struct_new(&mut self, fields: Vec<Value>, ty: TypeHandle) -> Value {
        let op = StructNewOp::new(self.ctx, fields, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit struct field extraction.
    pub fn extract(&mut self, aggregate: Value, index: u32, ty: TypeHandle) -> Value {
        let op = ExtractOp::new(self.ctx, aggregate, index, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit type conversion.
    pub fn convert(&mut self, val: Value, ty: TypeHandle) -> Value {
        let op = ConvertOp::new(self.ctx, val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit low-float decode.
    pub fn decode_low_float(&mut self, fmt: LowFloatFormat, val: Value, ty: TypeHandle) -> Value {
        let op = DecodeLowFloatOp::new(self.ctx, fmt, val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit low-float encode.
    pub fn encode_low_float(&mut self, fmt: LowFloatFormat, val: Value, ty: TypeHandle) -> Value {
        let op = EncodeLowFloatOp::new(self.ctx, fmt, val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit NVFP4 dequantize.
    pub fn dequantize_nvfp4(
        &mut self,
        fmt: LowFloatFormat,
        val: Value,
        b_scale: Value,
        g_scale: Value,
        b_app: ScaleApplication,
        g_app: ScaleApplication,
        ty: TypeHandle,
    ) -> Value {
        let op = DequantizeNvfp4Op::new(self.ctx, fmt, val, b_scale, g_scale, b_app, g_app, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit NVFP4 quantize.
    pub fn quantize_nvfp4(
        &mut self,
        fmt: LowFloatFormat,
        val: Value,
        b_scale: Value,
        g_scale: Value,
        b_app: ScaleApplication,
        g_app: ScaleApplication,
        ty: TypeHandle,
    ) -> Value {
        let op = QuantizeNvfp4Op::new(self.ctx, fmt, val, b_scale, g_scale, b_app, g_app, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit a unary operation.
    pub fn unary(&mut self, kind: crate::function::UnaryOp, val: Value, ty: TypeHandle) -> Value {
        let op = UnaryOp::new(self.ctx, kind, val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit alloca.
    pub fn alloca(&mut self, elem_ty: TypeHandle, ptr_ty: TypeHandle) -> Value {
        let op = AllocaOp::new(self.ctx, elem_ty, ptr_ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit a direct call.
    pub fn call(
        &mut self,
        callee: &str,
        args: Vec<Value>,
        res_tys: Vec<TypeHandle>,
        info: CallInfo,
    ) -> Vec<Value> {
        let op = CallOp::new(self.ctx, callee, args, res_tys, info);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).results().collect()
    }

    /// Emit an indirect call.
    pub fn call_indirect(
        &mut self,
        target: Value,
        args: Vec<Value>,
        res_tys: Vec<TypeHandle>,
        info: CallInfo,
    ) -> Vec<Value> {
        let op = CallIndirectOp::new(self.ctx, target, args, res_tys, info);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).results().collect()
    }

    /// Emit a global address reference.
    pub fn global_addr(&mut self, symbol: &str, via_got: bool, ty: TypeHandle) -> Value {
        let op = GlobalAddrOp::new(self.ctx, symbol, via_got, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit a load.
    pub fn load(&mut self, ptr: Value, volatile: bool, ty: TypeHandle) -> Value {
        let op = LoadOp::new(self.ctx, ptr, volatile, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit a store.
    pub fn store(&mut self, ptr: Value, val: Value, volatile: bool) {
        let op = StoreOp::new(self.ctx, val, ptr, volatile);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }

    /// Emit prefetch.
    pub fn prefetch(&mut self, ptr: Value) {
        let op = PrefetchOp::new(self.ctx, ptr);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }

    /// Emit va_start.
    pub fn va_start(&mut self, list: Value) {
        let op = VaStartOp::new(self.ctx, list);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }

    /// Emit va_arg.
    pub fn va_arg(&mut self, list: Value, ty: TypeHandle) -> Value {
        let op = VaArgOp::new(self.ctx, list, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit va_end.
    pub fn va_end(&mut self, list: Value) {
        let op = VaEndOp::new(self.ctx, list);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }

    /// Emit dot product.
    pub fn dot(&mut self, acc: Value, a: Value, b: Value, ty: TypeHandle) -> Value {
        let op = DotOp::new(self.ctx, acc, a, b, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit vector reduction.
    pub fn reduce(&mut self, kind: BinOp, vec: Value, ty: TypeHandle) -> Value {
        let op = ReduceOp::new(self.ctx, kind, vec, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit vector splat.
    pub fn splat(&mut self, val: Value, ty: TypeHandle) -> Value {
        let op = SplatOp::new(self.ctx, val, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit matrix multiplication.
    pub fn matmul(
        &mut self,
        a: Value,
        b: Value,
        c: Value,
        desc: MatMulDesc,
        ty: TypeHandle,
    ) -> Value {
        let op = MatMulOp::new(self.ctx, a, b, c, desc, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit a barrier.
    pub fn barrier(&mut self, scope: BarrierScope) {
        let op = BarrierOp::new(self.ctx, scope);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }

    /// Emit an atomic RMW operation.
    pub fn atomic_rmw(
        &mut self,
        op: AtomicOp,
        ptr: Value,
        val: Value,
        cmp: Option<Value>,
        ordering: AtomicOrdering,
        scope: AtomicScope,
        ty: TypeHandle,
    ) -> Value {
        let op = AtomicRmwOp::new(self.ctx, op, ptr, val, cmp, ordering, scope, ty);
        op.get_operation().insert_at_back(self.block, self.ctx);
        op.get_operation().deref(self.ctx).get_result(0)
    }

    /// Emit an unconditional jump.
    pub fn jump(&mut self, target: Ptr<BasicBlock>, args: Vec<Value>) {
        let op = JumpOp::new(self.ctx, target, args);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }

    /// Emit a conditional branch.
    pub fn cond_br(
        &mut self,
        cond: Value,
        then_dest: Ptr<BasicBlock>,
        then_args: Vec<Value>,
        else_dest: Ptr<BasicBlock>,
        else_args: Vec<Value>,
    ) {
        let op = CondBrOp::new(self.ctx, cond, then_dest, then_args, else_dest, else_args);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }

    /// Emit a return.
    pub fn ret(&mut self, values: Vec<Value>) {
        let op = RetOp::new(self.ctx, values);
        op.get_operation().insert_at_back(self.block, self.ctx);
    }
}

fn convert_legacy_type_to_pliron(
    ctx: &mut Context,
    types: &crate::types::TypeTable,
    ty: crate::types::Type,
) -> TypeHandle {
    match types.type_kind(ty) {
        crate::types::TypeKind::Bool => BoolType::get_type(ctx),
        crate::types::TypeKind::Int(desc) => IntType::get_type(ctx, desc.signed, desc.bits as u32),
        crate::types::TypeKind::Float(k) => FloatType::get_type(ctx, *k),
        crate::types::TypeKind::Ptr(s) => PtrType::get_type(ctx, *s),
        crate::types::TypeKind::Vector(v) => {
            let elem = convert_legacy_type_to_pliron(ctx, types, v.elem);
            VectorType::get_type(ctx, v.len, elem)
        }
        crate::types::TypeKind::Array(a) => {
            let elem = convert_legacy_type_to_pliron(ctx, types, a.elem);
            ArrayType::get_type(ctx, a.len, elem)
        }
        crate::types::TypeKind::Slice(s) => {
            let elem = convert_legacy_type_to_pliron(ctx, types, s.elem);
            SliceType::get_type(ctx, elem)
        }
        crate::types::TypeKind::Struct(fields) => {
            let field_tys: Vec<_> = fields
                .iter()
                .map(|&f| convert_legacy_type_to_pliron(ctx, types, f))
                .collect();
            StructType::get_type(ctx, field_tys)
        }
    }
}

fn convert_pliron_type_to_legacy(
    ctx: &Context,
    types: &mut crate::types::TypeTable,
    ty: TypeHandle,
) -> Result<crate::types::Type, ConvertError> {
    let kind = describe(ctx, ty).ok_or_else(|| {
        ConvertError::UnsupportedType(alloc::format!(
            "{}",
            ty.deref(ctx).print(ctx, &Default::default())
        ))
    })?;

    let l_kind = match kind {
        VoltKind::Bool => crate::types::TypeKind::Bool,
        VoltKind::Int { signed, bits } => crate::types::TypeKind::Int(crate::types::IntDesc {
            signed,
            bits: bits as u16,
        }),
        VoltKind::Float(k) => crate::types::TypeKind::Float(k),
        VoltKind::Ptr(s) => crate::types::TypeKind::Ptr(s),
        VoltKind::Vector { len, elem } => {
            let e = convert_pliron_type_to_legacy(ctx, types, elem)?;
            crate::types::TypeKind::Vector(crate::types::VectorDesc { len, elem: e })
        }
        VoltKind::Array { len, elem } => {
            let e = convert_pliron_type_to_legacy(ctx, types, elem)?;
            crate::types::TypeKind::Array(crate::types::ArrayDesc { len, elem: e })
        }
        VoltKind::Slice { elem } => {
            let e = convert_pliron_type_to_legacy(ctx, types, elem)?;
            crate::types::TypeKind::Slice(crate::types::SliceDesc { elem: e })
        }
        VoltKind::Struct(fields) => {
            let mut l_fields = Vec::with_capacity(fields.len());
            for f in fields {
                l_fields.push(convert_pliron_type_to_legacy(ctx, types, f)?);
            }
            crate::types::TypeKind::Struct(l_fields)
        }
    };
    Ok(types.intern(l_kind))
}

fn convert_matmul_desc_to_pliron(
    legacy: &crate::function::Function,
    m: &crate::function::MatMul,
) -> MatMulDesc {
    MatMulDesc {
        m: m.m,
        n: m.n,
        k: m.k,
        dtype: m.dtype,
        accumulate: m.accumulate,
        embedded: m.embedded,
        quant: m.quant.as_ref().map(|q| {
            let (scalar, col) = match q.scale {
                crate::function::MatMulScale::Scalar(s) => (Some(s), None),
                crate::function::MatMulScale::PerColumn(cl) => {
                    (None, Some(legacy.scale_list(cl).to_vec()))
                }
            };
            QuantDesc {
                scale_scalar: scalar,
                scale_per_column: col,
                relu: q.relu,
                out: q.out,
                bias: q.bias.map(|b| legacy.bias_list(b).to_vec()),
                zero_point: q.zero_point,
            }
        }),
        input_signs: m.input_signs,
    }
}

fn convert_matmul_desc_to_legacy(
    legacy: &mut crate::function::Function,
    desc: &MatMulDesc,
    a: crate::function::Value,
    b: crate::function::Value,
    c: crate::function::Value,
) -> crate::function::MatMul {
    let quant = desc.quant.as_ref().map(|q| {
        let scale = if let Some(s) = q.scale_scalar {
            crate::function::MatMulScale::Scalar(s)
        } else if let Some(cols) = &q.scale_per_column {
            crate::function::MatMulScale::PerColumn(legacy.intern_scales(cols))
        } else {
            crate::function::MatMulScale::Scalar(0)
        };
        let bias = q.bias.as_ref().map(|biases| legacy.intern_bias(biases));
        crate::function::MatMulQuant {
            scale,
            relu: q.relu,
            out: q.out,
            bias,
            zero_point: q.zero_point,
        }
    });

    crate::function::MatMul {
        a,
        b,
        c,
        m: desc.m,
        n: desc.n,
        k: desc.k,
        dtype: desc.dtype,
        accumulate: desc.accumulate,
        embedded: desc.embedded,
        quant,
        input_signs: desc.input_signs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_emits_valid_function() {
        let mut f = Function::new("test_fn", vec![], vec![]);
        let i32t = IntType::get_type(&f.ctx, true, 32);
        let entry = f.entry_block();
        let mut b = f.builder(entry);
        let v1 = b.iconst(10, i32t);
        let v2 = b.iconst(20, i32t);
        let sum = b.arith(BinOp::Add, v1, v2, i32t);
        b.ret(vec![sum]);

        let text = f.print();
        assert!(text.contains("volt.iconst 10"), "{text}");
        assert!(text.contains("volt.iconst 20"), "{text}");
        assert!(text.contains("volt.arith add"), "{text}");
        assert!(text.contains("volt.ret"), "{text}");
        assert!(f.verify().is_ok());
    }

    #[test]
    fn round_trip_legacy_function() {
        let mut legacy = crate::function::Function::new();
        let i32_t = legacy
            .types
            .intern(crate::types::TypeKind::Int(crate::types::IntDesc {
                signed: true,
                bits: 32,
            }));
        let b0 = legacy.append_block();
        let p0 = legacy.append_block_param(b0, i32_t);
        let c1 = legacy.append_inst(b0, i32_t, crate::function::Opcode::Iconst(42));
        let add = legacy.append_inst(
            b0,
            i32_t,
            crate::function::Opcode::Arith(crate::function::Arith {
                op: BinOp::Add,
                lhs: p0,
                rhs: c1,
            }),
        );
        legacy.set_terminator(
            b0,
            crate::function::Terminator::Ret(crate::function::Ret::one(add)),
        );

        let pliron_f = Function::from_legacy(&legacy);
        assert!(pliron_f.verify().is_ok());

        let round_tripped = pliron_f.to_legacy().unwrap();
        assert_eq!(round_tripped.block_count(), legacy.block_count());
        assert_eq!(round_tripped.block_params(b0).len(), 1);
        assert_eq!(round_tripped.block_insts(b0).len(), 2);
    }

    #[test]
    fn round_trip_control_flow() {
        let mut legacy = crate::function::Function::new();
        let bool_t = legacy.types.intern(crate::types::TypeKind::Bool);
        let i32_t = legacy
            .types
            .intern(crate::types::TypeKind::Int(crate::types::IntDesc {
                signed: true,
                bits: 32,
            }));
        let b0 = legacy.append_block();
        let cond = legacy.append_block_param(b0, bool_t);
        let b1 = legacy.append_block();
        let p1 = legacy.append_block_param(b1, i32_t);
        let b2 = legacy.append_block();
        let p2 = legacy.append_block_param(b2, i32_t);

        let c1 = legacy.append_inst(b0, i32_t, crate::function::Opcode::Iconst(1));
        let c2 = legacy.append_inst(b0, i32_t, crate::function::Opcode::Iconst(2));

        let then_edge = crate::function::EdgeDesc {
            target: b1,
            args: vec![c1],
        };
        let else_edge = crate::function::EdgeDesc {
            target: b2,
            args: vec![c2],
        };
        legacy.append_if(b0, cond, then_edge, else_edge);
        legacy.set_terminator(
            b1,
            crate::function::Terminator::Ret(crate::function::Ret::one(p1)),
        );
        legacy.set_terminator(
            b2,
            crate::function::Terminator::Ret(crate::function::Ret::one(p2)),
        );

        let pliron_f = Function::from_legacy(&legacy);
        assert!(pliron_f.verify().is_ok());

        let round_tripped = pliron_f.to_legacy().unwrap();
        assert_eq!(round_tripped.block_count(), 3);
        assert_eq!(round_tripped.block_params(b1).len(), 1);
        assert_eq!(round_tripped.block_params(b2).len(), 1);
    }

    #[test]
    fn round_trip_memory_ops() {
        let mut legacy = crate::function::Function::new();
        let i32_t = legacy
            .types
            .intern(crate::types::TypeKind::Int(crate::types::IntDesc {
                signed: true,
                bits: 32,
            }));
        let ptr_t = legacy.types.intern(crate::types::TypeKind::Ptr(
            crate::types::AddressSpace::Global,
        ));
        let b0 = legacy.append_block();
        let slot = legacy.append_inst(
            b0,
            ptr_t,
            crate::function::Opcode::Alloca(crate::function::Alloca { elem: i32_t }),
        );
        let c42 = legacy.append_inst(b0, i32_t, crate::function::Opcode::Iconst(42));
        legacy.append_store(b0, c42, slot);
        let loaded = legacy.append_inst(
            b0,
            i32_t,
            crate::function::Opcode::Load(crate::function::Load {
                ptr: slot,
                volatile: false,
            }),
        );
        legacy.set_terminator(
            b0,
            crate::function::Terminator::Ret(crate::function::Ret::one(loaded)),
        );

        let pliron_f = Function::from_legacy(&legacy);
        assert!(pliron_f.verify().is_ok());

        let round_tripped = pliron_f.to_legacy().unwrap();
        assert_eq!(round_tripped.block_count(), 1);
        assert_eq!(round_tripped.block_insts(b0).len(), 4);
    }
}
