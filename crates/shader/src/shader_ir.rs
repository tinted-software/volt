//! Build-time pliron LLVM-dialect IR builders and compiler driver for AMDGPU shaders.
//! Included by `build.rs` and by verification tests.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use pliron::builtin::attributes::{FPSingleAttr, IntegerAttr, StringAttr};
use pliron::builtin::op_interfaces::SingleBlockRegionInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::irbuild::inserter::{IRInserter, Inserter};
use pliron::irbuild::listener::DummyListener;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, TypedHandle};
use pliron::utils::apfloat::f32_to_single;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron_llvm::attributes::{FCmpPredicateAttr, FastmathFlagsAttr, IntegerOverflowFlagsAttr};
use pliron_llvm::op_interfaces::{BinArithOp, FastMathFlags, IntBinArithOpWithOverflowFlag};
use pliron_llvm::ops::{
    CallIntrinsicOp, ConstantOp, ExtractElementOp, FAddOp, FCmpOp, FDivOp, FMulOp, FSubOp, FuncOp,
    GepIndex, GetElementPtrOp, LoadOp, MulOp, ReturnOp, SelectOp,
};
use pliron_llvm::types::{FuncType, PointerType, VectorType, VectorTypeKind, VoidType};
use volt_codegen::targets::{ParameterDescription, ShaderStage as CodegenStage, Signature};
use volt_codegen_gfx10::compile_with_listing;
use volt_raster::gl;
use volt_shader_abi::{FLAG_USES_KILL, ShaderBlobOwned, ShaderStage};

pub const VERTEX_STRIDE: u32 = core::mem::size_of::<volt_raster::Vertex>() as u32;
pub const VS_USER_SGPRS: u32 = 2;
pub const PS_USER_SGPRS: u32 = 7;
pub const VS_PARAM_EXPORTS: u32 = 3;
pub const ENTRY_NAME: &str = "main";

const V_POS: u32 = 0;
const V_COLOR: u32 = 16;
const V_TEX0: u32 = 32;
const V_TEX1: u32 = 40;
const V_FOG: u32 = 48;

pub const ALPHA_FUNCS: [u32; 8] = [
    gl::NEVER,
    gl::LESS,
    gl::EQUAL,
    gl::LEQUAL,
    gl::GREATER,
    gl::NOTEQUAL,
    gl::GEQUAL,
    gl::ALWAYS,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PsKey {
    pub alpha_func: u32,
    pub fog_mode: u32,
}

impl PsKey {
    pub fn uses_kill(&self) -> bool {
        self.alpha_func != gl::ALWAYS
    }
}

pub enum Stage {
    Vertex,
    Pixel,
}

pub struct GpuShaderIr {
    pub ctx: Context,
    pub module: Ptr<Operation>,
    pub stage: Stage,
}

fn f32t(ctx: &Context) -> TypeHandle {
    FP32Type::get(ctx).into()
}
fn i32t(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}
fn i8t(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 8, Signedness::Signless).into()
}
fn void_t(ctx: &Context) -> TypeHandle {
    VoidType::get(ctx).into()
}
fn v2f32(ctx: &Context) -> TypeHandle {
    let f = f32t(ctx);
    VectorType::get(ctx, f, 2, VectorTypeKind::Fixed).into()
}

struct GpuBuilder<'c> {
    ctx: &'c mut Context,
    ins: IRInserter<DummyListener>,
}

macro_rules! emit {
    ($b:expr, $ctor:expr) => {{
        let b = &mut *$b;
        let op = ($ctor)(&mut *b.ctx);
        let opr = op.get_operation();
        b.ins.append_op(&*b.ctx, &op);
        opr.deref(&*b.ctx).get_result(0)
    }};
}

impl<'c> GpuBuilder<'c> {
    fn f(&mut self, x: f32) -> Value {
        emit!(self, |ctx: &mut Context| ConstantOp::new(
            ctx,
            Box::new(FPSingleAttr(f32_to_single(x)))
        ))
    }

    fn int(&mut self, width: usize, v: u32) -> Value {
        emit!(self, |ctx: &mut Context| {
            let ty = IntegerType::get(ctx, width as u32, Signedness::Signless);
            ConstantOp::new(
                ctx,
                Box::new(IntegerAttr::new(
                    ty,
                    APInt::from_u32(v, core::num::NonZeroUsize::new(width).unwrap()),
                )),
            )
        })
    }
    fn i32(&mut self, v: u32) -> Value {
        self.int(32, v)
    }
    fn i1(&mut self, v: bool) -> Value {
        self.int(1, v as u32)
    }

    fn fbin<T: BinArithOp + FastMathFlags>(&mut self, a: Value, b: Value) -> Value {
        emit!(self, |ctx: &mut Context| {
            let op = T::new(ctx, a, b);
            op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
            op
        })
    }
    fn fadd(&mut self, a: Value, b: Value) -> Value {
        self.fbin::<FAddOp>(a, b)
    }
    fn fsub(&mut self, a: Value, b: Value) -> Value {
        self.fbin::<FSubOp>(a, b)
    }
    fn fmul(&mut self, a: Value, b: Value) -> Value {
        self.fbin::<FMulOp>(a, b)
    }
    fn fdiv(&mut self, a: Value, b: Value) -> Value {
        self.fbin::<FDivOp>(a, b)
    }
    fn imul(&mut self, a: Value, b: Value) -> Value {
        emit!(self, |ctx: &mut Context| MulOp::new_with_overflow_flag(
            ctx,
            a,
            b,
            IntegerOverflowFlagsAttr::default()
        ))
    }
    fn fcmp(&mut self, pred: FCmpPredicateAttr, a: Value, b: Value) -> Value {
        emit!(self, |ctx: &mut Context| {
            let op = FCmpOp::new(ctx, pred, a, b);
            op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
            op
        })
    }
    fn sel(&mut self, c: Value, t: Value, f: Value) -> Value {
        emit!(self, |ctx: &mut Context| SelectOp::new(ctx, c, t, f))
    }
    fn clamp01(&mut self, x: Value) -> Value {
        let z = self.f(0.0);
        let o = self.f(1.0);
        let lo = self.fcmp(FCmpPredicateAttr::OLT, x, z);
        let m = self.sel(lo, z, x);
        let hi = self.fcmp(FCmpPredicateAttr::OGT, m, o);
        self.sel(hi, o, m)
    }
    fn extract(&mut self, v: Value, idx: u32) -> Value {
        let i = self.i32(idx);
        emit!(self, |ctx: &mut Context| ExtractElementOp::new(ctx, v, i))
    }
    fn gep_bytes(&mut self, base: Value, off: Value) -> Value {
        emit!(self, |ctx: &mut Context| {
            let t = i8t(ctx);
            GetElementPtrOp::new(ctx, base, vec![GepIndex::Value(off)], t)
        })
    }
    fn gep_const(&mut self, base: Value, off: u32) -> Value {
        emit!(self, |ctx: &mut Context| {
            let t = i8t(ctx);
            GetElementPtrOp::new(ctx, base, vec![GepIndex::Constant(off)], t)
        })
    }
    fn load_f32(&mut self, addr: Value) -> Value {
        emit!(self, |ctx: &mut Context| {
            let t = f32t(ctx);
            LoadOp::new(ctx, addr, t)
        })
    }

    fn intrinsic(&mut self, name: &str, ret: TypeHandle, args: Vec<Value>) -> Value {
        let name = name.to_string();
        emit!(self, |ctx: &mut Context| {
            let arg_tys: Vec<TypeHandle> = args
                .iter()
                .map(|a| pliron::r#type::Typed::get_type(a, ctx))
                .collect();
            let fty: TypedHandle<FuncType> = FuncType::get(ctx, ret, arg_tys, false);
            CallIntrinsicOp::new(ctx, StringAttr::new(name), fty, args)
        })
    }

    fn export(&mut self, target: u32, vals: [Value; 4], done: bool, vm: bool) {
        let tgt = self.i32(target);
        let en = self.i32(0xf);
        let done = self.i1(done);
        let vm = self.i1(vm);
        let ret = void_t(self.ctx);
        self.intrinsic(
            "llvm.amdgcn.exp.f32",
            ret,
            vec![tgt, en, vals[0], vals[1], vals[2], vals[3], done, vm],
        );
    }

    fn ret(&mut self) {
        let op = ReturnOp::new(self.ctx, None);
        self.ins.append_op(&*self.ctx, &op);
    }
}

fn new_program(
    ctx: &mut Context,
    params: Vec<TypeHandle>,
) -> (Ptr<Operation>, Ptr<pliron::basic_block::BasicBlock>) {
    let module = ModuleOp::new(ctx, Identifier::try_from("gpu").unwrap());
    let body = module.get_body(ctx, 0);
    let void = void_t(ctx);
    let fty = FuncType::get(ctx, void, params, false);
    let func = FuncOp::new(ctx, Identifier::try_from(ENTRY_NAME).unwrap(), fty);
    func.get_operation().insert_at_back(body, ctx);
    let entry = func.get_or_create_entry_block(ctx);
    (module.get_operation(), entry)
}

pub fn build_vertex_shader() -> GpuShaderIr {
    let mut ctx = Context::new();
    let gptr: TypeHandle = PointerType::get(&ctx, 1).into();
    let params = vec![gptr, i32t(&ctx)];
    let (module, entry) = new_program(&mut ctx, params);
    let args: Vec<Value> = entry.deref(&ctx).arguments().collect();
    let (vb, vid) = (args[0], args[1]);

    let mut b = GpuBuilder {
        ctx: &mut ctx,
        ins: IRInserter::new_at_block_end(entry),
    };
    let stride = b.i32(VERTEX_STRIDE);
    let off = b.imul(vid, stride);
    let base = b.gep_bytes(vb, off);
    let ld = |b: &mut GpuBuilder, byte: u32| {
        let p = b.gep_const(base, byte);
        b.load_f32(p)
    };
    let pos: Vec<Value> = (0..4).map(|i| ld(&mut b, V_POS + 4 * i)).collect();
    let col: Vec<Value> = (0..4).map(|i| ld(&mut b, V_COLOR + 4 * i)).collect();
    let tex = [
        ld(&mut b, V_TEX0),
        ld(&mut b, V_TEX0 + 4),
        ld(&mut b, V_TEX1),
        ld(&mut b, V_TEX1 + 4),
    ];
    let fog = ld(&mut b, V_FOG);
    let zero = b.f(0.0);

    const PARAM0: u32 = 32;
    const POS0: u32 = 12;
    b.export(POS0, [pos[0], pos[1], pos[2], pos[3]], true, false);
    b.export(PARAM0, [col[0], col[1], col[2], col[3]], false, false);
    b.export(PARAM0 + 1, tex, false, false);
    b.export(PARAM0 + 2, [fog, zero, zero, zero], false, false);
    b.ret();

    GpuShaderIr {
        ctx,
        module,
        stage: Stage::Vertex,
    }
}

pub fn build_pixel_shader(key: &PsKey) -> GpuShaderIr {
    let mut ctx = Context::new();
    let mut params: Vec<TypeHandle> = (0..PS_USER_SGPRS).map(|_| f32t(&ctx)).collect();
    params.push(i32t(&ctx));
    params.push(v2f32(&ctx));
    params.push(v2f32(&ctx));
    let (module, entry) = new_program(&mut ctx, params);
    let args: Vec<Value> = entry.deref(&ctx).arguments().collect();
    let u = &args[..PS_USER_SGPRS as usize];
    let (fog_r, fog_g, fog_b) = (u[0], u[1], u[2]);
    let (fog_start, fog_end, fog_density, alpha_ref) = (u[3], u[4], u[5], u[6]);
    let prim_mask = args[PS_USER_SGPRS as usize];
    let center = args[PS_USER_SGPRS as usize + 2];
    let mut b = GpuBuilder {
        ctx: &mut ctx,
        ins: IRInserter::new_at_block_end(entry),
    };
    let i = b.extract(center, 0);
    let j = b.extract(center, 1);
    let interp = |b: &mut GpuBuilder, attr: u32, chan: u32| {
        let f = f32t(b.ctx);
        let c = b.i32(chan);
        let a = b.i32(attr);
        let p1 = b.intrinsic("llvm.amdgcn.interp.p1", f, vec![i, c, a, prim_mask]);
        let c = b.i32(chan);
        let a = b.i32(attr);
        b.intrinsic("llvm.amdgcn.interp.p2", f, vec![p1, j, c, a, prim_mask])
    };
    let mut color: [Value; 4] = [
        interp(&mut b, 0, 0),
        interp(&mut b, 0, 1),
        interp(&mut b, 0, 2),
        interp(&mut b, 0, 3),
    ];

    if key.fog_mode != 0 {
        let fogc = interp(&mut b, 1, 0);
        let f = match key.fog_mode {
            1 => {
                let n = b.fsub(fog_end, fogc);
                let d = b.fsub(fog_end, fog_start);
                b.fdiv(n, d)
            }
            _ => {
                let d = b.fmul(fog_density, fogc);
                let x = if key.fog_mode == 3 { b.fmul(d, d) } else { d };
                let k = b.f(-core::f32::consts::LOG2_E);
                let x = b.fmul(x, k);
                let ft = f32t(b.ctx);
                b.intrinsic("llvm.exp2.f32", ft, vec![x])
            }
        };
        let f = b.clamp01(f);
        let one = b.f(1.0);
        let inv = b.fsub(one, f);
        for (c, fc) in color.iter_mut().take(3).zip([fog_r, fog_g, fog_b]) {
            let a = b.fmul(*c, f);
            let m = b.fmul(fc, inv);
            *c = b.fadd(a, m);
        }
    }

    if key.uses_kill() {
        let a = color[3];
        let pass = match key.alpha_func {
            gl::NEVER => b.i1(false),
            gl::LESS => b.fcmp(FCmpPredicateAttr::OLT, a, alpha_ref),
            gl::EQUAL => b.fcmp(FCmpPredicateAttr::OEQ, a, alpha_ref),
            gl::LEQUAL => b.fcmp(FCmpPredicateAttr::OLE, a, alpha_ref),
            gl::GREATER => b.fcmp(FCmpPredicateAttr::OGT, a, alpha_ref),
            gl::NOTEQUAL => b.fcmp(FCmpPredicateAttr::UNE, a, alpha_ref),
            gl::GEQUAL => b.fcmp(FCmpPredicateAttr::OGE, a, alpha_ref),
            _ => b.i1(true),
        };
        let void = void_t(b.ctx);
        b.intrinsic("llvm.amdgcn.kill", void, vec![pass]);
    }

    for c in color.iter_mut() {
        *c = b.clamp01(*c);
    }
    const MRT0: u32 = 0;
    b.export(MRT0, color, true, true);
    b.ret();

    GpuShaderIr {
        ctx,
        module,
        stage: Stage::Pixel,
    }
}

pub fn compile_to_blob(
    shader: &GpuShaderIr,
    processor: &str,
    uses_kill: bool,
) -> Result<ShaderBlobOwned, String> {
    let (stage, abi_stage, number_of_scalar_parameters, parameters, user_sgprs, param_exports) =
        match shader.stage {
            Stage::Vertex => (
                CodegenStage::Vertex,
                ShaderStage::Vertex,
                1,
                vec![
                    ParameterDescription {
                        size: 2,
                        vector: false,
                    },
                    ParameterDescription {
                        size: 1,
                        vector: false,
                    },
                ],
                VS_USER_SGPRS as u16,
                VS_PARAM_EXPORTS as u16,
            ),
            Stage::Pixel => (
                CodegenStage::Pixel,
                ShaderStage::Pixel,
                PS_USER_SGPRS + 1,
                (0..PS_USER_SGPRS + 1)
                    .map(|_| ParameterDescription {
                        size: 1,
                        vector: false,
                    })
                    .chain((0..2).map(|_| ParameterDescription {
                        size: 2,
                        vector: true,
                    }))
                    .collect(),
                PS_USER_SGPRS as u16,
                0,
            ),
        };
    let signature = Signature {
        stage,
        number_of_scalar_parameters,
        parameters,
    };
    let compiled = compile_with_listing(&shader.ctx, shader.module, &signature, processor, false)
        .map_err(|error| format!("{error:?}"))?;

    Ok(ShaderBlobOwned {
        stage: abi_stage,
        flags: if uses_kill { FLAG_USES_KILL } else { 0 },
        user_sgprs,
        param_exports,
        config: compiled.configuration,
        code: compiled.code,
    })
}
