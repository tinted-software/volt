extern crate alloc;

use alloc::vec::Vec;

use super::function::{BinOp, Block, CmpOp, Function, Inst, Opcode, Value};
use super::types::Type;
use super::types::{FloatKind, IntDesc, TypeKind};

pub const SYMBOLS: &[&str] = &[
    "__addtf3",
    "__subtf3",
    "__multf3",
    "__divtf3",
    "__eqtf2",
    "__netf2",
    "__lttf2",
    "__letf2",
    "__gttf2",
    "__getf2",
    "__extendsftf2",
    "__extenddftf2",
    "__extendhftf2",
    "__trunctfsf2",
    "__trunctfdf2",
    "__trunctfhf2",
    "__floatsitf",
    "__floatditf",
    "__floatunsitf",
    "__floatunditf",
    "__fixtfsi",
    "__fixtfdi",
    "__fixunstfsi",
    "__fixunstfdi",
    "__sqrttf2",
];

pub fn static_name(name: &str) -> Option<&'static str> {
    SYMBOLS.iter().find(|s| **s == name).copied()
}

fn is_f128(func: &Function, v: Value) -> bool {
    matches!(
        func.types.type_kind(func.value_type(v)),
        TypeKind::Float(FloatKind::F128)
    )
}

fn float_kind(func: &Function, v: Value) -> Option<FloatKind> {
    match func.types.type_kind(func.value_type(v)) {
        TypeKind::Float(f) => Some(*f),
        _ => None,
    }
}

struct IntInfo {
    bits: u16,
    signed: bool,
}

fn int_info(func: &Function, v: Value) -> Option<IntInfo> {
    match func.types.type_kind(func.value_type(v)) {
        TypeKind::Int(i) => Some(IntInfo {
            bits: i.bits,
            signed: i.signed,
        }),
        _ => None,
    }
}

fn arith_sym(op: BinOp) -> Option<&'static str> {
    match op {
        BinOp::Add => Some("__addtf3"),
        BinOp::Sub => Some("__subtf3"),
        BinOp::Mul => Some("__multf3"),
        BinOp::Div => Some("__divtf3"),
        _ => None,
    }
}

fn cmp_sym(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "__eqtf2",
        CmpOp::Ne => "__netf2",
        CmpOp::Lt => "__lttf2",
        CmpOp::Le => "__letf2",
        CmpOp::Gt => "__gttf2",
        CmpOp::Ge => "__getf2",
    }
}

pub fn lower(func: &mut Function) -> bool {
    let i32_ty = func.types.intern(TypeKind::Int(IntDesc {
        signed: true,
        bits: 32,
    }));
    let f128_ty = func.types.intern(TypeKind::Float(FloatKind::F128));
    let mut changed = false;
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let old = func.block_insts(block).to_vec();
        let mut order: Vec<Inst> = Vec::new();
        for inst in old {
            if lower_inst(func, inst, &mut order, i32_ty, f128_ty) {
                changed = true;
            } else {
                order.push(inst);
            }
        }
        if changed {
            func.set_block_insts(block, &order);
        }
    }
    changed
}

fn emit_call(
    func: &mut Function,
    order: &mut Vec<Inst>,
    ty: Type,
    name: &str,
    args: &[Value],
) -> Value {
    let symbol = func.intern_symbol(name);
    let list = func.intern_values(args);
    let v = func.create_inst(
        ty,
        Opcode::Call(super::function::Call {
            symbol,
            args: list,
            is_variadic: false,
            num_fixed: 0,
            ret_dest: None,
            ret_regs: 0,
            ret_pieces: [super::function::RetPiece::default(); 4],
            sret: false,
        }),
    );
    order.push(func.defining_inst(v).unwrap());
    v
}

fn emit_inst(func: &mut Function, order: &mut Vec<Inst>, ty: Type, op: Opcode) -> Value {
    let v = func.create_inst(ty, op);
    order.push(func.defining_inst(v).unwrap());
    v
}

#[allow(clippy::too_many_arguments)]
fn lower_inst(
    func: &mut Function,
    inst: Inst,
    order: &mut Vec<Inst>,
    i32_ty: Type,
    f128_ty: Type,
) -> bool {
    let Some(result) = func.inst_result(inst) else {
        return false;
    };
    match func.opcode(inst) {
        Opcode::Arith(a) => {
            if !is_f128(func, result) {
                return false;
            }
            let Some(sym) = arith_sym(a.op) else {
                return false;
            };
            let call = emit_call(func, order, f128_ty, sym, &[a.lhs, a.rhs]);
            func.replace_all_uses(result, call);
            true
        }
        Opcode::Icmp(c) => {
            if !is_f128(func, c.lhs) {
                return false;
            }
            let status = emit_call(func, order, i32_ty, cmp_sym(c.op), &[c.lhs, c.rhs]);
            let zero = emit_inst(func, order, i32_ty, Opcode::Iconst(0));
            let res_ty = func.value_type(result);
            let cmp = emit_inst(
                func,
                order,
                res_ty,
                Opcode::Icmp(super::function::Compare {
                    op: c.op,
                    lhs: status,
                    rhs: zero,
                }),
            );
            func.replace_all_uses(result, cmp);
            true
        }
        Opcode::Convert(cv) => {
            let src_q = is_f128(func, cv.value);
            let dst_q = is_f128(func, result);
            if !src_q && !dst_q {
                return false;
            }
            if src_q && dst_q {
                return false;
            }
            let res_ty = func.value_type(result);
            let sym = if dst_q {
                to_f128_sym(func, cv.value)
            } else {
                from_f128_sym(func, cv.value, res_ty, func)
            };
            let call = emit_call(func, order, res_ty, sym, &[cv.value]);
            func.replace_all_uses(result, call);
            true
        }
        Opcode::Unary(u) => {
            if u.op != super::function::UnaryOp::Sqrt || !is_f128(func, result) {
                return false;
            }
            let call = emit_call(func, order, f128_ty, "__sqrttf2", &[u.value]);
            func.replace_all_uses(result, call);
            true
        }
        _ => false,
    }
}

fn to_f128_sym(func: &Function, src: Value) -> &'static str {
    if let Some(fk) = float_kind(func, src) {
        return match fk {
            FloatKind::F32 => "__extendsftf2",
            FloatKind::F64 => "__extenddftf2",
            FloatKind::F16 => "__extendhftf2",
            FloatKind::F128 => unreachable!(),
        };
    }
    let ii = int_info(func, src).expect("softfp convert source is a number");
    if ii.signed {
        if ii.bits <= 32 {
            "__floatsitf"
        } else {
            "__floatditf"
        }
    } else if ii.bits <= 32 {
        "__floatunsitf"
    } else {
        "__floatunditf"
    }
}

fn from_f128_sym(func: &Function, src: Value, dst_ty: Type, _f: &Function) -> &'static str {
    let _ = src;
    match func.types.type_kind(dst_ty) {
        TypeKind::Float(FloatKind::F32) => "__trunctfsf2",
        TypeKind::Float(FloatKind::F64) => "__trunctfdf2",
        TypeKind::Float(FloatKind::F16) => "__trunctfhf2",
        TypeKind::Int(i) => {
            if i.signed {
                if i.bits <= 32 {
                    "__fixtfsi"
                } else {
                    "__fixtfdi"
                }
            } else if i.bits <= 32 {
                "__fixunstfsi"
            } else {
                "__fixunstfdi"
            }
        }
        _ => unreachable!(),
    }
}
