extern crate alloc;

use alloc::vec::Vec;

use super::function::{BinOp, Block, CmpOp, Convert, ConvertKind, Function, Inst, Opcode, Value};
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

/// The width of the integer `v` has, if it is one.
fn int_bits(func: &Function, v: Value) -> Option<u16> {
    match func.types.type_kind(func.value_type(v)) {
        TypeKind::Int(i) => Some(i.bits),
        _ => None,
    }
}

/// The width of the integer a soft-float routine takes or returns for a `bits`-wide one: `int`
/// (32) up to 32 bits and `long long` (64) up to 64 bits; nothing wider has a routine.
fn routine_bits(bits: u16) -> Option<u16> {
    match bits {
        1..=32 => Some(32),
        33..=64 => Some(64),
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

/// The comparison routine of a float compare and the SIGNED integer compare of its `int`
/// result against zero that gives the compare's value (the routines return `< 0`, `0`, `> 0`).
fn cmp_sym(op: CmpOp) -> Option<(&'static str, CmpOp)> {
    match op {
        CmpOp::Eq => Some(("__eqtf2", CmpOp::Eq)),
        CmpOp::Ne => Some(("__netf2", CmpOp::Ne)),
        CmpOp::Lt => Some(("__lttf2", CmpOp::Slt)),
        CmpOp::Le => Some(("__letf2", CmpOp::Sle)),
        CmpOp::Gt => Some(("__gttf2", CmpOp::Sgt)),
        CmpOp::Ge => Some(("__getf2", CmpOp::Sge)),
        _ => None,
    }
}

pub fn lower(func: &mut Function) -> bool {
    let i32_ty = func.types.intern(TypeKind::Int(IntDesc { bits: 32 }));
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
            let Some((routine, against_zero)) = cmp_sym(c.op) else {
                return false;
            };
            let status = emit_call(func, order, i32_ty, routine, &[c.lhs, c.rhs]);
            let zero = emit_inst(func, order, i32_ty, Opcode::Iconst(0));
            let res_ty = func.value_type(result);
            let cmp = emit_inst(
                func,
                order,
                res_ty,
                Opcode::Icmp(super::function::Compare {
                    op: against_zero,
                    lhs: status,
                    rhs: zero,
                }),
            );
            func.replace_all_uses(result, cmp);
            true
        }
        Opcode::Convert(cv) => {
            let (src_q, dst_q) = (is_f128(func, cv.value), is_f128(func, result));
            // Neither side is f128, or both are (not a conversion): nothing to call.
            if src_q == dst_q {
                return false;
            }
            let res_ty = func.value_type(result);
            let call = match (cv.kind, dst_q) {
                (ConvertKind::FpResize, true) => {
                    let routine = match float_kind(func, cv.value) {
                        Some(FloatKind::F32) => "__extendsftf2",
                        Some(FloatKind::F64) => "__extenddftf2",
                        Some(FloatKind::F16) => "__extendhftf2",
                        _ => return false,
                    };
                    emit_call(func, order, res_ty, routine, &[cv.value])
                }
                (ConvertKind::SiToFp | ConvertKind::UiToFp, true) => {
                    // The routines take an `int` / `long long`: a narrower source is widened
                    // first, sign- or zero-extending as the conversion reads it.
                    let signed = cv.kind == ConvertKind::SiToFp;
                    let Some(bits) = int_bits(func, cv.value) else {
                        return false;
                    };
                    let Some(arg_bits) = routine_bits(bits) else {
                        return false;
                    };
                    let arg_ty = func.types.intern(TypeKind::Int(IntDesc { bits: arg_bits }));
                    let arg = if bits < arg_bits {
                        let kind = if signed {
                            ConvertKind::Sext
                        } else {
                            ConvertKind::Zext
                        };
                        let widen = Opcode::Convert(Convert {
                            value: cv.value,
                            kind,
                        });
                        emit_inst(func, order, arg_ty, widen)
                    } else {
                        cv.value
                    };
                    let routine = match (signed, arg_bits) {
                        (true, 32) => "__floatsitf",
                        (true, _) => "__floatditf",
                        (false, 32) => "__floatunsitf",
                        (false, _) => "__floatunditf",
                    };
                    emit_call(func, order, res_ty, routine, &[arg])
                }
                (ConvertKind::FpResize, false) => {
                    let routine = match float_kind(func, result) {
                        Some(FloatKind::F32) => "__trunctfsf2",
                        Some(FloatKind::F64) => "__trunctfdf2",
                        Some(FloatKind::F16) => "__trunctfhf2",
                        _ => return false,
                    };
                    emit_call(func, order, res_ty, routine, &[cv.value])
                }
                (ConvertKind::FpToSi | ConvertKind::FpToUi, false) => {
                    // The routine returns an `int` / `long long`; a narrower destination is
                    // that result truncated.
                    let signed = cv.kind == ConvertKind::FpToSi;
                    let Some(bits) = int_bits(func, result) else {
                        return false;
                    };
                    let Some(ret_bits) = routine_bits(bits) else {
                        return false;
                    };
                    let ret_ty = func.types.intern(TypeKind::Int(IntDesc { bits: ret_bits }));
                    let routine = match (signed, ret_bits) {
                        (true, 32) => "__fixtfsi",
                        (true, _) => "__fixtfdi",
                        (false, 32) => "__fixunstfsi",
                        (false, _) => "__fixunstfdi",
                    };
                    let raw = emit_call(func, order, ret_ty, routine, &[cv.value]);
                    if bits < ret_bits {
                        let narrow = Opcode::Convert(Convert {
                            value: raw,
                            kind: ConvertKind::Trunc,
                        });
                        emit_inst(func, order, res_ty, narrow)
                    } else {
                        raw
                    }
                }
                _ => return false,
            };
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

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;

    use super::*;
    use crate::function::{Compare, Ret, Terminator};
    use crate::verify::{Profile, verify};

    /// One word per instruction of the entry block: the routine of a call, the kind of a
    /// convert, the operator of a compare.
    fn shape(func: &Function) -> Vec<String> {
        func.block_insts(Block::from_u32(0))
            .iter()
            .map(|&inst| match func.opcode(inst) {
                Opcode::Call(c) => format!("call {}", func.symbol_name(c.symbol)),
                Opcode::Convert(c) => format!("convert {}", c.kind.name()),
                Opcode::Icmp(c) => format!("icmp {}", c.op.symbol()),
                Opcode::Iconst(_) => String::from("iconst"),
                Opcode::Arith(a) => format!("arith {}", a.op.symbol()),
                other => panic!("unexpected {other:?}"),
            })
            .collect()
    }

    /// `fn(x: src) -> dst { ret convert <kind> dst, x }` after `lower`.
    fn lowered_convert(kind: ConvertKind, src: &str, dst: &str) -> (Function, bool) {
        let mut func = Function::new();
        let (s, d) = (
            func.types.parse_type(src).unwrap(),
            func.types.parse_type(dst).unwrap(),
        );
        let entry = func.append_block();
        let x = func.append_block_param(entry, s);
        let r = func.append_convert(entry, d, kind, x);
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        let changed = lower(&mut func);
        (func, changed)
    }

    fn check(kind: ConvertKind, src: &str, dst: &str, want: &[&str]) {
        let (func, changed) = lowered_convert(kind, src, dst);
        assert!(changed, "{kind:?} {src} -> {dst}");
        assert_eq!(shape(&func), want, "{kind:?} {src} -> {dst}");
        let diags = verify(&func, Profile::High);
        assert!(diags.ok(), "{kind:?} {src} -> {dst}: {:?}", diags.items());
        // The result the function returns is the last instruction's.
        let Some(Terminator::Ret(ret)) = func.terminator(Block::from_u32(0)) else {
            panic!("ret");
        };
        let last = *func.block_insts(Block::from_u32(0)).last().unwrap();
        assert_eq!(func.inst_result(last), Some(ret.values[0]));
        assert_eq!(func.value_type(ret.values[0]), {
            let mut types = func.types.clone();
            types.parse_type(dst).unwrap()
        });
    }

    #[test]
    fn integers_narrower_than_the_routine_are_widened_as_the_conversion_reads_them() {
        use ConvertKind::*;
        check(SiToFp, "i8", "f128", &["convert sext", "call __floatsitf"]);
        check(SiToFp, "i16", "f128", &["convert sext", "call __floatsitf"]);
        check(SiToFp, "i32", "f128", &["call __floatsitf"]);
        check(SiToFp, "i40", "f128", &["convert sext", "call __floatditf"]);
        check(SiToFp, "i64", "f128", &["call __floatditf"]);
        check(
            UiToFp,
            "i8",
            "f128",
            &["convert zext", "call __floatunsitf"],
        );
        check(UiToFp, "i32", "f128", &["call __floatunsitf"]);
        check(
            UiToFp,
            "i40",
            "f128",
            &["convert zext", "call __floatunditf"],
        );
        check(UiToFp, "i64", "f128", &["call __floatunditf"]);
    }

    #[test]
    fn f128_to_integer_calls_the_routine_of_the_destination_sign_and_truncates() {
        use ConvertKind::*;
        check(FpToSi, "f128", "i32", &["call __fixtfsi"]);
        check(FpToSi, "f128", "i8", &["call __fixtfsi", "convert trunc"]);
        check(FpToSi, "f128", "i64", &["call __fixtfdi"]);
        check(FpToSi, "f128", "i48", &["call __fixtfdi", "convert trunc"]);
        check(FpToUi, "f128", "i32", &["call __fixunstfsi"]);
        check(
            FpToUi,
            "f128",
            "i16",
            &["call __fixunstfsi", "convert trunc"],
        );
        check(FpToUi, "f128", "i64", &["call __fixunstfdi"]);
    }

    #[test]
    fn f128_float_resizes_call_the_extend_and_truncate_routines() {
        use ConvertKind::*;
        check(FpResize, "f32", "f128", &["call __extendsftf2"]);
        check(FpResize, "f64", "f128", &["call __extenddftf2"]);
        check(FpResize, "f16", "f128", &["call __extendhftf2"]);
        check(FpResize, "f128", "f32", &["call __trunctfsf2"]);
        check(FpResize, "f128", "f64", &["call __trunctfdf2"]);
        check(FpResize, "f128", "f16", &["call __trunctfhf2"]);
    }

    #[test]
    fn conversions_with_no_f128_or_no_routine_are_left_alone() {
        use ConvertKind::*;
        for (kind, src, dst) in [
            (SiToFp, "i32", "f32"),
            (FpToUi, "f64", "i64"),
            (FpResize, "f32", "f64"),
            (Zext, "i8", "i32"),
            // Wider than any routine's integer.
            (SiToFp, "i128", "f128"),
            (FpToSi, "f128", "i128"),
            // Kind and types disagree (the verifier rejects these).
            (Zext, "i8", "f128"),
            (FpResize, "i32", "f128"),
        ] {
            let (func, changed) = lowered_convert(kind, src, dst);
            assert!(!changed, "{kind:?} {src} -> {dst}");
            assert_eq!(shape(&func).len(), 1);
        }
    }

    #[test]
    fn f128_comparisons_test_the_routines_int_result_against_zero_signed() {
        for (op, routine, against_zero) in [
            (CmpOp::Eq, "__eqtf2", CmpOp::Eq),
            (CmpOp::Ne, "__netf2", CmpOp::Ne),
            (CmpOp::Lt, "__lttf2", CmpOp::Slt),
            (CmpOp::Le, "__letf2", CmpOp::Sle),
            (CmpOp::Gt, "__gttf2", CmpOp::Sgt),
            (CmpOp::Ge, "__getf2", CmpOp::Sge),
        ] {
            let mut func = Function::new();
            let f128 = func.types.parse_type("f128").unwrap();
            let bool_t = func.types.parse_type("bool").unwrap();
            let entry = func.append_block();
            let a = func.append_block_param(entry, f128);
            let b = func.append_block_param(entry, f128);
            let r = func.append_inst(entry, bool_t, Opcode::Icmp(Compare { op, lhs: a, rhs: b }));
            func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
            assert!(lower(&mut func));
            assert_eq!(
                shape(&func),
                vec![
                    format!("call {routine}"),
                    String::from("iconst"),
                    format!("icmp {}", against_zero.symbol()),
                ]
            );
            let diags = verify(&func, Profile::High);
            assert!(diags.ok(), "{op:?}: {:?}", diags.items());
        }
    }

    #[test]
    fn f128_arithmetic_and_non_f128_instructions() {
        let mut func = Function::new();
        let f128 = func.types.parse_type("f128").unwrap();
        let i32_t = func.types.parse_type("i32").unwrap();
        let entry = func.append_block();
        let a = func.append_block_param(entry, f128);
        let b = func.append_block_param(entry, f128);
        let x = func.append_block_param(entry, i32_t);
        let y = func.append_block_param(entry, i32_t);
        // Integer division is not an f128 operation and must not be touched.
        let q = func.append_inst(
            entry,
            i32_t,
            Opcode::Arith(crate::function::Arith {
                op: BinOp::SDiv,
                lhs: x,
                rhs: y,
            }),
        );
        let s = func.append_inst(
            entry,
            f128,
            Opcode::Arith(crate::function::Arith {
                op: BinOp::Div,
                lhs: a,
                rhs: b,
            }),
        );
        func.set_terminator(entry, Terminator::Ret(Ret::many(&[q, s])));
        assert!(lower(&mut func));
        assert_eq!(shape(&func), ["arith /s", "call __divtf3"]);
        let diags = verify(&func, Profile::High);
        assert!(diags.ok(), "{:?}", diags.items());
    }
}
