// SPDX-License-Identifier: Apache-2.0
//! Byte-level checks that signedness is chosen by the operation: the selector emits the signed
//! or the unsigned instruction (and the sign extension a signed operation needs) according to
//! the operator, never according to the type. They only compile code, so unlike `tests.rs`
//! (which executes it) they run on every host; the instruction text comes from this crate's
//! own disassembler, whose encodings are separately checked against llvm-mc.
use super::{Error, cases, compile, disasm};
use alloc::{format, string::String, vec::Vec};
use volt_ir::function::*;

/// The disassembled instructions of `f`, one string per instruction.
fn asm(f: &Function) -> Vec<String> {
    let code = compile(f).expect("compiles");
    disasm::format(&code)
        .lines()
        .map(|l| l.split_once(": ").expect("offset prefix").1.into())
        .collect()
}

fn has(lines: &[String], text: &str) -> bool {
    lines.iter().any(|l| l == text)
}

#[track_caller]
fn expect(lines: &[String], present: &[&str], absent: &[&str]) {
    for p in present {
        assert!(has(lines, p), "missing `{p}` in:\n{}", lines.join("\n"));
    }
    for a in absent {
        assert!(!has(lines, a), "unexpected `{a}` in:\n{}", lines.join("\n"));
    }
}

#[test]
fn every_case_compiles_to_decodable_code() {
    for case in cases::all() {
        let code = compile(&case.function).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let text = disasm::format(&code);
        assert!(!text.contains(".byte"), "{}:\n{text}", case.name);
        check_case_shape(&case);
    }
}

/// The arguments and expected results of a case match the function's signature, and an
/// expected integer result is already zero-extended from its declared width.
fn check_case_shape(case: &cases::Case) {
    use cases::Val;
    use volt_ir::types::{FloatKind, TypeKind};
    let f = &case.function;
    let class = |v: Value| match f.types.type_kind(f.value_type(v)) {
        TypeKind::Float(FloatKind::F32) => "f32",
        TypeKind::Float(FloatKind::F64) => "f64",
        _ => "int",
    };
    let params = f.block_params(Block(0));
    let Some(Terminator::Ret(ret)) = f.terminator(Block(0)) else {
        panic!("{}: no return", case.name);
    };
    let result = ret.slice()[0];
    let bits = match f.types.type_kind(f.value_type(result)) {
        TypeKind::Int(i) => i.bits as u32,
        TypeKind::Bool => 1,
        _ => 64,
    };
    assert!(!case.runs.is_empty(), "{}", case.name);
    for (args, want) in &case.runs {
        assert_eq!(args.len(), params.len(), "{}", case.name);
        for (a, &p) in args.iter().zip(params) {
            let want_class = match a {
                Val::Int(_) => "int",
                Val::F32(_) => "f32",
                Val::F64(_) => "f64",
            };
            assert_eq!(want_class, class(p), "{}", case.name);
        }
        let result_class = match want {
            Val::Int(_) => "int",
            Val::F32(_) => "f32",
            Val::F64(_) => "f64",
        };
        assert_eq!(result_class, class(result), "{}", case.name);
        if let Val::Int(_) = want {
            assert_eq!(
                want.bits() & !cases::mask(bits),
                0,
                "{} {args:x?}",
                case.name
            );
        }
    }
}

#[test]
fn division_follows_the_operator() {
    for (kind, wide) in [("i32", false), ("i64", true)] {
        let r11 = if wide { "r11" } else { "r11d" };
        let unsigned = asm(&cases::arith(kind, BinOp::UDiv));
        expect(
            &unsigned,
            &["xor edx, edx", &format!("div {r11}")],
            &["cdq", "cqo", &format!("idiv {r11}")],
        );
        let signed = asm(&cases::arith(kind, BinOp::SDiv));
        expect(
            &signed,
            &[if wide { "cqo" } else { "cdq" }, &format!("idiv {r11}")],
            &["xor edx, edx", &format!("div {r11}")],
        );
        // The remainder is the same divide read from rdx.
        let urem = asm(&cases::arith(kind, BinOp::URem));
        let srem = asm(&cases::arith(kind, BinOp::SRem));
        expect(&urem, &[&format!("div {r11}"), "mov r10, rdx"], &[]);
        expect(&srem, &[&format!("idiv {r11}"), "mov r10, rdx"], &[]);
    }
}

#[test]
fn narrow_signed_operations_sign_extend_their_operands_and_unsigned_ones_do_not() {
    let sign_extension = ["shl r10, 56", "sar r10, 56", "shl r11, 56", "sar r11, 56"];
    for op in [BinOp::SDiv, BinOp::SRem, BinOp::SMulh] {
        let lines = asm(&cases::arith("i8", op));
        expect(&lines, &sign_extension, &[]);
    }
    for op in [BinOp::UDiv, BinOp::URem, BinOp::UMulh] {
        let lines = asm(&cases::arith("i8", op));
        expect(&lines, &[], &["sar r10, 56", "sar r11, 56"]);
    }
    // An arithmetic shift extends only the value shifted; a logical one extends nothing.
    let sar = asm(&cases::arith("i8", BinOp::Sar));
    expect(
        &sar,
        &["shl r10, 56", "sar r10, 56", "sar r10d, cl"],
        &["sar r11, 56"],
    );
    let shr = asm(&cases::arith("i8", BinOp::Shr));
    expect(&shr, &["shr r10d, cl"], &["sar r10, 56", "sar r10d, cl"]);
    // Wrapping operations need neither.
    for op in [BinOp::Add, BinOp::Mul, BinOp::BitAnd] {
        let lines = asm(&cases::arith("i8", op));
        expect(&lines, &[], &["sar r10, 56"]);
    }
}

#[test]
fn shifts_follow_the_operator() {
    for (kind, r10) in [("i32", "r10d"), ("i64", "r10")] {
        expect(
            &asm(&cases::arith(kind, BinOp::Shl)),
            &[&format!("shl {r10}, cl")],
            &[&format!("shr {r10}, cl"), &format!("sar {r10}, cl")],
        );
        expect(
            &asm(&cases::arith(kind, BinOp::Shr)),
            &[&format!("shr {r10}, cl")],
            &[&format!("shl {r10}, cl"), &format!("sar {r10}, cl")],
        );
        expect(
            &asm(&cases::arith(kind, BinOp::Sar)),
            &[&format!("sar {r10}, cl")],
            &[&format!("shl {r10}, cl"), &format!("shr {r10}, cl")],
        );
    }
}

#[test]
fn high_product_follows_the_operator() {
    expect(
        &asm(&cases::arith("i32", BinOp::UMulh)),
        &["mul r11d"],
        &["imul r11d"],
    );
    expect(
        &asm(&cases::arith("i32", BinOp::SMulh)),
        &["imul r11d"],
        &["mul r11d"],
    );
    expect(
        &asm(&cases::arith("i64", BinOp::UMulh)),
        &["mul r11"],
        &["imul r11"],
    );
    expect(
        &asm(&cases::arith("i64", BinOp::SMulh)),
        &["imul r11"],
        &["mul r11"],
    );
    // Narrower than 32 bits: a 64-bit product shifted down logically or arithmetically.
    expect(
        &asm(&cases::arith("i16", BinOp::UMulh)),
        &["imul r10, r11", "shr r10, 16"],
        &["sar r10, 16"],
    );
    expect(
        &asm(&cases::arith("i16", BinOp::SMulh)),
        &["imul r10, r11", "sar r10, 16"],
        &["shr r10, 16"],
    );
    // Between 32 and 64 bits the double-width product spans rdx:rax.
    expect(
        &asm(&cases::arith("i48", BinOp::UMulh)),
        &["mul r11"],
        &["imul r11"],
    );
    expect(
        &asm(&cases::arith("i48", BinOp::SMulh)),
        &["imul r11"],
        &["mul r11"],
    );
}

#[test]
fn comparisons_pick_the_condition_code_by_operator() {
    // (operator, condition suffix of the set instruction)
    for (op, cc) in [
        (CmpOp::Eq, "e"),
        (CmpOp::Ne, "ne"),
        (CmpOp::Slt, "l"),
        (CmpOp::Sle, "le"),
        (CmpOp::Sgt, "g"),
        (CmpOp::Sge, "ge"),
        (CmpOp::Ult, "b"),
        (CmpOp::Ule, "be"),
        (CmpOp::Ugt, "a"),
        (CmpOp::Uge, "ae"),
    ] {
        for (kind, cmp) in [("i32", "cmp r10d, r11d"), ("i64", "cmp r10, r11")] {
            let lines = asm(&cases::compare(kind, op));
            expect(&lines, &[cmp, &format!("set{cc} r10b")], &[]);
        }
    }
    // Pointers compare unsigned.
    expect(
        &asm(&cases::compare("ptr", CmpOp::Ult)),
        &["cmp r10, r11", "setb r10b"],
        &["setl r10b"],
    );
}

#[test]
fn narrow_signed_comparisons_sign_extend_both_operands() {
    for op in [CmpOp::Slt, CmpOp::Sle, CmpOp::Sgt, CmpOp::Sge] {
        expect(
            &asm(&cases::compare("i8", op)),
            &["shl r10, 56", "sar r10, 56", "shl r11, 56", "sar r11, 56"],
            &[],
        );
        expect(
            &asm(&cases::compare("i48", op)),
            &["shl r10, 16", "sar r10, 16", "cmp r10, r11"],
            &[],
        );
    }
    for op in [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Ult,
        CmpOp::Ule,
        CmpOp::Ugt,
        CmpOp::Uge,
    ] {
        expect(
            &asm(&cases::compare("i8", op)),
            &[],
            &["sar r10, 56", "sar r11, 56"],
        );
    }
}

#[test]
fn integer_width_changes_pick_the_extension_by_conversion() {
    use ConvertKind::*;
    // 32 -> 64: a sign extension is `movsxd`; a zero extension needs no instruction at all,
    // because a 32-bit value is already held zero-extended.
    let sext = asm(&cases::convert("i32", "i64", Sext));
    let zext = asm(&cases::convert("i32", "i64", Zext));
    expect(&sext, &["movsxd r10, r10d"], &[]);
    expect(&zext, &[], &["movsxd r10, r10d"]);
    assert!(zext.len() < sext.len());
    // Narrower sources shift the sign (or zero) in.
    expect(
        &asm(&cases::convert("i8", "i64", Sext)),
        &["shl r10, 56", "sar r10, 56"],
        &[],
    );
    expect(
        &asm(&cases::convert("i8", "i64", Zext)),
        &[],
        &["sar r10, 56", "shr r10, 56"],
    );
    // A sign extension to a narrower-than-64 destination is zero-extended afterwards.
    expect(
        &asm(&cases::convert("i8", "i32", Sext)),
        &["shl r10, 56", "sar r10, 56", "mov r10d, r10d"],
        &[],
    );
    // Truncation keeps the low bits, zero-extended.
    expect(
        &asm(&cases::convert("i64", "i32", Trunc)),
        &["mov r10d, r10d"],
        &[],
    );
    expect(
        &asm(&cases::convert("i64", "i8", Trunc)),
        &["shl r10, 56", "shr r10, 56"],
        &["sar r10, 56"],
    );
    expect(
        &asm(&cases::convert("bool", "i64", Zext)),
        &[],
        &["sar r10, 56", "shr r10, 56"],
    );
}

#[test]
fn integer_to_float_conversions_pick_the_signedness_by_conversion() {
    use ConvertKind::*;
    // 64-bit: the unsigned conversion needs the halve-and-double path for a set top bit.
    let signed = asm(&cases::convert("i64", "f64", SiToFp));
    let unsigned = asm(&cases::convert("i64", "f64", UiToFp));
    expect(&signed, &["cvtsi2sd xmm13, r10"], &["test r10, r10"]);
    expect(
        &unsigned,
        &["cvtsi2sd xmm13, r10", "test r10, r10", "addsd xmm13, xmm13"],
        &[],
    );
    expect(
        &asm(&cases::convert("i64", "f32", UiToFp)),
        &["cvtsi2ss xmm13, r10", "test r10, r10", "addss xmm13, xmm13"],
        &[],
    );
    // 32-bit: the signed conversion sign-extends first, the unsigned one converts the
    // zero-extended value as a (non-negative) 64-bit integer.
    let signed = asm(&cases::convert("i32", "f64", SiToFp));
    let unsigned = asm(&cases::convert("i32", "f64", UiToFp));
    expect(&signed, &["movsxd r10, r10d", "cvtsi2sd xmm13, r10"], &[]);
    expect(
        &unsigned,
        &["cvtsi2sd xmm13, r10"],
        &["movsxd r10, r10d", "test r10, r10"],
    );
    // Narrow: sign-extended by shifts for the signed conversion only.
    expect(
        &asm(&cases::convert("i8", "f64", SiToFp)),
        &["shl r10, 56", "sar r10, 56"],
        &[],
    );
    expect(
        &asm(&cases::convert("i8", "f64", UiToFp)),
        &["cvtsi2sd xmm13, r10"],
        &["sar r10, 56"],
    );
}

#[test]
fn float_to_integer_conversions_pick_the_signedness_by_conversion() {
    use ConvertKind::*;
    // 64-bit unsigned results above 2^63 take the subtract-and-flip path.
    let signed = asm(&cases::convert("f64", "i64", FpToSi));
    let unsigned = asm(&cases::convert("f64", "i64", FpToUi));
    expect(
        &signed,
        &["cvttsd2si r10, xmm13"],
        &["ucomisd xmm13, xmm14"],
    );
    expect(
        &unsigned,
        &[
            "cvttsd2si r10, xmm13",
            "ucomisd xmm13, xmm14",
            "subsd xmm13, xmm14",
        ],
        &[],
    );
    expect(
        &asm(&cases::convert("f32", "i64", FpToUi)),
        &[
            "cvttss2si r10, xmm13",
            "ucomiss xmm13, xmm14",
            "subss xmm13, xmm14",
        ],
        &[],
    );
    // A signed 32-bit result is a 32-bit truncating convert; an unsigned one is a 64-bit
    // convert (so values past 2^31 survive) whose low 32 bits are kept.
    expect(
        &asm(&cases::convert("f64", "i32", FpToSi)),
        &["cvttsd2si r10d, xmm13", "mov r10d, r10d"],
        &["cvttsd2si r10, xmm13"],
    );
    expect(
        &asm(&cases::convert("f64", "i32", FpToUi)),
        &["cvttsd2si r10, xmm13", "mov r10d, r10d"],
        &["cvttsd2si r10d, xmm13"],
    );
}

#[test]
fn float_resize_converts_between_widths() {
    use ConvertKind::*;
    expect(
        &asm(&cases::convert("f32", "f64", FpResize)),
        &["cvtss2sd xmm13, xmm13"],
        &["cvtsd2ss xmm13, xmm13"],
    );
    expect(
        &asm(&cases::convert("f64", "f32", FpResize)),
        &["cvtsd2ss xmm13, xmm13"],
        &["cvtss2sd xmm13, xmm13"],
    );
}

#[test]
fn loads_zero_extend_and_never_sign_extend() {
    for (kind, load) in [
        ("i8", "movzx r10, byte ptr [r11]"),
        ("i16", "movzx r10, word ptr [r11]"),
        ("i32", "mov r10d, dword ptr [r11]"),
        ("i64", "mov r10, qword ptr [r11]"),
    ] {
        let mut f = Function::new();
        let p = f.types.parse_type("ptr").unwrap();
        let t = f.types.parse_type(kind).unwrap();
        let b = f.append_block();
        let ptr = f.append_block_param(b, p);
        let v = f.append_load(b, t, ptr, MemFlags::new());
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        let lines = asm(&f);
        expect(&lines, &[load], &[]);
        assert!(
            lines.iter().all(|l| !l.starts_with("movsx")),
            "{kind}:\n{}",
            lines.join("\n")
        );
    }
}

#[test]
fn narrow_arguments_and_call_results_are_zero_extended() {
    // An argument is reloaded from its home slot at its declared width, which discards
    // whatever the caller left above it.
    let lines = asm(&cases::arith("i8", BinOp::Add));
    let loads = lines
        .iter()
        .filter(|l| l.starts_with("movzx r10, byte ptr [rbp - "))
        .count();
    assert_eq!(loads, 2, "{lines:?}");
    // A call result is cleaned before use: the ABI leaves the upper bits undefined.
    let mut f = Function::new();
    let byte = f.types.parse_type("i8").unwrap();
    let p = f.types.parse_type("ptr").unwrap();
    let b = f.append_block();
    let target = f.append_block_param(b, p);
    let args = f.intern_values(&[]);
    let r = f.append_inst(
        b,
        byte,
        Opcode::CallIndirect(CallIndirect {
            target,
            args,
            is_variadic: false,
            num_fixed: 0,
            ret_dest: None,
            ret_regs: 0,
            ret_pieces: [RetPiece::default(); 4],
            sret: false,
        }),
    );
    f.set_terminator(b, Terminator::Ret(Ret::one(r)));
    expect(&asm(&f), &["shl rax, 56", "shr rax, 56"], &[]);
}

#[test]
fn immediates_are_truncated_to_the_operand_width() {
    // `udiv i8 x, -56` divides by 200, not by 2^64 - 56.
    let lines = asm(&cases::arith_imm("i8", BinOp::UDiv, -56));
    expect(&lines, &["mov r11, 200"], &[]);
    let lines = asm(&cases::arith_imm("i32", BinOp::UDiv, -3));
    expect(&lines, &["mov r11, 4294967293"], &[]);
    let lines = asm(&cases::arith_imm("i64", BinOp::UDiv, -3));
    expect(&lines, &["mov r11, 18446744073709551613"], &[]);
}

fn single(kind: &str, op: impl FnOnce(Value, Value) -> Opcode, ret: &str) -> Function {
    let mut f = Function::new();
    let t = f.types.parse_type(kind).unwrap();
    let r = f.types.parse_type(ret).unwrap();
    let b = f.append_block();
    let x = f.append_block_param(b, t);
    let y = f.append_block_param(b, t);
    let v = f.append_inst(b, r, op(x, y));
    f.set_terminator(b, Terminator::Ret(Ret::one(v)));
    f
}

#[test]
fn operations_applied_to_the_wrong_class_fail_closed() {
    use Opcode::{Arith as A, Icmp};
    let arith = |op| move |lhs, rhs| A(Arith { op, lhs, rhs });
    let cmp = |op| move |lhs, rhs| Icmp(Compare { op, lhs, rhs });
    // The floating-point division on integers, the integer divisions on floats.
    for op in [BinOp::Div, BinOp::Rem] {
        let f = single("i32", arith(op), "i32");
        assert_eq!(compile(&f), Err(Error::InvalidFunction), "{op:?}");
    }
    for op in [
        BinOp::UDiv,
        BinOp::SDiv,
        BinOp::URem,
        BinOp::SRem,
        BinOp::Shr,
        BinOp::Sar,
    ] {
        let f = single("f64", arith(op), "f64");
        assert!(matches!(compile(&f), Err(Error::Unsupported(_))), "{op:?}");
    }
    // The float orders on integers, the integer orders on floats.
    for op in [CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge] {
        let f = single("i32", cmp(op), "bool");
        assert_eq!(compile(&f), Err(Error::InvalidFunction), "{op:?}");
    }
    for op in [CmpOp::Slt, CmpOp::Ult, CmpOp::Sge, CmpOp::Uge] {
        let f = single("f64", cmp(op), "bool");
        assert_eq!(compile(&f), Err(Error::InvalidFunction), "{op:?}");
    }
    // An integer conversion of a float, a float conversion of the wrong class.
    use ConvertKind::*;
    for (from, to, kind) in [
        ("f64", "i64", Zext),
        ("i32", "i64", FpResize),
        ("i32", "f64", FpToSi),
        ("f64", "f32", SiToFp),
    ] {
        let f = cases::convert(from, to, kind);
        assert!(compile(&f).is_err(), "{from} {kind:?} {to}");
    }
}

#[test]
fn integer_dot_products_are_not_lowered() {
    let mut f = Function::new();
    let vector = f.types.parse_type("<4 x i32>").unwrap();
    let int = f.types.parse_type("i32").unwrap();
    let b = f.append_block();
    let acc = f.append_block_param(b, int);
    let x = f.append_block_param(b, vector);
    let y = f.append_block_param(b, vector);
    for signed in [false, true] {
        let mut f = f.clone_func();
        let v = f.append_dot(b, acc, x, y, signed);
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        assert!(
            matches!(compile(&f), Err(Error::Unsupported(_))),
            "signed={signed}"
        );
    }
}
