// SPDX-License-Identifier: Apache-2.0
//! Execution cases for the operations whose result depends on signedness. Each [`Case`] is one
//! function with the arguments to call it with and the value a correct execution returns; the
//! expected values come from the plain Rust reference arithmetic below, written independently
//! of the instruction selector. `tests.rs` executes the cases natively; `codegen_tests.rs`
//! compiles them on every host and checks the machine code decodes.
//!
//! Integer registers hold values narrower than 64 bits zero-extended, so a correct function
//! also returns them zero-extended, and ignores whatever the caller left above the declared
//! width of an argument. The cases pin both: arguments carry dirty upper bits on purpose and
//! results are compared exactly.
use alloc::{format, string::String, vec, vec::Vec};
use volt_ir::function::*;

/// An argument or return value.
#[derive(Clone, Copy, Debug)]
pub enum Val {
    Int(u64),
    F32(f32),
    F64(f64),
}

impl Val {
    /// The raw register contents (an `f32` in the low 32 bits).
    pub fn bits(self) -> u64 {
        match self {
            Val::Int(v) => v,
            Val::F32(v) => v.to_bits() as u64,
            Val::F64(v) => v.to_bits(),
        }
    }
}

pub struct Case {
    pub name: String,
    pub function: Function,
    /// `(arguments, expected result)` pairs.
    pub runs: Vec<(Vec<Val>, Val)>,
}

pub fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

/// `v` read as a two's complement integer of `bits` bits.
pub fn sext(v: u64, bits: u32) -> i64 {
    let s = 64 - bits;
    ((v << s) as i64) >> s
}

/// `a` with the bits above `bits` filled with garbage when `dirty` is odd.
fn dirty(a: u64, bits: u32, k: usize) -> u64 {
    if k & 1 == 1 { a | !mask(bits) } else { a }
}

fn build(
    params: &[&str],
    ret: &str,
    body: impl FnOnce(&mut Function, Block, &[Value], volt_ir::types::Type) -> Value,
) -> Function {
    let mut f = Function::new();
    let b = f.append_block();
    let args: Vec<Value> = params
        .iter()
        .map(|p| {
            let t = f.types.parse_type(p).unwrap();
            f.append_block_param(b, t)
        })
        .collect();
    let t = f.types.parse_type(ret).unwrap();
    let v = body(&mut f, b, &args, t);
    f.set_terminator(b, Terminator::Ret(Ret::one(v)));
    f
}

pub fn arith(kind: &str, op: BinOp) -> Function {
    build(&[kind, kind], kind, |f, b, a, t| {
        f.append_inst(
            b,
            t,
            Opcode::Arith(Arith {
                op,
                lhs: a[0],
                rhs: a[1],
            }),
        )
    })
}

pub fn arith_imm(kind: &str, op: BinOp, imm: i64) -> Function {
    build(&[kind], kind, |f, b, a, t| {
        f.append_inst(b, t, Opcode::ArithImm(ArithImm { op, lhs: a[0], imm }))
    })
}

pub fn compare(kind: &str, op: CmpOp) -> Function {
    build(&[kind, kind], "bool", |f, b, a, t| {
        f.append_inst(
            b,
            t,
            Opcode::Icmp(Compare {
                op,
                lhs: a[0],
                rhs: a[1],
            }),
        )
    })
}

pub fn convert(from: &str, to: &str, kind: ConvertKind) -> Function {
    build(&[from], to, |f, b, a, t| {
        f.append_inst(b, t, Opcode::Convert(Convert { value: a[0], kind }))
    })
}

/// The value of `op` on two `bits`-bit integers, or `None` where the hardware traps (a zero
/// divisor, or the most negative value divided by -1 at the 32 and 64 bit widths).
pub fn arith_reference(op: BinOp, bits: u32, a: u64, b: u64) -> Option<u64> {
    let (a, b) = (a & mask(bits), b & mask(bits));
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    let trapping_signed = (bits == 32 || bits == 64) && sa == i64::MIN >> (64 - bits) && sb == -1;
    let r = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a << b,
        BinOp::Shr => a >> b,
        BinOp::Sar => (sa >> b) as u64,
        BinOp::UDiv if b != 0 => a / b,
        BinOp::URem if b != 0 => a % b,
        BinOp::SDiv if b != 0 && !trapping_signed => sa.wrapping_div(sb) as u64,
        BinOp::SRem if b != 0 && !trapping_signed => sa.wrapping_rem(sb) as u64,
        BinOp::UMulh => ((a as u128 * b as u128) >> bits) as u64,
        BinOp::SMulh => ((sa as i128 * sb as i128) >> bits) as u64,
        _ => return None,
    };
    Some(r & mask(bits))
}

pub const WIDTHS: [u32; 6] = [8, 16, 32, 33, 48, 64];

pub fn samples(bits: u32) -> Vec<u64> {
    let m = mask(bits);
    let mut v = vec![
        0,
        1,
        2,
        3,
        5,
        7,
        100,
        200,
        m,
        m - 1,
        m >> 1,
        (m >> 1) + 1,
        ((m >> 1) + 1) | 1,
        0x5555_5555_5555_5555 & m,
        0xaaaa_aaaa_aaaa_aaaa & m,
        0x0123_4567_89ab_cdef & m,
    ];
    for x in &mut v {
        *x &= m;
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn is_shift(op: BinOp) -> bool {
    matches!(op, BinOp::Shl | BinOp::Shr | BinOp::Sar)
}

/// Constants: the payload is any `i64`, and the register must hold its low `bits`
/// zero-extended, whatever the sign of the payload.
fn iconst_cases(out: &mut Vec<Case>) {
    for bits in WIDTHS {
        let kind = format!("i{bits}");
        for n in [
            -1i64,
            -2,
            1,
            -100,
            127,
            -128,
            255,
            0x1234_5678_9abc_def0,
            i64::MIN,
        ] {
            out.push(Case {
                name: format!("{kind} Iconst {n}"),
                function: build(&[], &kind, |f, b, _, t| {
                    f.append_inst(b, t, Opcode::Iconst(n))
                }),
                runs: vec![(vec![], Val::Int(n as u64 & mask(bits)))],
            });
        }
    }
}

fn arith_cases(out: &mut Vec<Case>) {
    for bits in WIDTHS {
        let kind = format!("i{bits}");
        for op in [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::BitAnd,
            BinOp::BitOr,
            BinOp::BitXor,
            BinOp::Shl,
            BinOp::Shr,
            BinOp::Sar,
            BinOp::UDiv,
            BinOp::SDiv,
            BinOp::URem,
            BinOp::SRem,
            BinOp::UMulh,
            BinOp::SMulh,
        ] {
            let mut runs = Vec::new();
            let mut k = 0;
            for &a in &samples(bits) {
                for &b in &samples(bits) {
                    // A shift amount is only meaningful below the width.
                    let b = if is_shift(op) { b % bits as u64 } else { b };
                    if let Some(want) = arith_reference(op, bits, a, b) {
                        k += 1;
                        runs.push((
                            vec![Val::Int(dirty(a, bits, k)), Val::Int(dirty(b, bits, k / 2))],
                            Val::Int(want),
                        ));
                    }
                }
            }
            out.push(Case {
                name: format!("{kind} {op:?}"),
                function: arith(&kind, op),
                runs,
            });
        }
    }
}

/// `op` against a constant: the immediate is the value an `Iconst` of the operand type would
/// hold, so a negative immediate on a narrow type means its unsigned reading.
fn immediate_cases(out: &mut Vec<Case>) {
    for (bits, op, imm) in [
        (8, BinOp::Add, -1),
        (8, BinOp::UDiv, -56),
        (8, BinOp::URem, -56),
        (8, BinOp::SDiv, -3),
        (8, BinOp::SRem, -3),
        (8, BinOp::Shl, 3),
        (8, BinOp::Shr, 3),
        (8, BinOp::Sar, 3),
        (16, BinOp::UDiv, -1000),
        (16, BinOp::SRem, -1000),
        (16, BinOp::Sar, 15),
        (32, BinOp::UDiv, -3),
        (32, BinOp::SDiv, -3),
        (32, BinOp::SRem, -3),
        (32, BinOp::Sar, 31),
        (32, BinOp::Shr, 31),
        (48, BinOp::UDiv, -3),
        (48, BinOp::SDiv, -3),
        (48, BinOp::Sar, 47),
        (64, BinOp::SDiv, -3),
        (64, BinOp::UDiv, -3),
        (64, BinOp::Sar, 63),
        (64, BinOp::Shr, 63),
    ] {
        let kind = format!("i{bits}");
        let mut runs = Vec::new();
        for (k, &a) in samples(bits).iter().enumerate() {
            if let Some(want) = arith_reference(op, bits, a, imm as u64) {
                runs.push((vec![Val::Int(dirty(a, bits, k))], Val::Int(want)));
            }
        }
        out.push(Case {
            name: format!("{kind} {op:?} imm {imm}"),
            function: arith_imm(&kind, op, imm),
            runs,
        });
    }
}

pub fn compare_reference(op: CmpOp, bits: u32, a: u64, b: u64) -> bool {
    let (a, b) = (a & mask(bits), b & mask(bits));
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Slt => sa < sb,
        CmpOp::Sle => sa <= sb,
        CmpOp::Sgt => sa > sb,
        CmpOp::Sge => sa >= sb,
        CmpOp::Ult => a < b,
        CmpOp::Ule => a <= b,
        CmpOp::Ugt => a > b,
        CmpOp::Uge => a >= b,
        CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => unreachable!("float compare"),
    }
}

fn compare_cases(out: &mut Vec<Case>) {
    let signed = [CmpOp::Slt, CmpOp::Sle, CmpOp::Sgt, CmpOp::Sge];
    let unsigned = [CmpOp::Ult, CmpOp::Ule, CmpOp::Ugt, CmpOp::Uge];
    let equality = [CmpOp::Eq, CmpOp::Ne];
    // `ptr` is an unsigned 64-bit quantity: it has no signed order.
    for kind in WIDTHS
        .map(|b| format!("i{b}"))
        .into_iter()
        .chain(["ptr".into()])
    {
        let bits = kind[1..].parse().unwrap_or(64);
        let ops: Vec<CmpOp> = if kind == "ptr" {
            equality.iter().chain(&unsigned).copied().collect()
        } else {
            equality
                .iter()
                .chain(&signed)
                .chain(&unsigned)
                .copied()
                .collect()
        };
        for op in ops {
            let mut runs = Vec::new();
            let mut k = 0;
            for &a in &samples(bits) {
                for &b in &samples(bits) {
                    k += 1;
                    runs.push((
                        vec![Val::Int(dirty(a, bits, k)), Val::Int(dirty(b, bits, k / 3))],
                        Val::Int(compare_reference(op, bits, a, b) as u64),
                    ));
                }
            }
            out.push(Case {
                name: format!("{kind} {op:?}"),
                function: compare(&kind, op),
                runs,
            });
        }
    }
}

fn integer_convert_cases(out: &mut Vec<Case>) {
    // (from, to) widths of the integer width changes.
    let trunc = [
        (64, 32),
        (64, 8),
        (64, 16),
        (64, 48),
        (48, 8),
        (33, 8),
        (32, 16),
        (32, 8),
        (16, 8),
    ];
    let ext = [
        (8, 16),
        (8, 32),
        (8, 48),
        (8, 64),
        (16, 32),
        (16, 64),
        (32, 33),
        (32, 64),
        (33, 64),
        (48, 64),
    ];
    for (kind, pairs) in [
        (ConvertKind::Trunc, &trunc[..]),
        (ConvertKind::Zext, &ext[..]),
        (ConvertKind::Sext, &ext[..]),
    ] {
        for &(from, to) in pairs {
            let mut runs = Vec::new();
            for (k, &a) in samples(from).iter().enumerate() {
                let want = match kind {
                    ConvertKind::Trunc => a & mask(to),
                    ConvertKind::Zext => a,
                    _ => sext(a, from) as u64 & mask(to),
                };
                runs.push((vec![Val::Int(dirty(a, from, k))], Val::Int(want)));
            }
            out.push(Case {
                name: format!("i{from} {kind:?} i{to}"),
                function: convert(&format!("i{from}"), &format!("i{to}"), kind),
                runs,
            });
        }
    }
    for to in [32, 64] {
        out.push(Case {
            name: format!("bool Zext i{to}"),
            function: convert("bool", &format!("i{to}"), ConvertKind::Zext),
            runs: vec![
                (vec![Val::Int(0)], Val::Int(0)),
                (vec![Val::Int(1)], Val::Int(1)),
                // Only the declared bit of the argument counts.
                (vec![Val::Int(0xffff_ff00)], Val::Int(0)),
                (vec![Val::Int(0xffff_ff01)], Val::Int(1)),
            ],
        });
    }
}

fn float_convert_cases(out: &mut Vec<Case>) {
    for bits in [8, 16, 32, 48, 64] {
        for (fkind, double) in [("f64", true), ("f32", false)] {
            for kind in [ConvertKind::SiToFp, ConvertKind::UiToFp] {
                let mut runs = Vec::new();
                for (k, &a) in samples(bits).iter().enumerate() {
                    let want = match (kind == ConvertKind::SiToFp, double) {
                        (true, true) => Val::F64(sext(a, bits) as f64),
                        (true, false) => Val::F32(sext(a, bits) as f32),
                        (false, true) => Val::F64(a as f64),
                        (false, false) => Val::F32(a as f32),
                    };
                    runs.push((vec![Val::Int(dirty(a, bits, k))], want));
                }
                out.push(Case {
                    name: format!("i{bits} {kind:?} {fkind}"),
                    function: convert(&format!("i{bits}"), fkind, kind),
                    runs,
                });
            }
        }
    }
    // In-range sources only: out-of-range results are the hardware's "integer indefinite".
    let signed_sources = |bits: u32| -> Vec<f64> {
        match bits {
            8 => vec![0.0, 1.5, -1.5, 100.9, -100.9, 127.0, -128.0],
            16 => vec![0.0, 1.5, -1.5, 30000.7, -30000.7, 32767.0, -32768.0],
            32 => vec![0.0, 1.5, -1.5, 2.0e9, -2.0e9, 2147483520.0, -2147483648.0],
            _ => vec![
                0.0,
                1.5,
                -1.5,
                1.0e15,
                -1.0e15,
                9.0e18,
                -9.0e18,
                -9223372036854775808.0,
            ],
        }
    };
    let unsigned_sources = |bits: u32| -> Vec<f64> {
        match bits {
            8 => vec![0.0, 1.9, 200.5, 255.0],
            16 => vec![0.0, 1.9, 40000.5, 65535.0],
            32 => vec![0.0, 0.5, 3.0e9, 4294967295.0],
            _ => vec![
                0.0,
                1.5,
                9.0e18,
                9223372036854775807.0,
                9223372036854775808.0,
                1.8e19,
                18446744073709549568.0,
            ],
        }
    };
    for bits in [8, 16, 32, 64] {
        for (fkind, double) in [("f64", true), ("f32", false)] {
            for kind in [ConvertKind::FpToSi, ConvertKind::FpToUi] {
                let sources = if kind == ConvertKind::FpToSi {
                    signed_sources(bits)
                } else {
                    unsigned_sources(bits)
                };
                let mut runs = Vec::new();
                for v in sources {
                    // Round the source to the source type first; the reference then sees
                    // exactly the value the function is given.
                    let (arg, truncated) = if double {
                        (Val::F64(v), v)
                    } else {
                        (Val::F32(v as f32), v as f32 as f64)
                    };
                    let half = 2f64.powi(bits as i32 - 1);
                    let in_range = if kind == ConvertKind::FpToSi {
                        truncated >= -half && truncated < half
                    } else {
                        truncated > -1.0 && truncated < 2.0 * half
                    };
                    if !in_range {
                        continue;
                    }
                    let want = if kind == ConvertKind::FpToSi {
                        truncated as i64 as u64 & mask(bits)
                    } else {
                        truncated as u64 & mask(bits)
                    };
                    runs.push((vec![arg], Val::Int(want)));
                }
                out.push(Case {
                    name: format!("{fkind} {kind:?} i{bits}"),
                    function: convert(fkind, &format!("i{bits}"), kind),
                    runs,
                });
            }
        }
    }
    out.push(Case {
        name: "f32 FpResize f64".into(),
        function: convert("f32", "f64", ConvertKind::FpResize),
        runs: [0.0f32, 1.5, -2.25, 1.0e10, 3.0e38, 1.0e-30]
            .map(|v| (vec![Val::F32(v)], Val::F64(v as f64)))
            .into(),
    });
    out.push(Case {
        name: "f64 FpResize f32".into(),
        function: convert("f64", "f32", ConvertKind::FpResize),
        runs: [0.0f64, 1.5, -2.25, 1.0e10, 3.0e38, 0.1]
            .map(|v| (vec![Val::F64(v)], Val::F32(v as f32)))
            .into(),
    });
}

/// `<4 x i32>` lanes: `lhs = [a0, a1, a1, a0]`, `rhs = [b0, b1, b0, b1]`, and the function
/// returns lane 0 | lane 2 << 32 of `lhs <op> rhs`.
fn vector_function(op: Result<BinOp, CmpOp>) -> Function {
    build(&["i32", "i32", "i32", "i32"], "i64", |f, b, p, wide| {
        let vector = f.types.parse_type("<4 x i32>").unwrap();
        let int = f.types.parse_type("i32").unwrap();
        let fields = f.intern_values(&[p[0], p[1], p[1], p[0]]);
        let lhs = f.append_inst(b, vector, Opcode::StructNew(StructNew { fields }));
        let fields = f.intern_values(&[p[2], p[3], p[2], p[3]]);
        let rhs = f.append_inst(b, vector, Opcode::StructNew(StructNew { fields }));
        let v = match op {
            Ok(op) => f.append_inst(b, vector, Opcode::Arith(Arith { op, lhs, rhs })),
            Err(op) => f.append_inst(b, vector, Opcode::Icmp(Compare { op, lhs, rhs })),
        };
        let lane = |f: &mut Function, index| {
            let x = f.append_inst(
                b,
                int,
                Opcode::Extract(Extract {
                    aggregate: v,
                    index,
                }),
            );
            f.append_inst(
                b,
                wide,
                Opcode::Convert(Convert {
                    value: x,
                    kind: ConvertKind::Zext,
                }),
            )
        };
        let low = lane(f, 0);
        let high = lane(f, 2);
        let high = f.append_inst(
            b,
            wide,
            Opcode::ArithImm(ArithImm {
                op: BinOp::Shl,
                lhs: high,
                imm: 32,
            }),
        );
        f.append_inst(
            b,
            wide,
            Opcode::Arith(Arith {
                op: BinOp::BitOr,
                lhs: low,
                rhs: high,
            }),
        )
    })
}

fn vector_cases(out: &mut Vec<Case>) {
    let lanes: [u64; 6] = [0, 1, 7, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
    let combos = || {
        lanes.into_iter().flat_map(move |a0| {
            lanes.into_iter().flat_map(move |a1| {
                lanes
                    .into_iter()
                    .flat_map(move |b0| lanes.into_iter().map(move |b1| [a0, a1, b0, b1]))
            })
        })
    };
    for op in [
        BinOp::Add,
        BinOp::Mul,
        BinOp::Shr,
        BinOp::Sar,
        BinOp::UDiv,
        BinOp::SDiv,
        BinOp::URem,
        BinOp::SRem,
        BinOp::UMulh,
        BinOp::SMulh,
    ] {
        let mut runs = Vec::new();
        for [a0, a1, b0, b1] in combos() {
            let (b0, b1) = if is_shift(op) {
                (b0 % 32, b1 % 32)
            } else {
                (b0, b1)
            };
            // Every lane executes, so a lane that would trap rules the whole input out.
            let lanes = [(a0, b0), (a1, b1), (a1, b0), (a0, b1)]
                .map(|(a, b)| arith_reference(op, 32, a, b));
            if let [Some(l0), Some(_), Some(l2), Some(_)] = lanes {
                runs.push((
                    [a0, a1, b0, b1].map(Val::Int).into(),
                    Val::Int(l0 | l2 << 32),
                ));
            }
        }
        out.push(Case {
            name: format!("<4 x i32> {op:?}"),
            function: vector_function(Ok(op)),
            runs,
        });
    }
    for op in [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Slt,
        CmpOp::Sle,
        CmpOp::Sgt,
        CmpOp::Sge,
        CmpOp::Ult,
        CmpOp::Ule,
        CmpOp::Ugt,
        CmpOp::Uge,
    ] {
        let lane = |a, b| {
            if compare_reference(op, 32, a, b) {
                0xffff_ffffu64
            } else {
                0
            }
        };
        let runs = combos()
            .map(|[a0, a1, b0, b1]| {
                (
                    [a0, a1, b0, b1].map(Val::Int).into(),
                    Val::Int(lane(a0, b0) | lane(a1, b0) << 32),
                )
            })
            .collect();
        out.push(Case {
            name: format!("<4 x i32> {op:?}"),
            function: vector_function(Err(op)),
            runs,
        });
    }
}

pub fn all() -> Vec<Case> {
    let mut out = Vec::new();
    arith_cases(&mut out);
    iconst_cases(&mut out);
    immediate_cases(&mut out);
    compare_cases(&mut out);
    integer_convert_cases(&mut out);
    float_convert_cases(&mut out);
    vector_cases(&mut out);
    out
}
