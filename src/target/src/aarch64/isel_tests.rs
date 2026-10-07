// Rust selector boundary tests for the Vulcan-derived AArch64 port (Apache-2.0).
use super::*;
use volt_ir::function::{
    Arith, BinOp, CmpOp, Compare, EdgeDesc, Function, Load, Opcode, Ret, Terminator,
};
use volt_ir::types::{FloatKind, IntDesc, Type, TypeKind};

fn integer(f: &mut Function, signed: bool, bits: u16) -> Type {
    f.types.intern(TypeKind::Int(IntDesc { signed, bits }))
}

#[test]
fn emits_native_divide_remainder_and_high_product() {
    for signed in [false, true] {
        for op in [BinOp::Div, BinOp::Rem, BinOp::Mulh] {
            let mut f = Function::new();
            let ty = integer(&mut f, signed, 64);
            let b = f.append_block();
            let a = f.append_block_param(b, ty);
            let c = f.append_block_param(b, ty);
            let result = f.append_inst(b, ty, Opcode::Arith(Arith { op, lhs: a, rhs: c }));
            f.set_terminator(b, Terminator::Ret(Ret::one(result)));
            let code = compile(&f).unwrap();
            assert_eq!(code.last(), Some(&crate::aarch64::encode::ret()));
            let opcode = if op == BinOp::Mulh {
                if signed { 0x9b407c00 } else { 0x9bc07c00 }
            } else if signed {
                0x9ac00c00
            } else {
                0x9ac00800
            };
            assert!(code.iter().any(|w| w & 0xffe0fc00 == opcode));
        }
    }
}

#[test]
fn scalar_half_and_quad_data_survive_selection() {
    for fp in [
        FloatKind::F16,
        FloatKind::F32,
        FloatKind::F64,
        FloatKind::F128,
    ] {
        let mut f = Function::new();
        let ty = f.types.intern(TypeKind::Float(fp));
        let b = f.append_block();
        let v = f.append_block_param(b, ty);
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        let code = compile(&f).unwrap();
        assert!(!code.is_empty());
    }
}

#[test]
fn object_calls_keep_owned_relocations() {
    let mut f = Function::new();
    let ty = integer(&mut f, false, 64);
    let b = f.append_block();
    let value = f.append_call(b, ty, "external_target", &[]);
    f.set_terminator(b, Terminator::Ret(Ret::one(value)));
    let compiled = compile_function(&f, &ModelCaps::default()).unwrap();
    assert_eq!(compiled.relocs.len(), 1);
    assert_eq!(compiled.relocs[0].symbol, "external_target");
    assert_eq!(compiled.relocs[0].kind, RelocKind::Call);
}

#[cfg(target_arch = "aarch64")]
fn mapped(f: &Function) -> crate::native::JittedFunction {
    let code = compile(f).unwrap();
    let bytes: Vec<u8> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
    crate::native::map_code(bytes).unwrap()
}

#[cfg(target_arch = "aarch64")]
#[test]
fn division_edges_execute_with_aarch64_semantics() {
    for signed in [false, true] {
        for op in [BinOp::Div, BinOp::Rem] {
            let mut f = Function::new();
            let ty = integer(&mut f, signed, 64);
            let b = f.append_block();
            let a = f.append_block_param(b, ty);
            let c = f.append_block_param(b, ty);
            let result = f.append_inst(b, ty, Opcode::Arith(Arith { op, lhs: a, rhs: c }));
            f.set_terminator(b, Terminator::Ret(Ret::one(result)));
            let image = mapped(&f);
            // SAFETY: function is generated for precisely this AAPCS64 signature.
            let entry: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { image.entry() };
            let zero = unsafe { entry(123, 0) };
            assert_eq!(zero, if op == BinOp::Div { 0 } else { 123 });
            if signed {
                let overflow = unsafe { entry(i64::MIN as u64, u64::MAX) };
                assert_eq!(overflow, if op == BinOp::Div { i64::MIN as u64 } else { 0 });
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn loop_edges_execute_parallel_parameter_permutations() {
    let mut f = Function::new();
    let ty = integer(&mut f, false, 64);
    let bool_ty = f.types.intern(TypeKind::Bool);
    let entry = f.append_block();
    let head = f.append_block();
    let body = f.append_block();
    let done = f.append_block();
    let n = f.append_block_param(entry, ty);
    let a = f.append_block_param(entry, ty);
    let b = f.append_block_param(entry, ty);
    let hn = f.append_block_param(head, ty);
    let ha = f.append_block_param(head, ty);
    let hb = f.append_block_param(head, ty);
    f.set_jump(entry, head, &[n, a, b]);
    let zero = f.append_inst(head, ty, Opcode::Iconst(0));
    let cond = f.append_inst(
        head,
        bool_ty,
        Opcode::Icmp(Compare {
            op: CmpOp::Ne,
            lhs: hn,
            rhs: zero,
        }),
    );
    f.append_if(head, cond, EdgeDesc::bare(body), EdgeDesc::bare(done));
    let one = f.append_inst(body, ty, Opcode::Iconst(1));
    let next = f.append_inst(
        body,
        ty,
        Opcode::Arith(Arith {
            op: BinOp::Sub,
            lhs: hn,
            rhs: one,
        }),
    );
    f.set_jump(body, head, &[next, hb, ha]);
    f.set_terminator(done, Terminator::Ret(Ret::one(ha)));
    let image = mapped(&f);
    let entry: unsafe extern "C" fn(u64, u64, u64) -> u64 = unsafe { image.entry() };
    assert_eq!(unsafe { entry(7, 11, 29) }, 29);
    assert_eq!(unsafe { entry(8, 11, 29) }, 11);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn signed_byte_load_widens_without_overreading() {
    let mut f = Function::new();
    let ptr = f.types.ptr_global();
    let byte = integer(&mut f, true, 8);
    let wide = integer(&mut f, true, 64);
    let b = f.append_block();
    let p = f.append_block_param(b, ptr);
    let value = f.append_inst(
        b,
        byte,
        Opcode::Load(Load {
            ptr: p,
            volatile: true,
        }),
    );
    let value = f.append_inst(
        b,
        wide,
        Opcode::Convert(volt_ir::function::Convert { value }),
    );
    f.set_terminator(b, Terminator::Ret(Ret::one(value)));
    let image = mapped(&f);
    let entry: unsafe extern "C" fn(*const i8) -> i64 = unsafe { image.entry() };
    let byte = -97i8;
    assert_eq!(unsafe { entry(&byte) }, -97);
}

/// A compiled `u64 -> u64` function and the mapping that keeps it alive.
#[cfg(target_arch = "aarch64")]
struct NarrowOp {
    _image: crate::native::JittedFunction,
    call: unsafe extern "C" fn(u64) -> u64,
}

/// `x -> convert(convert(x to ty) op k)`, with `k` either an `ArithImm` or a
/// register operand from an `Iconst`, to run both lowerings on the same input.
#[cfg(target_arch = "aarch64")]
fn narrow_op(signed: bool, bits: u16, op: BinOp, k: i64, immediate: bool) -> NarrowOp {
    let mut f = Function::new();
    let wide = integer(&mut f, false, 64);
    let ty = integer(&mut f, signed, bits);
    let b = f.append_block();
    let x = f.append_block_param(b, wide);
    let narrow = f.append_inst(
        b,
        ty,
        Opcode::Convert(volt_ir::function::Convert { value: x }),
    );
    let result = if immediate {
        f.append_arith_imm(b, ty, op, narrow, k)
    } else {
        let k = f.append_inst(b, ty, Opcode::Iconst(k));
        f.append_inst(
            b,
            ty,
            Opcode::Arith(Arith {
                op,
                lhs: narrow,
                rhs: k,
            }),
        )
    };
    let back = f.append_inst(
        b,
        wide,
        Opcode::Convert(volt_ir::function::Convert { value: result }),
    );
    f.set_terminator(b, Terminator::Ret(Ret::one(back)));
    let image = mapped(&f);
    // SAFETY: generated for exactly this AAPCS64 signature; `image` outlives the call.
    let call = unsafe { image.entry() };
    NarrowOp {
        _image: image,
        call,
    }
}

/// Whether `k` is a value of the integer type, so both lowerings see the same constant.
#[cfg(target_arch = "aarch64")]
fn representable(signed: bool, bits: u16, k: i64) -> bool {
    if bits >= 64 {
        return true;
    }
    if signed {
        let half = 1i64 << (bits - 1);
        (-half..half).contains(&k)
    } else {
        (0..1i64 << bits).contains(&k)
    }
}

/// `x op k` evaluated in the integer type: truncate, operate, wrap.
#[cfg(target_arch = "aarch64")]
fn reference(signed: bool, bits: u16, op: BinOp, k: i64, x: u64) -> u64 {
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let extend = |v: u64| -> i64 {
        let shift = 64 - u32::from(bits);
        ((v << shift) as i64) >> shift
    };
    let (t, kk) = (x & mask, k as u64 & mask);
    let wrapped = match op {
        BinOp::Shl => t << k,
        BinOp::Shr if signed => (extend(t) >> k) as u64,
        BinOp::Shr => t >> k,
        BinOp::BitAnd => t & kk,
        BinOp::BitOr => t | kk,
        BinOp::BitXor => t ^ kk,
        BinOp::Add => t.wrapping_add(kk),
        BinOp::Sub => t.wrapping_sub(kk),
        _ => unreachable!(),
    } & mask;
    if signed {
        extend(wrapped) as u64
    } else {
        wrapped
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn immediate_operations_agree_with_register_operations() {
    let inputs = [
        0u64,
        1,
        0x7f,
        0x80,
        0xff,
        0x100,
        0x7fff,
        0x8000,
        0xffff,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x1_0000_0000,
        0x1234_5678_9abc_def0,
        u64::MAX,
    ];
    let shifts = [0i64, 1, 3, 7, 8, 12, 15, 16, 31, 32, 63];
    let masks = [
        1i64,
        0x7f,
        0xff,
        0xf0f0,
        0xfff0,
        0x5555_5555,
        0x8000_0000,
        0xffff_ffff,
        0xffff_ffff_ffff_f007u64 as i64,
        0x00ff_00ff_00ff_00ff,
        -2,
    ];
    for (signed, bits) in [
        (false, 1u16),
        (false, 8),
        (false, 16),
        (false, 32),
        (true, 8),
        (true, 32),
        (false, 64),
        (true, 64),
    ] {
        let cases = [
            (BinOp::Shl, &shifts[..]),
            (BinOp::Shr, &shifts[..]),
            (BinOp::BitAnd, &masks[..]),
            (BinOp::BitOr, &masks[..]),
            (BinOp::BitXor, &masks[..]),
            (BinOp::Add, &masks[..]),
            (BinOp::Sub, &masks[..]),
        ];
        for (op, constants) in cases {
            for &k in constants
                .iter()
                .filter(|k| representable(signed, bits, **k))
            {
                let with_imm = narrow_op(signed, bits, op, k, true);
                let with_reg = narrow_op(signed, bits, op, k, false);
                for &x in &inputs {
                    // SAFETY: both functions were generated for this signature.
                    let (a, b) = unsafe { ((with_imm.call)(x), (with_reg.call)(x)) };
                    let name = format!(
                        "{op:?} {k:#x} on {}{bits} with input {x:#x}",
                        if signed { "i" } else { "u" }
                    );
                    assert_eq!(a, b, "{name}");
                    if matches!(op, BinOp::Shl | BinOp::Shr) && k >= i64::from(bits) {
                        continue;
                    }
                    let expected = reference(signed, bits, op, k, x);
                    // An unsigned result must be canonical (upper bits clear);
                    // a signed one is compared on the bits of its type.
                    let mask = if bits == 64 {
                        u64::MAX
                    } else {
                        (1u64 << bits) - 1
                    };
                    if signed {
                        assert_eq!(a & mask, expected & mask, "{name}");
                    } else {
                        assert_eq!(a, expected, "{name}");
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn narrow_results_stay_canonical_for_later_uses() {
    let run = |signed, bits, op, k, x| {
        let f = narrow_op(signed, bits, op, k, true);
        // SAFETY: generated for this signature.
        unsafe { (f.call)(x) }
    };
    // u8: a shift that overflows the type wraps to 8 bits.
    assert_eq!(run(false, 8, BinOp::Shl, 4, 0xff), 0xf0);
    assert_eq!(run(false, 8, BinOp::Shl, 4, 0x10f), 0xf0);
    // u8 & and >> keep the value inside 8 bits.
    assert_eq!(run(false, 8, BinOp::BitAnd, 0x70, 0xabcd), 0xcd & 0x70);
    assert_eq!(run(false, 8, BinOp::Shr, 4, 0x1ab), 0xa);
    // i8 keeps its sign through an arithmetic shift.
    assert_eq!(run(true, 8, BinOp::Shr, 2, 0xf0) as u8, 0xfc);
}
