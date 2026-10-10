// Rust selector boundary tests for the Vulcan-derived AArch64 port (Apache-2.0).
use super::*;
use volt_ir::attribute::Endianness;
use volt_ir::function::{
    Arith, AtomicOrdering, BinOp, Block, CmpOp, Compare, Convert, ConvertKind, Dot, EdgeDesc,
    Extract, Function, Load, MemFlags, Opcode, Ret, Terminator, Value,
};
use volt_ir::types::{FloatKind, IntDesc, Type, TypeKind, VectorDesc};

fn integer(f: &mut Function, bits: u16) -> Type {
    f.types.intern(TypeKind::Int(IntDesc { bits }))
}

fn convert(f: &mut Function, b: Block, ty: Type, value: Value, kind: ConvertKind) -> Value {
    f.append_inst(b, ty, Opcode::Convert(Convert { value, kind }))
}

/// `x` (an integer of `from` bits) widened to the 64-bit type `wide`, by sign when `signed`;
/// a 64-bit value is already that wide.
fn widen(f: &mut Function, b: Block, wide: Type, x: Value, from: u16, signed: bool) -> Value {
    if from == 64 {
        return x;
    }
    let kind = if signed {
        ConvertKind::Sext
    } else {
        ConvertKind::Zext
    };
    convert(f, b, wide, x, kind)
}

/// The low `bits` of the 64-bit `x`.
fn narrow(f: &mut Function, b: Block, ty: Type, x: Value, bits: u16) -> Value {
    if bits == 64 {
        return x;
    }
    convert(f, b, ty, x, ConvertKind::Trunc)
}

#[test]
fn emits_native_divide_remainder_and_high_product() {
    for (op, opcode) in [
        (BinOp::UDiv, 0x9ac00800),
        (BinOp::SDiv, 0x9ac00c00),
        (BinOp::URem, 0x9ac00800),
        (BinOp::SRem, 0x9ac00c00),
        (BinOp::UMulh, 0x9bc07c00),
        (BinOp::SMulh, 0x9b407c00),
    ] {
        let mut f = Function::new();
        let ty = integer(&mut f, 64);
        let b = f.append_block();
        let a = f.append_block_param(b, ty);
        let c = f.append_block_param(b, ty);
        let result = f.append_inst(b, ty, Opcode::Arith(Arith { op, lhs: a, rhs: c }));
        f.set_terminator(b, Terminator::Ret(Ret::one(result)));
        let code = compile(&f).unwrap();
        assert_eq!(code.last(), Some(&crate::aarch64::encode::ret()), "{op:?}");
        assert!(code.iter().any(|w| w & 0xffe0fc00 == opcode), "{op:?}");
        // A remainder is the divide followed by a multiply-subtract.
        let msub = code.iter().any(|w| w & 0xffe08000 == 0x9b008000);
        assert_eq!(msub, matches!(op, BinOp::URem | BinOp::SRem), "{op:?}");
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
    let ty = integer(&mut f, 64);
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
    for (op, signed) in [
        (BinOp::UDiv, false),
        (BinOp::SDiv, true),
        (BinOp::URem, false),
        (BinOp::SRem, true),
    ] {
        let mut f = Function::new();
        let ty = integer(&mut f, 64);
        let b = f.append_block();
        let a = f.append_block_param(b, ty);
        let c = f.append_block_param(b, ty);
        let result = f.append_inst(b, ty, Opcode::Arith(Arith { op, lhs: a, rhs: c }));
        f.set_terminator(b, Terminator::Ret(Ret::one(result)));
        let image = mapped(&f);
        // SAFETY: function is generated for precisely this AAPCS64 signature.
        let entry: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { image.entry() };
        let is_div = matches!(op, BinOp::UDiv | BinOp::SDiv);
        let zero = unsafe { entry(123, 0) };
        assert_eq!(zero, if is_div { 0 } else { 123 });
        if signed {
            let overflow = unsafe { entry(i64::MIN as u64, u64::MAX) };
            assert_eq!(overflow, if is_div { i64::MIN as u64 } else { 0 });
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn loop_edges_execute_parallel_parameter_permutations() {
    let mut f = Function::new();
    let ty = integer(&mut f, 64);
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
    let byte = integer(&mut f, 8);
    let wide = integer(&mut f, 64);
    let b = f.append_block();
    let p = f.append_block_param(b, ptr);
    let value = f.append_inst(
        b,
        byte,
        Opcode::Load(Load {
            ptr: p,
            mem: MemFlags {
                volatile: true,
                ..MemFlags::new()
            },
        }),
    );
    let value = convert(&mut f, b, wide, value, ConvertKind::Sext);
    f.set_terminator(b, Terminator::Ret(Ret::one(value)));
    let image = mapped(&f);
    let entry: unsafe extern "C" fn(*const i8) -> i64 = unsafe { image.entry() };
    let byte = -97i8;
    assert_eq!(unsafe { entry(&byte) }, -97);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn big_endian_loads_are_byte_swapped() {
    let bytes: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
    for (bits, expected) in [
        (16u16, u16::from_be_bytes([bytes[0], bytes[1]]) as u64),
        (
            32,
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
        ),
        (64, u64::from_be_bytes(bytes)),
    ] {
        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let ty = integer(&mut f, bits);
        let wide = integer(&mut f, 64);
        let b = f.append_block();
        let p = f.append_block_param(b, ptr);
        let mem = MemFlags {
            endian: Endianness::Big,
            ..MemFlags::new()
        };
        let value = f.append_load(b, ty, p, mem);
        let value = widen(&mut f, b, wide, value, bits, false);
        f.set_terminator(b, Terminator::Ret(Ret::one(value)));
        let image = mapped(&f);
        let entry: unsafe extern "C" fn(*const u8) -> u64 = unsafe { image.entry() };
        assert_eq!(unsafe { entry(bytes.as_ptr()) }, expected, "{bits}-bit");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn big_endian_stores_are_byte_swapped() {
    for bits in [16u16, 32, 64] {
        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let wide = integer(&mut f, 64);
        let ty = integer(&mut f, bits);
        let b = f.append_block();
        let p = f.append_block_param(b, ptr);
        let x = f.append_block_param(b, wide);
        let narrow = narrow(&mut f, b, ty, x, bits);
        let mem = MemFlags {
            endian: Endianness::Big,
            ..MemFlags::new()
        };
        f.append_store_mem(b, narrow, p, mem);
        f.set_terminator(b, Terminator::Ret(Ret::none()));
        let image = mapped(&f);
        let entry: unsafe extern "C" fn(*mut u8, u64) = unsafe { image.entry() };
        let mut out = [0u8; 8];
        let x = 0x0123_4567_89ab_cdefu64;
        unsafe { entry(out.as_mut_ptr(), x) };
        let n = bits as usize / 8;
        assert_eq!(out[..n], x.to_be_bytes()[8 - n..], "{bits}-bit");
        assert!(out[n..].iter().all(|b| *b == 0), "{bits}-bit overwrote");
    }
}

/// `(p: ptr) -> 64-bit` loading a `bits`-wide integer `offset` bytes past `p` with `mem` and
/// widening it (sign- or zero-extending to match `signed`).
fn ordered_load_fn(bits: u16, signed: bool, mem: MemFlags, offset: i64) -> Function {
    let mut f = Function::new();
    let ptr = f.types.ptr_global();
    let ty = integer(&mut f, bits);
    let wide = integer(&mut f, 64);
    let b = f.append_block();
    let p = f.append_block_param(b, ptr);
    let addr = if offset == 0 {
        p
    } else {
        f.append_arith_imm(b, ptr, BinOp::Add, p, offset)
    };
    let value = f.append_load(b, ty, addr, mem);
    let value = widen(&mut f, b, wide, value, bits, signed);
    f.set_terminator(b, Terminator::Ret(Ret::one(value)));
    f
}

/// `(p: ptr, x: u64)` storing the low `bits` of `x` `offset` bytes past `p` with `mem`.
fn ordered_store_fn(bits: u16, mem: MemFlags, offset: i64) -> Function {
    let mut f = Function::new();
    let ptr = f.types.ptr_global();
    let wide = integer(&mut f, 64);
    let ty = integer(&mut f, bits);
    let b = f.append_block();
    let p = f.append_block_param(b, ptr);
    let x = f.append_block_param(b, wide);
    let narrow = narrow(&mut f, b, ty, x, bits);
    let addr = if offset == 0 {
        p
    } else {
        f.append_arith_imm(b, ptr, BinOp::Add, p, offset)
    };
    f.append_store_mem(b, narrow, addr, mem);
    f.set_terminator(b, Terminator::Ret(Ret::none()));
    f
}

fn with_ordering(ordering: Option<AtomicOrdering>, bits: u16) -> MemFlags {
    MemFlags {
        ordering,
        align: u32::from(bits / 8),
        ..MemFlags::new()
    }
}

const LOAD_ORDERINGS: [Option<AtomicOrdering>; 4] = [
    None,
    Some(AtomicOrdering::Relaxed),
    Some(AtomicOrdering::Acquire),
    Some(AtomicOrdering::SeqCst),
];
const STORE_ORDERINGS: [Option<AtomicOrdering>; 4] = [
    None,
    Some(AtomicOrdering::Relaxed),
    Some(AtomicOrdering::Release),
    Some(AtomicOrdering::SeqCst),
];

/// Disassembly of every word `compile` emits for `f`.
fn listing(f: &Function) -> Vec<String> {
    compile(f)
        .unwrap()
        .iter()
        .map(|&w| crate::aarch64::listing::one(w))
        .collect()
}

fn mnemonic(line: &str) -> &str {
    line.split(' ').next().unwrap()
}

/// Acquire and sequentially consistent loads select `ldar{b,h,}`, release and sequentially
/// consistent stores `stlr{b,h,}`, each addressed through a bare `[xn]` even when an offset
/// could have been folded; relaxed and plain accesses stay ordinary `ldr`/`str`.
#[test]
fn ordered_accesses_select_load_acquire_and_store_release() {
    for (bits, load_acq, store_rel, load_plain, store_plain) in [
        (8u16, "ldarb", "stlrb", "ldrb", "strb"),
        (16, "ldarh", "stlrh", "ldrh", "strh"),
        (32, "ldar", "stlr", "ldr", "str"),
        (64, "ldar", "stlr", "ldr", "str"),
    ] {
        for offset in [0i64, 8] {
            for ordering in LOAD_ORDERINGS {
                let lines = listing(&ordered_load_fn(
                    bits,
                    false,
                    with_ordering(ordering, bits),
                    offset,
                ));
                let ordered = matches!(
                    ordering,
                    Some(AtomicOrdering::Acquire | AtomicOrdering::SeqCst)
                );
                let want = if ordered { load_acq } else { load_plain };
                let access: Vec<&String> = lines
                    .iter()
                    .filter(|l| mnemonic(l).starts_with("ldr") || mnemonic(l).starts_with("ldar"))
                    .collect();
                assert_eq!(access.len(), 1, "{bits} {ordering:?} {lines:?}");
                assert_eq!(mnemonic(access[0]), want, "{bits} {ordering:?} {lines:?}");
                if ordered {
                    assert!(
                        access[0].ends_with(']') && !access[0].contains(", #"),
                        "{lines:?}"
                    );
                    assert!(!lines.iter().any(|l| mnemonic(l) == load_plain));
                }
            }
            for ordering in STORE_ORDERINGS {
                let lines = listing(&ordered_store_fn(
                    bits,
                    with_ordering(ordering, bits),
                    offset,
                ));
                let ordered = matches!(
                    ordering,
                    Some(AtomicOrdering::Release | AtomicOrdering::SeqCst)
                );
                let want = if ordered { store_rel } else { store_plain };
                let access: Vec<&String> = lines
                    .iter()
                    .filter(|l| mnemonic(l).starts_with("str") || mnemonic(l).starts_with("stlr"))
                    .collect();
                assert_eq!(access.len(), 1, "{bits} {ordering:?} {lines:?}");
                assert_eq!(mnemonic(access[0]), want, "{bits} {ordering:?} {lines:?}");
                if ordered {
                    assert!(
                        access[0].ends_with(']') && !access[0].contains(", #"),
                        "{lines:?}"
                    );
                }
            }
        }
    }
}

/// An ordering that has no instruction for the access kind, or a type with no ordered form,
/// must fail closed rather than emit a weaker access.
#[test]
fn unorderable_accesses_fail_closed() {
    for ordering in [AtomicOrdering::Release, AtomicOrdering::AcqRel] {
        let f = ordered_load_fn(32, false, with_ordering(Some(ordering), 32), 0);
        assert!(
            matches!(compile(&f), Err(Error::Unsupported)),
            "{ordering:?}"
        );
    }
    for ordering in [AtomicOrdering::Acquire, AtomicOrdering::AcqRel] {
        let f = ordered_store_fn(32, with_ordering(Some(ordering), 32), 0);
        assert!(
            matches!(compile(&f), Err(Error::Unsupported)),
            "{ordering:?}"
        );
    }
    // A 24-bit integer has no ordered form, even when relaxed.
    let f = ordered_load_fn(
        24,
        false,
        with_ordering(Some(AtomicOrdering::Relaxed), 0),
        0,
    );
    assert!(matches!(compile(&f), Err(Error::Unsupported)));
}

#[repr(align(16))]
struct Image([u8; 16]);

const IMAGE: Image = Image([
    0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70,
]);

/// Every width, signedness and ordering loads the same value an ordinary load of that type
/// would (zero- or sign-extended), at offset 0 and at a folded-looking offset 8.
#[cfg(target_arch = "aarch64")]
#[test]
fn ordered_loads_produce_the_plain_load_value() {
    let image = IMAGE;
    for bits in [8u16, 16, 32, 64] {
        for signed in [false, true] {
            for offset in [0usize, 8] {
                let n = usize::from(bits / 8);
                let mut raw = [0u8; 8];
                raw[..n].copy_from_slice(&image.0[offset..offset + n]);
                let zero = u64::from_le_bytes(raw);
                let expected = if signed && bits < 64 {
                    (((zero << (64 - bits)) as i64) >> (64 - bits)) as u64
                } else {
                    zero
                };
                for ordering in LOAD_ORDERINGS {
                    let f =
                        ordered_load_fn(bits, signed, with_ordering(ordering, bits), offset as i64);
                    let jitted = mapped(&f);
                    let entry: unsafe extern "C" fn(*const u8) -> u64 = unsafe { jitted.entry() };
                    assert_eq!(
                        unsafe { entry(image.0.as_ptr()) },
                        expected,
                        "{bits}-bit signed={signed} offset={offset} {ordering:?}"
                    );
                }
            }
        }
    }
}

/// Every width and ordering stores exactly its own bytes.
#[cfg(target_arch = "aarch64")]
#[test]
fn ordered_stores_write_exactly_their_width() {
    let x = 0x0123_4567_89ab_cdefu64;
    for bits in [8u16, 16, 32, 64] {
        for offset in [0usize, 8] {
            let n = usize::from(bits / 8);
            for ordering in STORE_ORDERINGS {
                let f = ordered_store_fn(bits, with_ordering(ordering, bits), offset as i64);
                let jitted = mapped(&f);
                let entry: unsafe extern "C" fn(*mut u8, u64) = unsafe { jitted.entry() };
                let mut out = Image([0xee; 16]);
                unsafe { entry(out.0.as_mut_ptr(), x) };
                let mut want = [0xeeu8; 16];
                want[offset..offset + n].copy_from_slice(&x.to_le_bytes()[..n]);
                assert_eq!(out.0, want, "{bits}-bit offset={offset} {ordering:?}");
            }
        }
    }
}

/// Float accesses go through an integer register (legalize reinterprets them) and pointers
/// are 64-bit integers.
#[cfg(target_arch = "aarch64")]
#[test]
fn ordered_float_and_pointer_accesses_execute() {
    let mem = |ordering, bytes: u32| MemFlags {
        ordering: Some(ordering),
        align: bytes,
        ..MemFlags::new()
    };
    for ordering in [AtomicOrdering::Acquire, AtomicOrdering::SeqCst] {
        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let ty = f.types.intern(TypeKind::Float(FloatKind::F32));
        let b = f.append_block();
        let p = f.append_block_param(b, ptr);
        let v = f.append_load(b, ty, p, mem(ordering, 4));
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        assert!(listing(&f).iter().any(|l| mnemonic(l) == "ldar"));
        let jitted = mapped(&f);
        let entry: unsafe extern "C" fn(*const f32) -> f32 = unsafe { jitted.entry() };
        let x = -1.5f32;
        assert_eq!(unsafe { entry(&x) }, -1.5);

        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let ty = f.types.intern(TypeKind::Float(FloatKind::F64));
        let b = f.append_block();
        let p = f.append_block_param(b, ptr);
        let v = f.append_load(b, ty, p, mem(ordering, 8));
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        let jitted = mapped(&f);
        let entry: unsafe extern "C" fn(*const f64) -> f64 = unsafe { jitted.entry() };
        let x = 6.25f64;
        assert_eq!(unsafe { entry(&x) }, 6.25);
    }
    for ordering in [AtomicOrdering::Release, AtomicOrdering::SeqCst] {
        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let ty = f.types.intern(TypeKind::Float(FloatKind::F64));
        let b = f.append_block();
        let p = f.append_block_param(b, ptr);
        let x = f.append_block_param(b, ty);
        f.append_store_mem(b, x, p, mem(ordering, 8));
        f.set_terminator(b, Terminator::Ret(Ret::none()));
        assert!(listing(&f).iter().any(|l| mnemonic(l) == "stlr"));
        let jitted = mapped(&f);
        let entry: unsafe extern "C" fn(*mut f64, f64) = unsafe { jitted.entry() };
        let mut out = 0.0f64;
        unsafe { entry(&mut out, -2.75) };
        assert_eq!(out, -2.75);
    }
    // Pointer-typed: load a pointer acquire, store it back release elsewhere.
    let mut f = Function::new();
    let ptr = f.types.ptr_global();
    let b = f.append_block();
    let src = f.append_block_param(b, ptr);
    let dst = f.append_block_param(b, ptr);
    let v = f.append_load(b, ptr, src, mem(AtomicOrdering::Acquire, 8));
    f.append_store_mem(b, v, dst, mem(AtomicOrdering::Release, 8));
    f.set_terminator(b, Terminator::Ret(Ret::none()));
    let lines = listing(&f);
    assert!(lines.iter().any(|l| mnemonic(l) == "ldar"));
    assert!(lines.iter().any(|l| mnemonic(l) == "stlr"));
    let jitted = mapped(&f);
    let entry: unsafe extern "C" fn(*const *const u8, *mut *const u8) = unsafe { jitted.entry() };
    let target = 0x1234usize as *const u8;
    let mut out = core::ptr::null();
    unsafe { entry(&target, &mut out) };
    assert_eq!(out, target);
}

/// Message passing: the writer stores data, then a release flag; the reader acquire-loads the
/// flag and then reads the data in the same compiled function. A reader that observes round
/// `i` must observe round `i`'s data. An ack handshake keeps the writer from running ahead, so
/// the expected data is exact on every round.
#[cfg(target_arch = "aarch64")]
#[test]
fn release_acquire_message_passing_between_threads() {
    use core::sync::atomic::{AtomicU64, Ordering};

    // writer(data, flag, value, round): *data = value; release-store *flag = round.
    let writer = {
        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let u64t = integer(&mut f, 64);
        let b = f.append_block();
        let data = f.append_block_param(b, ptr);
        let flag = f.append_block_param(b, ptr);
        let value = f.append_block_param(b, u64t);
        let round = f.append_block_param(b, u64t);
        f.append_store_mem(b, value, data, with_ordering(None, 64));
        f.append_store_mem(
            b,
            round,
            flag,
            with_ordering(Some(AtomicOrdering::Release), 64),
        );
        f.set_terminator(b, Terminator::Ret(Ret::none()));
        f
    };
    // reader(flag, data, out) -> flag: acquire-load the flag, then load the data into *out.
    let reader = {
        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let u64t = integer(&mut f, 64);
        let b = f.append_block();
        let flag = f.append_block_param(b, ptr);
        let data = f.append_block_param(b, ptr);
        let out = f.append_block_param(b, ptr);
        let seen = f.append_load(
            b,
            u64t,
            flag,
            with_ordering(Some(AtomicOrdering::Acquire), 64),
        );
        let value = f.append_load(b, u64t, data, with_ordering(None, 64));
        f.append_store_mem(b, value, out, with_ordering(None, 64));
        f.set_terminator(b, Terminator::Ret(Ret::one(seen)));
        f
    };
    let writer_image = mapped(&writer);
    let reader_image = mapped(&reader);
    let write: unsafe extern "C" fn(*mut u64, *mut u64, u64, u64) = unsafe { writer_image.entry() };
    let read: unsafe extern "C" fn(*const u64, *const u64, *mut u64) -> u64 =
        unsafe { reader_image.entry() };

    const ROUNDS: u64 = 1000;
    let expected = |round: u64| round.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut data = 0u64;
    let mut flag = 0u64;
    // The raw addresses cross the thread boundary as integers; each is touched only by the
    // compiled code (and the final reads happen after the scope joins).
    let (data_addr, flag_addr) = (&raw mut data as usize, &raw mut flag as usize);
    let ack = AtomicU64::new(0);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for round in 1..=ROUNDS {
                unsafe {
                    write(
                        data_addr as *mut u64,
                        flag_addr as *mut u64,
                        expected(round),
                        round,
                    )
                };
                while ack.load(Ordering::SeqCst) != round {
                    std::thread::yield_now();
                }
            }
        });
        let mut out = 0u64;
        for round in 1..=ROUNDS {
            loop {
                let seen =
                    unsafe { read(flag_addr as *const u64, data_addr as *const u64, &mut out) };
                if seen == round {
                    break;
                }
                assert!(seen < round, "flag ran ahead of the ack handshake");
                std::thread::yield_now();
            }
            assert_eq!(out, expected(round), "round {round}");
            ack.store(round, Ordering::SeqCst);
        }
    });
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
    // The right shift of a signed value is arithmetic.
    let op = if signed && op == BinOp::Shr {
        BinOp::Sar
    } else {
        op
    };
    let mut f = Function::new();
    let wide = integer(&mut f, 64);
    let ty = integer(&mut f, bits);
    let b = f.append_block();
    let x = f.append_block_param(b, wide);
    let narrow = narrow(&mut f, b, ty, x, bits);
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
    let back = widen(&mut f, b, wide, result, bits, signed);
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
    // i8: a shift that overflows the type wraps to 8 bits.
    assert_eq!(run(false, 8, BinOp::Shl, 4, 0xff), 0xf0);
    assert_eq!(run(false, 8, BinOp::Shl, 4, 0x10f), 0xf0);
    // i8 & and >> keep the value inside 8 bits.
    assert_eq!(run(false, 8, BinOp::BitAnd, 0x70, 0xabcd), 0xcd & 0x70);
    assert_eq!(run(false, 8, BinOp::Shr, 4, 0x1ab), 0xa);
    // A signed i8 keeps its sign through an arithmetic shift.
    assert_eq!(run(true, 8, BinOp::Shr, 2, 0xf0) as u8, 0xfc);
}

/// Signed and unsigned behavior is chosen by the operation: each pair of operations below sees
/// the same bits and must differ exactly where the sign matters.
#[cfg(target_arch = "aarch64")]
mod signedness {
    use super::*;

    const WIDTHS: [u16; 4] = [8, 16, 32, 64];

    fn mask(bits: u16) -> u64 {
        if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    }

    /// `v`'s low `bits` as a signed number.
    fn sx(bits: u16, v: u64) -> i64 {
        let shift = 64 - u32::from(bits);
        ((v << shift) as i64) >> shift
    }

    /// Values of a `bits`-wide integer (zero-extended) around every sign boundary.
    fn samples(bits: u16) -> Vec<u64> {
        let m = mask(bits);
        let max = m >> 1;
        let mut v = vec![
            0,
            1,
            2,
            3,
            7,
            100,
            0x55 & m,
            0xa5 & m,
            127 & m,
            128 & m,
            200 & m,
            max,
            max + 1,
            m - 1,
            m,
            0x0123_4567_89ab_cdef & m,
            0xfedc_ba98_7654_3210 & m,
        ];
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The result of `op` on the `bits`-wide operands `a` and `b` (zero-extended), zero-extended.
    fn model(op: BinOp, bits: u16, a: u64, b: u64) -> u64 {
        let m = mask(bits);
        let (sa, sb) = (i128::from(sx(bits, a)), i128::from(sx(bits, b)));
        let r = match op {
            BinOp::UDiv => {
                if b == 0 {
                    0
                } else {
                    a / b
                }
            }
            BinOp::SDiv => {
                if b == 0 {
                    0
                } else {
                    (sa / sb) as u64
                }
            }
            BinOp::URem => {
                if b == 0 {
                    a
                } else {
                    a % b
                }
            }
            BinOp::SRem => {
                if b == 0 {
                    a
                } else {
                    (sa % sb) as u64
                }
            }
            BinOp::Shr => a >> b,
            BinOp::Sar => (sx(bits, a) >> b) as u64,
            BinOp::UMulh => ((u128::from(a) * u128::from(b)) >> bits) as u64,
            BinOp::SMulh => ((sa * sb) >> bits) as u64,
            _ => unreachable!("{op:?}"),
        };
        r & m
    }

    /// `(x: u64, y: u64) -> u64`: `op` on the low `bits` of both, zero-extended back.
    fn binary_fn(bits: u16, op: BinOp) -> Function {
        let mut f = Function::new();
        let wide = integer(&mut f, 64);
        let ty = integer(&mut f, bits);
        let b = f.append_block();
        let x = f.append_block_param(b, wide);
        let y = f.append_block_param(b, wide);
        let nx = narrow(&mut f, b, ty, x, bits);
        let ny = narrow(&mut f, b, ty, y, bits);
        let r = f.append_inst(
            b,
            ty,
            Opcode::Arith(Arith {
                op,
                lhs: nx,
                rhs: ny,
            }),
        );
        let r = widen(&mut f, b, wide, r, bits, false);
        f.set_terminator(b, Terminator::Ret(Ret::one(r)));
        f
    }

    /// `(x: u64) -> u64`: `x op k` with `k` an `ArithImm` constant.
    fn immediate_fn(bits: u16, op: BinOp, k: i64) -> Function {
        let mut f = Function::new();
        let wide = integer(&mut f, 64);
        let ty = integer(&mut f, bits);
        let b = f.append_block();
        let x = f.append_block_param(b, wide);
        let nx = narrow(&mut f, b, ty, x, bits);
        let r = f.append_arith_imm(b, ty, op, nx, k);
        let r = widen(&mut f, b, wide, r, bits, false);
        f.set_terminator(b, Terminator::Ret(Ret::one(r)));
        f
    }

    fn call2(image: &crate::native::JittedFunction, x: u64, y: u64) -> u64 {
        // SAFETY: every caller maps a function built for this signature.
        let entry: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { image.entry() };
        unsafe { entry(x, y) }
    }

    fn call1(image: &crate::native::JittedFunction, x: u64) -> u64 {
        // SAFETY: every caller maps a function built for this signature.
        let entry: unsafe extern "C" fn(u64) -> u64 = unsafe { image.entry() };
        unsafe { entry(x) }
    }

    /// Runs `op` on every pair of samples, with the right operand in a register and (when
    /// `immediates`) as a sign-extended `ArithImm`, against `model`.
    fn check_binary(ops: &[BinOp], amounts: Option<fn(u16) -> Vec<u64>>, immediates: bool) {
        for &bits in &WIDTHS {
            for &op in ops {
                let rights = amounts.map_or_else(|| samples(bits), |f| f(bits));
                let image = mapped(&binary_fn(bits, op));
                for &b in &rights {
                    let imm = immediates.then(|| mapped(&immediate_fn(bits, op, sx(bits, b))));
                    for &a in &samples(bits) {
                        let want = model(op, bits, a, b);
                        // The operands carry junk above their width: only the type counts.
                        let junk = !mask(bits);
                        let got = call2(&image, a | junk, b | junk);
                        assert_eq!(got, want, "{op:?} i{bits} {a:#x}, {b:#x}");
                        if let Some(imm) = &imm {
                            let got = call1(imm, a | junk);
                            assert_eq!(got, want, "{op:?} i{bits} {a:#x}, #{b:#x}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn division_and_remainder_differ_by_operation() {
        check_binary(
            &[BinOp::UDiv, BinOp::SDiv, BinOp::URem, BinOp::SRem],
            None,
            true,
        );
        // -7 / 2 and -7 % 2 for i8: signed and unsigned disagree, and both are exact.
        let (a, b) = (0xf9u64, 2u64);
        assert_eq!(model(BinOp::SDiv, 8, a, b), 0xfd);
        assert_eq!(model(BinOp::UDiv, 8, a, b), 124);
        assert_eq!(model(BinOp::SRem, 8, a, b), 0xff);
        assert_eq!(model(BinOp::URem, 8, a, b), 1);
    }

    #[test]
    fn right_shifts_differ_by_operation() {
        fn amounts(bits: u16) -> Vec<u64> {
            [0, 1, 3, 7, u64::from(bits / 2), u64::from(bits) - 1]
                .into_iter()
                .filter(|s| *s < u64::from(bits))
                .collect()
        }
        check_binary(&[BinOp::Shr, BinOp::Sar], Some(amounts), true);
        let image = mapped(&binary_fn(8, BinOp::Sar));
        assert_eq!(call2(&image, 0x80, 7), 0xff);
        let image = mapped(&binary_fn(8, BinOp::Shr));
        assert_eq!(call2(&image, 0x80, 7), 1);
    }

    #[test]
    fn high_products_differ_by_operation() {
        check_binary(&[BinOp::UMulh, BinOp::SMulh], None, true);
        // i64: (-1) * (-1) has high half 0 signed, but 0xffff_ffff_ffff_fffe unsigned.
        let image = mapped(&binary_fn(64, BinOp::SMulh));
        assert_eq!(call2(&image, u64::MAX, u64::MAX), 0);
        let image = mapped(&binary_fn(64, BinOp::UMulh));
        assert_eq!(call2(&image, u64::MAX, u64::MAX), u64::MAX - 1);
    }

    /// `(x, y) -> (x cmp y) as u64` on the low `bits`; branching on the compare when `branch`,
    /// so the compare-and-branch fusion sees it.
    fn compare_fn(bits: u16, op: CmpOp, branch: bool) -> Function {
        let mut f = Function::new();
        let wide = integer(&mut f, 64);
        let ty = integer(&mut f, bits);
        let bool_ty = f.types.intern(TypeKind::Bool);
        let b = f.append_block();
        let x = f.append_block_param(b, wide);
        let y = f.append_block_param(b, wide);
        let nx = narrow(&mut f, b, ty, x, bits);
        let ny = narrow(&mut f, b, ty, y, bits);
        let cond = f.append_inst(
            b,
            bool_ty,
            Opcode::Icmp(Compare {
                op,
                lhs: nx,
                rhs: ny,
            }),
        );
        if branch {
            let yes = f.append_block();
            let no = f.append_block();
            f.append_if(b, cond, EdgeDesc::bare(yes), EdgeDesc::bare(no));
            let one = f.append_inst(yes, wide, Opcode::Iconst(1));
            f.set_terminator(yes, Terminator::Ret(Ret::one(one)));
            let zero = f.append_inst(no, wide, Opcode::Iconst(0));
            f.set_terminator(no, Terminator::Ret(Ret::one(zero)));
        } else {
            let r = convert(&mut f, b, wide, cond, ConvertKind::Zext);
            f.set_terminator(b, Terminator::Ret(Ret::one(r)));
        }
        f
    }

    fn compare_model(op: CmpOp, bits: u16, a: u64, b: u64) -> bool {
        let (sa, sb) = (sx(bits, a), sx(bits, b));
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
            _ => unreachable!("{op:?}"),
        }
    }

    #[test]
    fn integer_compares_differ_by_operation() {
        let ops = [
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
        ];
        for bits in WIDTHS {
            for op in ops {
                for branch in [false, true] {
                    let image = mapped(&compare_fn(bits, op, branch));
                    let junk = !mask(bits);
                    for &a in &samples(bits) {
                        for &b in &samples(bits) {
                            let got = call2(&image, a | junk, b | junk);
                            let want = u64::from(compare_model(op, bits, a, b));
                            assert_eq!(
                                got, want,
                                "{op:?} i{bits} {a:#x}, {b:#x} (branch: {branch})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pointer_compares_are_unsigned() {
        let mut f = Function::new();
        let ptr = f.types.ptr_global();
        let wide = integer(&mut f, 64);
        let bool_ty = f.types.intern(TypeKind::Bool);
        let b = f.append_block();
        let p = f.append_block_param(b, ptr);
        let q = f.append_block_param(b, ptr);
        let cond = f.append_inst(
            b,
            bool_ty,
            Opcode::Icmp(Compare {
                op: CmpOp::Ult,
                lhs: p,
                rhs: q,
            }),
        );
        let r = convert(&mut f, b, wide, cond, ConvertKind::Zext);
        f.set_terminator(b, Terminator::Ret(Ret::one(r)));
        let image = mapped(&f);
        assert_eq!(call2(&image, 1, u64::MAX), 1);
        assert_eq!(call2(&image, u64::MAX, 1), 0);
    }

    /// `(x: u64) -> u64`: `x` truncated to `from` bits, converted to `to` bits with `kind`, and
    /// zero-extended to 64 bits.
    fn int_convert_fn(kind: ConvertKind, from: u16, to: u16) -> Function {
        let mut f = Function::new();
        let wide = integer(&mut f, 64);
        let a = integer(&mut f, from);
        let c = integer(&mut f, to);
        let b = f.append_block();
        let x = f.append_block_param(b, wide);
        let x = narrow(&mut f, b, a, x, from);
        let y = convert(&mut f, b, c, x, kind);
        let y = widen(&mut f, b, wide, y, to, false);
        f.set_terminator(b, Terminator::Ret(Ret::one(y)));
        f
    }

    #[test]
    fn extensions_and_truncations_differ_by_kind() {
        for from in WIDTHS {
            for to in WIDTHS {
                if from < to {
                    for kind in [ConvertKind::Zext, ConvertKind::Sext] {
                        let image = mapped(&int_convert_fn(kind, from, to));
                        for &x in &samples(from) {
                            let want = if kind == ConvertKind::Sext {
                                sx(from, x) as u64 & mask(to)
                            } else {
                                x
                            };
                            let got = call1(&image, x | !mask(from));
                            assert_eq!(got, want, "{kind:?} i{from} -> i{to} {x:#x}");
                        }
                    }
                } else if from > to {
                    let image = mapped(&int_convert_fn(ConvertKind::Trunc, from, to));
                    for &x in &samples(from) {
                        let got = call1(&image, x);
                        assert_eq!(got, x & mask(to), "Trunc i{from} -> i{to} {x:#x}");
                    }
                }
            }
        }
    }

    const FLOAT_INPUTS: [f64; 23] = [
        0.0,
        1.5,
        -1.5,
        127.0,
        128.0,
        200.0,
        -129.0,
        255.9,
        256.0,
        40000.0,
        70000.0,
        -70000.0,
        2147483647.0,
        2147483648.0,
        4294967295.0,
        4294967296.0,
        -1.0e10,
        1.0e10,
        9.3e18,
        -9.3e18,
        1.0e30,
        -1.0e30,
        f64::NAN,
    ];

    fn float_type(f: &mut Function, double: bool) -> Type {
        f.types.intern(TypeKind::Float(if double {
            FloatKind::F64
        } else {
            FloatKind::F32
        }))
    }

    /// `(x: u64) -> float`: the low `bits` of `x`, converted with `kind`.
    fn int_to_float_fn(kind: ConvertKind, bits: u16, double: bool) -> Function {
        let mut f = Function::new();
        let wide = integer(&mut f, 64);
        let ty = integer(&mut f, bits);
        let fty = float_type(&mut f, double);
        let b = f.append_block();
        let x = f.append_block_param(b, wide);
        let x = narrow(&mut f, b, ty, x, bits);
        let y = convert(&mut f, b, fty, x, kind);
        f.set_terminator(b, Terminator::Ret(Ret::one(y)));
        f
    }

    /// `(x: float) -> u64`: `x` converted to `bits` with `kind`, zero-extended.
    fn float_to_int_fn(kind: ConvertKind, bits: u16, double: bool) -> Function {
        let mut f = Function::new();
        let wide = integer(&mut f, 64);
        let ty = integer(&mut f, bits);
        let fty = float_type(&mut f, double);
        let b = f.append_block();
        let x = f.append_block_param(b, fty);
        let y = convert(&mut f, b, ty, x, kind);
        let y = widen(&mut f, b, wide, y, bits, false);
        f.set_terminator(b, Terminator::Ret(Ret::one(y)));
        f
    }

    #[test]
    fn integer_to_float_differs_by_kind() {
        let inputs = {
            let mut v = Vec::new();
            for bits in WIDTHS {
                v.extend(samples(bits));
            }
            v
        };
        for bits in WIDTHS {
            for kind in [ConvertKind::SiToFp, ConvertKind::UiToFp] {
                let signed = kind == ConvertKind::SiToFp;
                let single = mapped(&int_to_float_fn(kind, bits, false));
                let double = mapped(&int_to_float_fn(kind, bits, true));
                // SAFETY: both were built for these signatures.
                let single: unsafe extern "C" fn(u64) -> f32 = unsafe { single.entry() };
                let double: unsafe extern "C" fn(u64) -> f64 = unsafe { double.entry() };
                for &x in &inputs {
                    let v = x & mask(bits);
                    let (want32, want64) = if signed {
                        (sx(bits, v) as f32, sx(bits, v) as f64)
                    } else {
                        (v as f32, v as f64)
                    };
                    // Garbage above the type must not matter.
                    let arg = x | !mask(bits);
                    let arg = if bits == 64 { x } else { arg };
                    let (got32, got64) = unsafe { (single(arg), double(arg)) };
                    assert_eq!(got32, want32, "{kind:?} i{bits} {x:#x} -> f32");
                    assert_eq!(got64, want64, "{kind:?} i{bits} {x:#x} -> f64");
                }
            }
        }
    }

    #[test]
    fn float_to_integer_differs_by_kind() {
        for bits in WIDTHS {
            for kind in [ConvertKind::FpToSi, ConvertKind::FpToUi] {
                let signed = kind == ConvertKind::FpToSi;
                let single = mapped(&float_to_int_fn(kind, bits, false));
                let double = mapped(&float_to_int_fn(kind, bits, true));
                // SAFETY: both were built for these signatures.
                let single: unsafe extern "C" fn(f32) -> u64 = unsafe { single.entry() };
                let double: unsafe extern "C" fn(f64) -> u64 = unsafe { double.entry() };
                let want = |narrow32: u64, wide64: u64| {
                    // A 64-bit destination saturates at 64 bits; every narrower one keeps the
                    // low bits of the 32-bit saturating conversion.
                    if bits == 64 {
                        wide64
                    } else {
                        narrow32 & mask(bits)
                    }
                };
                for &x in &FLOAT_INPUTS {
                    let (w32, w64) = if signed {
                        (
                            u64::from((x as f32 as i32) as u32),
                            (x as f32 as i64) as u64,
                        )
                    } else {
                        (u64::from(x as f32 as u32), x as f32 as u64)
                    };
                    assert_eq!(
                        unsafe { single(x as f32) },
                        want(w32, w64),
                        "{kind:?} f32 {x} -> i{bits}"
                    );
                    let (w32, w64) = if signed {
                        (u64::from((x as i32) as u32), (x as i64) as u64)
                    } else {
                        (u64::from(x as u32), x as u64)
                    };
                    assert_eq!(
                        unsafe { double(x) },
                        want(w32, w64),
                        "{kind:?} f64 {x} -> i{bits}"
                    );
                }
            }
        }
    }

    #[test]
    fn float_resize_extends_and_truncates() {
        let build = |from_double: bool| {
            let mut f = Function::new();
            let a = float_type(&mut f, from_double);
            let c = float_type(&mut f, !from_double);
            let b = f.append_block();
            let x = f.append_block_param(b, a);
            let y = convert(&mut f, b, c, x, ConvertKind::FpResize);
            f.set_terminator(b, Terminator::Ret(Ret::one(y)));
            mapped(&f)
        };
        let widen = build(false);
        let shrink = build(true);
        // SAFETY: built for these signatures.
        let widen: unsafe extern "C" fn(f32) -> f64 = unsafe { widen.entry() };
        let shrink: unsafe extern "C" fn(f64) -> f32 = unsafe { shrink.entry() };
        for x in [0.0f64, -1.5, 1.0e10, 3.4e38, 1.0e300, f64::INFINITY] {
            assert_eq!(unsafe { shrink(x) }, x as f32);
            assert_eq!(unsafe { widen(x as f32) }, f64::from(x as f32));
        }
    }

    /// Integer vector lanes are held zero-extended; widening them is up to the program.
    #[test]
    fn vector_lanes_extend_by_explicit_conversion() {
        for (kind, want) in [
            (ConvertKind::Sext, 0xffff_ffff_8000_0000u64),
            (ConvertKind::Zext, 0x8000_0000),
        ] {
            let mut f = Function::new();
            let i32t = integer(&mut f, 32);
            let wide = integer(&mut f, 64);
            let vt = f
                .types
                .intern(TypeKind::Vector(VectorDesc { len: 4, elem: i32t }));
            let b = f.append_block();
            let p: Vec<Value> = (0..4).map(|_| f.append_block_param(b, i32t)).collect();
            let v = f.append_struct_new(b, vt, &p);
            let lane = f.append_inst(
                b,
                i32t,
                Opcode::Extract(Extract {
                    aggregate: v,
                    index: 2,
                }),
            );
            let r = convert(&mut f, b, wide, lane, kind);
            f.set_terminator(b, Terminator::Ret(Ret::one(r)));
            let image = mapped(&f);
            // SAFETY: built for this signature.
            let entry: unsafe extern "C" fn(u32, u32, u32, u32) -> u64 = unsafe { image.entry() };
            assert_eq!(unsafe { entry(1, 2, 0x8000_0000, 4) }, want, "{kind:?}");
        }
    }

    #[test]
    fn dot_selects_signed_or_unsigned_by_flag() {
        for signed in [false, true] {
            let mut f = Function::new();
            let i8t = integer(&mut f, 8);
            let i32t = integer(&mut f, 32);
            let bytes = f
                .types
                .intern(TypeKind::Vector(VectorDesc { len: 16, elem: i8t }));
            let acc_ty = f
                .types
                .intern(TypeKind::Vector(VectorDesc { len: 4, elem: i32t }));
            let b = f.append_block();
            let acc = f.append_block_param(b, acc_ty);
            let x = f.append_block_param(b, bytes);
            let y = f.append_block_param(b, bytes);
            let r = f.append_inst(
                b,
                acc_ty,
                Opcode::Dot(Dot {
                    acc,
                    a: x,
                    b: y,
                    signed,
                }),
            );
            f.set_terminator(b, Terminator::Ret(Ret::one(r)));
            let code = compile(&f).unwrap();
            let v = crate::aarch64::encode::Reg::from_u5(0);
            let sdot = crate::aarch64::encode::sdot(v, v, v);
            let udot = crate::aarch64::encode::udot(v, v, v);
            let find = |word: u32| code.iter().any(|w| w & 0xffe0_fc00 == word & 0xffe0_fc00);
            assert_eq!(find(sdot), signed, "signed={signed}");
            assert_eq!(find(udot), !signed, "signed={signed}");
        }
    }
}
