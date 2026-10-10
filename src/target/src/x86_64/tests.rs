// Ported from Vulcan native/cases/link test boundaries (Apache-2.0).
// Copyright LilithSemi; Rust implementation modified for Volt.
use super::cases::{self, Val};
use super::*;
use volt_ir::function::*;
use volt_ir::types::Type;
struct Executable {
    ptr: *mut libc::c_void,
    len: usize,
}
impl Executable {
    fn new(f: &Function) -> Self {
        let code = compile(f).unwrap();
        let len = (code.len() + 4095) & !4095;
        // SAFETY: an anonymous private writable mapping owns exactly `len` bytes.
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        // SAFETY: code fits the mapping; permissions switch to RX before invocation.
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len());
            assert_eq!(
                libc::mprotect(ptr, len, libc::PROT_READ | libc::PROT_EXEC),
                0
            );
        }
        Self { ptr, len }
    }
}
impl Drop for Executable {
    fn drop(&mut self) {
        // SAFETY: this mapping is uniquely owned and no invocation outlives it.
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}
fn ty(f: &mut Function, s: &str) -> Type {
    f.types.parse_type(s).unwrap()
}
/// Calls `code` with `args`; the result's raw register contents (an `f32` in the low 32 bits).
fn call(code: &Executable, args: &[Val], want: Val) -> u64 {
    use Val::*;
    let p = code.ptr;
    // SAFETY: every transmute names the System V signature of the function that was compiled
    // for exactly this argument/result shape.
    unsafe {
        match (args, want) {
            ([Int(a)], Int(_)) => {
                let f: unsafe extern "C" fn(u64) -> u64 = core::mem::transmute(p);
                f(*a)
            }
            ([Int(a), Int(b)], Int(_)) => {
                let f: unsafe extern "C" fn(u64, u64) -> u64 = core::mem::transmute(p);
                f(*a, *b)
            }
            ([Int(a), Int(b), Int(c), Int(d)], Int(_)) => {
                let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 = core::mem::transmute(p);
                f(*a, *b, *c, *d)
            }
            ([F64(a)], Int(_)) => {
                let f: unsafe extern "C" fn(f64) -> u64 = core::mem::transmute(p);
                f(*a)
            }
            ([F32(a)], Int(_)) => {
                let f: unsafe extern "C" fn(f32) -> u64 = core::mem::transmute(p);
                f(*a)
            }
            ([Int(a)], F64(_)) => {
                let f: unsafe extern "C" fn(u64) -> f64 = core::mem::transmute(p);
                f(*a).to_bits()
            }
            ([Int(a)], F32(_)) => {
                let f: unsafe extern "C" fn(u64) -> f32 = core::mem::transmute(p);
                f(*a).to_bits() as u64
            }
            ([F32(a)], F64(_)) => {
                let f: unsafe extern "C" fn(f32) -> f64 = core::mem::transmute(p);
                f(*a).to_bits()
            }
            ([F64(a)], F32(_)) => {
                let f: unsafe extern "C" fn(f64) -> f32 = core::mem::transmute(p);
                f(*a).to_bits() as u64
            }
            _ => panic!("no calling convention for {args:?} -> {want:?}"),
        }
    }
}
/// Every operation whose result depends on signedness, at every width, against the reference
/// arithmetic of `cases`: arguments carry garbage above their declared width and results must
/// come back exactly zero-extended.
#[test]
fn signedness_cases_match_the_reference() {
    for case in cases::all() {
        let code = Executable::new(&case.function);
        for (args, want) in &case.runs {
            assert_eq!(
                call(&code, args, *want),
                want.bits(),
                "{} {args:x?}",
                case.name
            );
        }
    }
}
#[test]
fn scalar_division_shifts_and_high_product() {
    for (kind, op, a, b, want) in [
        ("i64", BinOp::UDiv, u64::MAX, 2, u64::MAX / 2),
        ("i64", BinOp::URem, 100, 9, 1),
        ("i64", BinOp::Shr, 1 << 63, 63, 1),
        ("i64", BinOp::Sar, 1 << 63, 63, u64::MAX),
        ("i64", BinOp::Shl, 7, 65, 14),
        ("i64", BinOp::UMulh, u64::MAX, u64::MAX, u64::MAX - 1),
        ("i64", BinOp::SDiv, (-100i64) as u64, 9, (-11i64) as u64),
        ("i64", BinOp::SRem, (-100i64) as u64, 9, (-1i64) as u64),
        (
            "i64",
            BinOp::UDiv,
            (-100i64) as u64,
            9,
            ((-100i64) as u64) / 9,
        ),
        ("i64", BinOp::SMulh, (-2i64) as u64, 3, u64::MAX),
    ] {
        let mem = Executable::new(&cases::arith(kind, op));
        // SAFETY: the compiled function has two 64-bit integer parameters and one result.
        let entry: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { core::mem::transmute(mem.ptr) };
        assert_eq!(unsafe { entry(a, b) }, want, "{kind} {op:?}");
    }
}
/// The one narrow signed divide that does not trap: the most negative value divided by -1
/// wraps, because the 32-bit hardware divide it runs on has room for the true quotient.
#[test]
fn narrow_signed_divide_of_minimum_by_minus_one_wraps() {
    for (kind, min) in [("i8", 0x80u64), ("i16", 0x8000)] {
        let neg_one = u64::MAX;
        for (op, want) in [(BinOp::SDiv, min), (BinOp::SRem, 0)] {
            let mem = Executable::new(&cases::arith(kind, op));
            let entry: unsafe extern "C" fn(u64, u64) -> u64 =
                unsafe { core::mem::transmute(mem.ptr) };
            assert_eq!(unsafe { entry(min, neg_one) }, want, "{kind} {op:?}");
        }
    }
}
#[test]
fn narrow_store_keeps_neighbor_bytes() {
    let mut f = Function::new();
    let p = ty(&mut f, "ptr");
    let byte = ty(&mut f, "i8");
    let word = ty(&mut f, "i32");
    let b = f.append_block();
    let ptr = f.append_block_param(b, p);
    let v = f.append_inst(b, byte, Opcode::Iconst(0xab));
    f.append_store(b, v, ptr);
    let r = f.append_inst(
        b,
        word,
        Opcode::Load(Load {
            ptr,
            mem: MemFlags::new(),
        }),
    );
    f.set_terminator(b, Terminator::Ret(Ret::one(r)));
    let mem = Executable::new(&f);
    let entry: unsafe extern "C" fn(*mut u8) -> u32 = unsafe { core::mem::transmute(mem.ptr) };
    let mut bytes = [0x11, 0x22, 0x33, 0x44, 0x55];
    assert_eq!(unsafe { entry(bytes.as_mut_ptr()) }, 0x443322ab);
    assert_eq!(bytes, [0xab, 0x22, 0x33, 0x44, 0x55]);
}
#[test]
fn big_endian_loads_are_byte_swapped() {
    let bytes: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
    for (kind, expected) in [
        ("i16", u16::from_be_bytes([bytes[0], bytes[1]]) as u64),
        (
            "i32",
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
        ),
        ("i64", u64::from_be_bytes(bytes)),
    ] {
        let mut f = Function::new();
        let p = ty(&mut f, "ptr");
        let t = ty(&mut f, kind);
        let wide = ty(&mut f, "i64");
        let b = f.append_block();
        let ptr = f.append_block_param(b, p);
        let mem = MemFlags {
            endian: volt_ir::attribute::Endianness::Big,
            ..MemFlags::new()
        };
        let mut v = f.append_load(b, t, ptr, mem);
        if t != wide {
            v = f.append_inst(
                b,
                wide,
                Opcode::Convert(Convert {
                    value: v,
                    kind: ConvertKind::Zext,
                }),
            );
        }
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        let code = Executable::new(&f);
        let entry: unsafe extern "C" fn(*const u8) -> u64 =
            unsafe { core::mem::transmute(code.ptr) };
        assert_eq!(unsafe { entry(bytes.as_ptr()) }, expected, "{kind}");
    }
}
/// `(p: ptr) -> i64` loading a `kind` integer `offset` bytes past `p` with `mem` and widening
/// it: sign-extending when `signed`, zero-extending otherwise (a load itself never extends
/// with the sign).
fn ordered_load(kind: &str, signed: bool, mem: MemFlags, offset: i64) -> Function {
    let mut f = Function::new();
    let p = ty(&mut f, "ptr");
    let t = ty(&mut f, kind);
    let w = ty(&mut f, "i64");
    let b = f.append_block();
    let ptr = f.append_block_param(b, p);
    let addr = if offset == 0 {
        ptr
    } else {
        f.append_arith_imm(b, p, BinOp::Add, ptr, offset)
    };
    let mut v = f.append_load(b, t, addr, mem);
    if t != w {
        v = f.append_inst(
            b,
            w,
            Opcode::Convert(Convert {
                value: v,
                kind: if signed {
                    ConvertKind::Sext
                } else {
                    ConvertKind::Zext
                },
            }),
        );
    }
    f.set_terminator(b, Terminator::Ret(Ret::one(v)));
    f
}
/// `(p: ptr, x: i64)` storing the low bits of `x` as a `kind` integer `offset` bytes past `p`.
fn ordered_store(kind: &str, mem: MemFlags, offset: i64) -> Function {
    let mut f = Function::new();
    let p = ty(&mut f, "ptr");
    let t = ty(&mut f, kind);
    let w = ty(&mut f, "i64");
    let b = f.append_block();
    let ptr = f.append_block_param(b, p);
    let x = f.append_block_param(b, w);
    let narrow = if t == w {
        x
    } else {
        f.append_inst(
            b,
            t,
            Opcode::Convert(Convert {
                value: x,
                kind: ConvertKind::Trunc,
            }),
        )
    };
    let addr = if offset == 0 {
        ptr
    } else {
        f.append_arith_imm(b, p, BinOp::Add, ptr, offset)
    };
    f.append_store_mem(b, narrow, addr, mem);
    f.set_terminator(b, Terminator::Ret(Ret::none()));
    f
}
fn ordering_mem(ordering: Option<AtomicOrdering>, bytes: u32) -> MemFlags {
    MemFlags {
        ordering,
        align: bytes,
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
#[repr(align(16))]
struct Image([u8; 16]);
const IMAGE: Image = Image([
    0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70,
]);
/// Every width and ordering loads the same value an ordinary load would, zero- or
/// sign-extended as the explicit conversion says.
#[test]
fn ordered_loads_produce_the_plain_load_value() {
    let image = IMAGE;
    for bits in [8u32, 16, 32, 64] {
        for signed in [false, true] {
            if signed && bits == 64 {
                continue;
            }
            for offset in [0usize, 8] {
                let n = (bits / 8) as usize;
                let mut raw = [0u8; 8];
                raw[..n].copy_from_slice(&image.0[offset..offset + n]);
                let zero = u64::from_le_bytes(raw);
                let expected = if signed {
                    (((zero << (64 - bits)) as i64) >> (64 - bits)) as u64
                } else {
                    zero
                };
                for ordering in LOAD_ORDERINGS {
                    let f = ordered_load(
                        &format!("i{bits}"),
                        signed,
                        ordering_mem(ordering, bits / 8),
                        offset as i64,
                    );
                    let code = Executable::new(&f);
                    let entry: unsafe extern "C" fn(*const u8) -> u64 =
                        unsafe { core::mem::transmute(code.ptr) };
                    assert_eq!(
                        unsafe { entry(image.0.as_ptr()) },
                        expected,
                        "i{bits} signed={signed} offset={offset} {ordering:?}"
                    );
                }
            }
        }
    }
}
/// Every width and ordering (including the `xchg` of a sequentially consistent store) writes
/// exactly its own bytes.
#[test]
fn ordered_stores_write_exactly_their_width() {
    let x = 0x0123_4567_89ab_cdefu64;
    for (kind, bits) in [("i8", 8u32), ("i16", 16), ("i32", 32), ("i64", 64)] {
        for offset in [0usize, 8] {
            let n = (bits / 8) as usize;
            for ordering in STORE_ORDERINGS {
                let f = ordered_store(kind, ordering_mem(ordering, bits / 8), offset as i64);
                let code = Executable::new(&f);
                let entry: unsafe extern "C" fn(*mut u8, u64) =
                    unsafe { core::mem::transmute(code.ptr) };
                let mut out = Image([0xee; 16]);
                unsafe { entry(out.0.as_mut_ptr(), x) };
                let mut want = [0xeeu8; 16];
                want[offset..offset + n].copy_from_slice(&x.to_le_bytes()[..n]);
                assert_eq!(out.0, want, "{kind} offset={offset} {ordering:?}");
            }
        }
    }
}
/// Float and pointer accesses: atomic floats go through an integer register and pointers are
/// 64-bit integers.
#[test]
fn ordered_float_and_pointer_accesses_execute() {
    for ordering in [AtomicOrdering::Acquire, AtomicOrdering::SeqCst] {
        let mut f = Function::new();
        let p = ty(&mut f, "ptr");
        let t = ty(&mut f, "f64");
        let b = f.append_block();
        let ptr = f.append_block_param(b, p);
        let v = f.append_load(b, t, ptr, ordering_mem(Some(ordering), 8));
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        let code = Executable::new(&f);
        let entry: unsafe extern "C" fn(*const f64) -> f64 =
            unsafe { core::mem::transmute(code.ptr) };
        let x = 6.25f64;
        assert_eq!(unsafe { entry(&x) }, 6.25);
    }
    for ordering in [AtomicOrdering::Release, AtomicOrdering::SeqCst] {
        let mut f = Function::new();
        let p = ty(&mut f, "ptr");
        let t = ty(&mut f, "f32");
        let b = f.append_block();
        let ptr = f.append_block_param(b, p);
        let x = f.append_block_param(b, t);
        f.append_store_mem(b, x, ptr, ordering_mem(Some(ordering), 4));
        f.set_terminator(b, Terminator::Ret(Ret::none()));
        let code = Executable::new(&f);
        let entry: unsafe extern "C" fn(*mut f32, f32) = unsafe { core::mem::transmute(code.ptr) };
        let mut out = 0.0f32;
        unsafe { entry(&mut out, -2.75) };
        assert_eq!(out, -2.75);
    }
    let mut f = Function::new();
    let p = ty(&mut f, "ptr");
    let b = f.append_block();
    let src = f.append_block_param(b, p);
    let dst = f.append_block_param(b, p);
    let v = f.append_load(b, p, src, ordering_mem(Some(AtomicOrdering::Acquire), 8));
    f.append_store_mem(b, v, dst, ordering_mem(Some(AtomicOrdering::SeqCst), 8));
    f.set_terminator(b, Terminator::Ret(Ret::none()));
    let code = Executable::new(&f);
    let entry: unsafe extern "C" fn(*const *const u8, *mut *const u8) =
        unsafe { core::mem::transmute(code.ptr) };
    let target = 0x1234usize as *const u8;
    let mut out = core::ptr::null();
    unsafe { entry(&target, &mut out) };
    assert_eq!(out, target);
}
#[test]
fn sign_and_zero_extension_differ() {
    for (kind, want) in [
        (ConvertKind::Sext, -42i64 as u64),
        (ConvertKind::Zext, 0xffff_ffd6),
    ] {
        let mem = Executable::new(&cases::convert("i32", "i64", kind));
        let entry: unsafe extern "C" fn(i32) -> u64 = unsafe { core::mem::transmute(mem.ptr) };
        assert_eq!(unsafe { entry(-42) }, want, "{kind:?}");
    }
}
#[test]
fn float_compare_with_nan_is_unordered() {
    for (cmp, want) in [
        (CmpOp::Eq, 0),
        (CmpOp::Ne, 1),
        (CmpOp::Lt, 0),
        (CmpOp::Le, 0),
        (CmpOp::Gt, 0),
        (CmpOp::Ge, 0),
    ] {
        let mut f = Function::new();
        let t = ty(&mut f, "f64");
        let bool_t = ty(&mut f, "bool");
        let b = f.append_block();
        let l = f.append_block_param(b, t);
        let r = f.append_block_param(b, t);
        let v = f.append_inst(
            b,
            bool_t,
            Opcode::Icmp(Compare {
                op: cmp,
                lhs: l,
                rhs: r,
            }),
        );
        f.set_terminator(b, Terminator::Ret(Ret::one(v)));
        let mem = Executable::new(&f);
        let entry: unsafe extern "C" fn(f64, f64) -> u64 = unsafe { core::mem::transmute(mem.ptr) };
        assert_eq!(unsafe { entry(f64::NAN, 1.0) }, want, "{cmp:?}");
    }
}
#[test]
fn cfg_edge_parameter_cycles() {
    let mut f = Function::new();
    let t = ty(&mut f, "i64");
    let boolean = ty(&mut f, "bool");
    let entry = f.append_block();
    let loop_b = f.append_block();
    let done = f.append_block();
    let a = f.append_block_param(entry, t);
    let b = f.append_block_param(entry, t);
    let zero = f.append_inst(entry, t, Opcode::Iconst(0));
    f.set_jump(entry, loop_b, &[a, b, zero]);
    let x = f.append_block_param(loop_b, t);
    let y = f.append_block_param(loop_b, t);
    let n = f.append_block_param(loop_b, t);
    let one = f.append_inst(loop_b, t, Opcode::Iconst(1));
    let again = f.append_inst(
        loop_b,
        boolean,
        Opcode::Icmp(Compare {
            op: CmpOp::Eq,
            lhs: n,
            rhs: zero,
        }),
    );
    f.append_if(
        loop_b,
        again,
        EdgeDesc {
            target: loop_b,
            args: alloc::vec![y, x, one],
        },
        EdgeDesc {
            target: done,
            args: alloc::vec![x],
        },
    );
    let r = f.append_block_param(done, t);
    f.set_terminator(done, Terminator::Ret(Ret::one(r)));
    let mem = Executable::new(&f);
    let entry: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { core::mem::transmute(mem.ptr) };
    assert_eq!(unsafe { entry(13, 29) }, 29);
}
#[test]
fn vector_reduce_expansion() {
    let mut f = Function::new();
    let p = ty(&mut f, "ptr");
    let v = ty(&mut f, "<4 x f32>");
    let b = f.append_block();
    let ptr = f.append_block_param(b, p);
    let loaded = f.append_inst(
        b,
        v,
        Opcode::Load(Load {
            ptr,
            mem: MemFlags::new(),
        }),
    );
    let r = f.append_reduce(b, BinOp::Add, loaded);
    f.set_terminator(b, Terminator::Ret(Ret::one(r)));
    let mem = Executable::new(&f);
    let entry: unsafe extern "C" fn(*const f32) -> f32 = unsafe { core::mem::transmute(mem.ptr) };
    assert_eq!(unsafe { entry([1.0, 2.0, 3.0, 4.0].as_ptr()) }, 10.0);
}
#[test]
fn indirect_call_preserves_live_values() {
    unsafe extern "C" fn helper(x: u64) -> u64 {
        x * 3
    }
    let mut f = Function::new();
    let t = ty(&mut f, "i64");
    let p = ty(&mut f, "ptr");
    let b = f.append_block();
    let x = f.append_block_param(b, t);
    let target = f.append_inst(b, p, Opcode::Iconst(helper as *const () as usize as i64));
    let args = f.intern_values(&[x]);
    let r = f.append_inst(
        b,
        t,
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
    let sum = f.append_inst(
        b,
        t,
        Opcode::Arith(Arith {
            op: BinOp::Add,
            lhs: r,
            rhs: x,
        }),
    );
    f.set_terminator(b, Terminator::Ret(Ret::one(sum)));
    let mem = Executable::new(&f);
    let entry: unsafe extern "C" fn(u64) -> u64 = unsafe { core::mem::transmute(mem.ptr) };
    assert_eq!(unsafe { entry(13) }, 52);
}
core::arch::global_asm!(
    ".globl volt_test_dirty_return",
    "volt_test_dirty_return:",
    // al = 0x80, everything above it set: the System V ABI leaves those bits undefined.
    "mov rax, -128",
    "ret",
);
unsafe extern "C" {
    fn volt_test_dirty_return() -> u8;
}
/// The bits above a narrow integer a callee returns are undefined, so a call result is
/// canonicalized (zero-extended from its declared width) before anything reads it.
#[test]
fn narrow_call_results_are_zero_extended_from_their_width() {
    for (kind, convert, want) in [
        (ConvertKind::Zext, "i64", 0x80u64),
        (ConvertKind::Sext, "i64", (-128i64) as u64),
        (ConvertKind::Zext, "i32", 0x80),
        (ConvertKind::Sext, "i32", 0xffff_ff80),
    ] {
        let mut f = Function::new();
        let byte = ty(&mut f, "i8");
        let wide = ty(&mut f, convert);
        let p = ty(&mut f, "ptr");
        let b = f.append_block();
        let target = f.append_inst(
            b,
            p,
            Opcode::Iconst(volt_test_dirty_return as *const () as usize as i64),
        );
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
        let w = f.append_inst(b, wide, Opcode::Convert(Convert { value: r, kind }));
        f.set_terminator(b, Terminator::Ret(Ret::one(w)));
        let mem = Executable::new(&f);
        let entry: unsafe extern "C" fn() -> u64 = unsafe { core::mem::transmute(mem.ptr) };
        assert_eq!(unsafe { entry() }, want, "{kind:?} to {convert}");
    }
}
#[test]
fn softfp_relocation_owns_injected_symbol() {
    let f = cases::arith("f128", BinOp::Add);
    let out = isel::compile_object(&f).unwrap();
    assert!(out.relocs.iter().any(|r| r.symbol == "__addtf3"));
    assert_eq!(f.symbol_count(), 0);
}

#[test]
fn narrow_arithmetic_wraps_and_compares_by_operation() {
    let mut f = Function::new();
    let byte = ty(&mut f, "i8");
    let boolean = ty(&mut f, "bool");
    let block = f.append_block();
    let input = f.append_block_param(block, byte);
    let one = f.append_inst(block, byte, Opcode::Iconst(1));
    let sum = f.append_inst(
        block,
        byte,
        Opcode::Arith(Arith {
            op: BinOp::Add,
            lhs: input,
            rhs: one,
        }),
    );
    let zero = f.append_inst(block, byte, Opcode::Iconst(0));
    let equal = f.append_inst(
        block,
        boolean,
        Opcode::Icmp(Compare {
            op: CmpOp::Eq,
            lhs: sum,
            rhs: zero,
        }),
    );
    f.set_terminator(block, Terminator::Ret(Ret::one(equal)));
    let memory = Executable::new(&f);
    let entry: unsafe extern "C" fn(u8) -> u64 = unsafe { core::mem::transmute(memory.ptr) };
    assert_eq!(unsafe { entry(255) }, 1);
    assert_eq!(unsafe { entry(1) }, 0);

    // 0xff is -1 as a signed byte (less than 0) and 255 as an unsigned one (not less).
    for (op, want) in [
        (CmpOp::Slt, 1),
        (CmpOp::Ult, 0),
        (CmpOp::Sgt, 0),
        (CmpOp::Ugt, 1),
    ] {
        let mut f = Function::new();
        let byte = ty(&mut f, "i8");
        let boolean = ty(&mut f, "bool");
        let block = f.append_block();
        let minus_one = f.append_inst(block, byte, Opcode::Iconst(-1));
        let zero = f.append_inst(block, byte, Opcode::Iconst(0));
        let result = f.append_inst(
            block,
            boolean,
            Opcode::Icmp(Compare {
                op,
                lhs: minus_one,
                rhs: zero,
            }),
        );
        f.set_terminator(block, Terminator::Ret(Ret::one(result)));
        let memory = Executable::new(&f);
        let entry: unsafe extern "C" fn() -> u64 = unsafe { core::mem::transmute(memory.ptr) };
        assert_eq!(unsafe { entry() }, want, "{op:?}");
    }
}

#[test]
fn high_product_uses_the_declared_integer_width() {
    for (kind, op, a, b, expected) in [
        ("i8", BinOp::UMulh, 255, 255, 254),
        ("i16", BinOp::UMulh, 65535, 65535, 65534),
        (
            "i48",
            BinOp::UMulh,
            (1u64 << 48) - 1,
            (1u64 << 48) - 1,
            (1u64 << 48) - 2,
        ),
        // 0xfe is -2: the signed high byte of -2 * 3 = -6 is -1, the unsigned one of 254 * 3 = 762
        // is 2.
        ("i8", BinOp::SMulh, 0xfe, 3, 0xff),
        ("i8", BinOp::UMulh, 0xfe, 3, 2),
        ("i16", BinOp::SMulh, 0xfffe, 3, 0xffff),
        ("i16", BinOp::UMulh, 0xfffe, 3, 2),
        ("i32", BinOp::SMulh, 0xffff_fffe, 3, 0xffff_ffff),
        ("i32", BinOp::UMulh, 0xffff_fffe, 3, 2),
        ("i48", BinOp::SMulh, (1u64 << 48) - 2, 3, (1u64 << 48) - 1),
        ("i48", BinOp::UMulh, (1u64 << 48) - 2, 3, 2),
    ] {
        let memory = Executable::new(&cases::arith(kind, op));
        let entry: unsafe extern "C" fn(u64, u64) -> u64 =
            unsafe { core::mem::transmute(memory.ptr) };
        assert_eq!(unsafe { entry(a, b) }, expected, "{kind} {op:?}");
    }
}

#[test]
fn sret_returns_the_original_hidden_pointer_after_its_last_explicit_use() {
    unsafe extern "C" fn consume(value: u64) {
        core::hint::black_box(value);
    }
    let mut f = Function::new();
    f.sret = true;
    let pointer = ty(&mut f, "ptr");
    let word = ty(&mut f, "i64");
    let block = f.append_block();
    let destination = f.append_block_param(block, pointer);
    let value = f.append_inst(block, word, Opcode::Iconst(7));
    f.append_store(block, value, destination);
    let argument = f.append_inst(block, word, Opcode::Iconst(42));
    let helper = f.append_inst(
        block,
        pointer,
        Opcode::Iconst(consume as *const () as usize as i64),
    );
    f.append_void_call_indirect(block, helper, &[argument]);
    f.set_terminator(block, Terminator::Ret(Ret::none()));
    let memory = Executable::new(&f);
    let entry: unsafe extern "C" fn(*mut u64) -> *mut u64 =
        unsafe { core::mem::transmute(memory.ptr) };
    let mut output = 0;
    assert_eq!(unsafe { entry(&mut output) }, &mut output as *mut u64);
    assert_eq!(output, 7);
}
