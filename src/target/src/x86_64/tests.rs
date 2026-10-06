// Ported from Vulcan native/cases/link test boundaries (Apache-2.0).
// Copyright LilithSemi; Rust implementation modified for Volt.
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
fn binary(kind: &str, op: BinOp) -> Function {
    let mut f = Function::new();
    let t = ty(&mut f, kind);
    let b = f.append_block();
    let l = f.append_block_param(b, t);
    let r = f.append_block_param(b, t);
    let out = f.append_inst(b, t, Opcode::Arith(Arith { op, lhs: l, rhs: r }));
    f.set_terminator(b, Terminator::Ret(Ret::one(out)));
    f
}
#[test]
fn scalar_division_shifts_and_high_product() {
    for (kind, op, a, b, want) in [
        ("u64", BinOp::Div, u64::MAX, 2, u64::MAX / 2),
        ("u64", BinOp::Rem, 100, 9, 1),
        ("u64", BinOp::Shr, 1 << 63, 63, 1),
        ("u64", BinOp::Shl, 7, 65, 14),
        ("u64", BinOp::Mulh, u64::MAX, u64::MAX, u64::MAX - 1),
        ("i64", BinOp::Div, (-100i64) as u64, 9, (-11i64) as u64),
        ("i64", BinOp::Mulh, (-2i64) as u64, 3, u64::MAX),
    ] {
        let mem = Executable::new(&binary(kind, op));
        // SAFETY: the compiled function has two 64-bit integer parameters and one result.
        let entry: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { core::mem::transmute(mem.ptr) };
        assert_eq!(unsafe { entry(a, b) }, want, "{kind} {op:?}");
    }
}
#[test]
fn narrow_store_keeps_neighbor_bytes() {
    let mut f = Function::new();
    let p = ty(&mut f, "ptr");
    let byte = ty(&mut f, "u8");
    let word = ty(&mut f, "u32");
    let b = f.append_block();
    let ptr = f.append_block_param(b, p);
    let v = f.append_inst(b, byte, Opcode::Iconst(0xab));
    f.append_store(b, v, ptr);
    let r = f.append_inst(
        b,
        word,
        Opcode::Load(Load {
            ptr,
            volatile: false,
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
fn signed_conversion_and_float_nan_compare() {
    let mut f = Function::new();
    let i = ty(&mut f, "i32");
    let o = ty(&mut f, "i64");
    let b = f.append_block();
    let v = f.append_block_param(b, i);
    let r = f.append_inst(b, o, Opcode::Convert(Convert { value: v }));
    f.set_terminator(b, Terminator::Ret(Ret::one(r)));
    let mem = Executable::new(&f);
    let entry: unsafe extern "C" fn(i32) -> i64 = unsafe { core::mem::transmute(mem.ptr) };
    assert_eq!(unsafe { entry(-42) }, -42);
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
        assert_eq!(unsafe { entry(f64::NAN, 1.0) }, want);
    }
}
#[test]
fn cfg_edge_parameter_cycles() {
    let mut f = Function::new();
    let t = ty(&mut f, "u64");
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
            volatile: false,
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
    let t = ty(&mut f, "u64");
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
#[test]
fn softfp_relocation_owns_injected_symbol() {
    let f = binary("f128", BinOp::Add);
    let out = isel::compile_object(&f).unwrap();
    assert!(out.relocs.iter().any(|r| r.symbol == "__addtf3"));
    assert_eq!(f.symbol_count(), 0);
}

#[test]
fn narrow_arithmetic_wraps_and_compares_signed_values() {
    let mut f = Function::new();
    let byte = ty(&mut f, "u8");
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

    let mut f = Function::new();
    let signed = ty(&mut f, "i8");
    let boolean = ty(&mut f, "bool");
    let block = f.append_block();
    let minus_one = f.append_inst(block, signed, Opcode::Iconst(-1));
    let zero = f.append_inst(block, signed, Opcode::Iconst(0));
    let negative = f.append_inst(
        block,
        boolean,
        Opcode::Icmp(Compare {
            op: CmpOp::Lt,
            lhs: minus_one,
            rhs: zero,
        }),
    );
    f.set_terminator(block, Terminator::Ret(Ret::one(negative)));
    let memory = Executable::new(&f);
    let entry: unsafe extern "C" fn() -> u64 = unsafe { core::mem::transmute(memory.ptr) };
    assert_eq!(unsafe { entry() }, 1);
}

#[test]
fn high_product_uses_the_declared_integer_width() {
    for (kind, a, b, expected) in [
        ("u8", 255, 255, 254),
        ("u16", 65535, 65535, 65534),
        ("u48", (1u64 << 48) - 1, (1u64 << 48) - 1, (1u64 << 48) - 2),
        ("i8", (-2i64) as u64, 3, 255),
    ] {
        let memory = Executable::new(&binary(kind, BinOp::Mulh));
        let entry: unsafe extern "C" fn(u64, u64) -> u64 =
            unsafe { core::mem::transmute(memory.ptr) };
        let mask = if kind == "u48" {
            (1u64 << 48) - 1
        } else if kind == "u16" {
            65535
        } else {
            255
        };
        assert_eq!(unsafe { entry(a, b) } & mask, expected, "{kind}");
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
    let word = ty(&mut f, "u64");
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
