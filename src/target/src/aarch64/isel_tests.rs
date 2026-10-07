// Rust selector boundary tests for the Vulcan-derived AArch64 port (Apache-2.0).
use super::*;
use volt_ir::function::{Arith, BinOp, Function, Opcode, Ret, Terminator};
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
