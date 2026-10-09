//! Tests lift real encodings with a flat test environment and run them on the
//! host through the native JIT.
use crate::{Builder, Environment, LiftError, Lifter, MemoryAccess};
use alloc::vec::Vec;
use core::mem::offset_of;
use volt_ir::function::{BinOp as B, Function, Value};
use volt_target::{
    aarch64::decode::*,
    native::{self, JittedFunction},
};

/// Guest state of the test environment.
#[repr(C)]
#[derive(Default)]
struct State {
    x: [u64; 31],
    sp: u64,
    flags: u32,
    v: [[u64; 2]; 32],
}

/// Memory is the host's own: a guest address is a host pointer and every
/// load or store is an ordinary IR load or store. Counts the lifter's hooks.
#[derive(Default)]
struct Flat {
    begun: usize,
    exits: usize,
}
impl Flat {
    /// `address + delta` as a host pointer.
    fn host(b: &Builder, address: Value, delta: u64) -> Value {
        b.imm(b.ptr, B::Add, address, delta)
    }
    fn load(b: &Builder, address: Value, delta: u64, size: Size, rt: u8, signed: SignExtend) {
        let bits = size as u16 * 8;
        let raw = b.load_ptr(b.int(bits), Self::host(b, address, delta));
        let mut v = b.cv(b.i64, raw);
        if signed != SignExtend::None && bits < 64 {
            let n = 64 - bits as u64;
            let extended = b.shift(Width::X64, b.imm(b.i64, B::Shl, v, n), Shift::Asr, n);
            v = if signed == SignExtend::To32 {
                b.imm(b.i64, B::BitAnd, extended, 0xffff_ffff)
            } else {
                extended
            };
        }
        b.put(rt, v);
    }
    fn store(b: &Builder, address: Value, delta: u64, size: Size, rt: u8) {
        let t = b.int(size as u16 * 8);
        b.store_ptr(
            Self::host(b, address, delta),
            b.cv(t, b.reg(b.i64, rt, true)),
        );
    }
    fn vector(b: &Builder, address: Value, bytes: u8, rt: u8, op: MemoryOp, delta: u64) {
        let at = b.v_at(rt);
        let chunk = b.int(bytes.min(8) as u16 * 8);
        let chunks = if bytes == 16 { 2 } else { 1 };
        for i in 0..chunks {
            let host = Self::host(b, address, delta + i as u64 * 8);
            if op == MemoryOp::Store {
                b.store_ptr(host, b.cv(chunk, b.load(b.i64, at + i * 8)));
            } else {
                b.store(at + i * 8, b.cv(b.i64, b.load_ptr(chunk, host)));
            }
        }
        if op == MemoryOp::Load && bytes <= 8 {
            b.store(at + 8, b.k(b.i64, 0));
        }
    }
}
impl Environment for Flat {
    fn x_offset(&self, r: u8) -> usize {
        offset_of!(State, x) + r as usize * 8
    }
    fn sp_offset(&self) -> usize {
        offset_of!(State, sp)
    }
    fn flags_offset(&self) -> usize {
        offset_of!(State, flags)
    }
    fn v_offset(&self, r: u8) -> usize {
        offset_of!(State, v) + r as usize * 16
    }
    fn begin_instruction(&mut self, _: &Builder) {
        self.begun += 1;
    }
    fn exit(&mut self, b: &Builder, next_pc: Value) {
        self.exits += 1;
        b.set_ret(next_pc)
    }
    fn memory(&mut self, b: &Builder, _: u64, access: MemoryAccess<'_>) -> Result<(), LiftError> {
        let MemoryAccess {
            insn,
            address,
            writeback,
        } = access;
        match (*insn, address) {
            (Instruction::Memory(a), Some(address)) if a.op == MemoryOp::Store => {
                Self::store(b, address, 0, a.size, a.rt)
            }
            (Instruction::Memory(a), Some(address)) => {
                Self::load(b, address, 0, a.size, a.rt, a.signed)
            }
            (Instruction::Literal(a), Some(address)) => {
                Self::load(b, address, 0, a.size, a.rt, SignExtend::None)
            }
            (Instruction::Pair(a), Some(address)) if a.op == MemoryOp::Store => {
                Self::store(b, address, 0, a.size, a.rt);
                Self::store(b, address, a.size as u64, a.size, a.rt2);
            }
            (Instruction::Pair(a), Some(address)) => {
                Self::load(b, address, 0, a.size, a.rt, a.signed);
                Self::load(b, address, a.size as u64, a.size, a.rt2, a.signed);
            }
            (Instruction::SimdMemory(a), Some(address)) => {
                Self::vector(b, address, a.bytes, a.rt, a.op, 0)
            }
            (Instruction::SimdLiteral(a), Some(address)) => {
                Self::vector(b, address, a.bytes, a.rt, MemoryOp::Load, 0)
            }
            (Instruction::SimdPair(a), Some(address)) => {
                Self::vector(b, address, a.bytes, a.rt, a.op, 0);
                Self::vector(b, address, a.bytes, a.rt2, a.op, a.bytes as u64);
            }
            _ => return Err(LiftError::Unsupported),
        }
        if let Some(w) = writeback {
            b.put_sp(w.register, w.value);
        }
        Ok(())
    }
    fn event(&mut self, _: &Builder, _: u64, _: &Instruction) -> Result<(), LiftError> {
        Err(LiftError::Unsupported)
    }
}

/// Lift `bytes` as one block at `pc`: instructions until one terminates it
/// (plain memory accesses do not), then an exit at the next pc.
fn lift(pc: u64, bytes: &[u8]) -> Result<(Function, Flat), LiftError> {
    let mut lifter = Lifter::new(Flat::default());
    let mut at = pc;
    let mut ended = false;
    for chunk in bytes.chunks_exact(4) {
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        let instruction = decode(word).unwrap();
        ended = instruction.terminates_with(true);
        lifter.lift(at, instruction)?;
        at += 4;
        if ended {
            break;
        }
    }
    if !ended {
        lifter.exit(at);
    }
    Ok(lifter.into_parts())
}
struct Block {
    _image: JittedFunction,
    entry: unsafe extern "C" fn(*mut State) -> u64,
}
impl Block {
    fn run(&self, state: &mut State) -> u64 {
        // SAFETY: the lifted function has exactly this signature and `_image`
        // keeps its mapping alive.
        unsafe { (self.entry)(state) }
    }
}
fn compile(pc: u64, bytes: &[u8]) -> Result<Block, LiftError> {
    let (function, _) = lift(pc, bytes)?;
    let image = native::compile_owned(function).unwrap();
    // SAFETY: see `Block::run`.
    let entry = unsafe { image.entry() };
    Ok(Block {
        _image: image,
        entry,
    })
}
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[test]
fn zero_division_and_signed_overflow_do_not_trap_host() {
    // UDIV X2,X0,X1; SDIV X3,X0,X1.
    let block = compile(0, &bytes(&[0x9ac10802, 0x9ac10c03])).unwrap();
    let mut cpu = State::default();
    cpu.x[0] = 1 << 63;
    cpu.x[1] = 0;
    block.run(&mut cpu);
    assert_eq!(cpu.x[2], 0);
    assert_eq!(cpu.x[3], 0);
    cpu.x[1] = u64::MAX;
    block.run(&mut cpu);
    assert_eq!(cpu.x[3], 1 << 63);
}
#[test]
fn signed_and_unsigned_integer_ops_select_distinct_semantics() {
    // sdiv/udiv/asrv (w and x), smulh/umulh, sxtw, sbfx, asr #imm, then
    // `cmp` + `cset` for each signed/unsigned condition on w and x.
    let words = [
        0x1ac10c02, 0x1ac10803, 0x1ac12804, 0x9ac12805, 0x9b417c06, 0x9bc17c07, 0x93407c08,
        0x93442c09, 0x13037c0a, 0x6b01001f, 0x1a9fa7eb, 0x1a9fb7ec, 0x1a9f27ed, 0x1a9f97ee,
        0x1a9fd7ef, 0x1a9fc7f0, 0x1a9f87f1, 0xeb01001f, 0x9a9fa7f2, 0x9a9f27f3, 0x9a9fd7f4,
    ];
    let block = compile(0, &bytes(&words)).unwrap();
    let pairs: [(u64, u64); 9] = [
        (0xffff_ffff_8000_0005, 3),
        (0x8000_0000, 0xffff_ffff),
        (0x7fff_ffff_ffff_ffff, 0x8000_0000_0000_0000),
        (0x8000_0000_0000_0000, u64::MAX),
        (3, 0),
        (7, 7),
        (0xffff_ffff_0000_0007, 0x0000_0001_0000_0009),
        (0x8000_0001, 2),
        (2, 0x8000_0001),
    ];
    for (a, b) in pairs {
        let mut cpu = State::default();
        cpu.x[0] = a;
        cpu.x[1] = b;
        block.run(&mut cpu);
        let (w, wb) = (a as u32, b as u32);
        let (sw, swb) = (w as i32, wb as i32);
        let sdiv = if wb == 0 {
            0
        } else {
            sw.wrapping_div(swb) as u32
        };
        let udiv = if wb == 0 { 0 } else { w / wb };
        assert_eq!(cpu.x[2], sdiv as u64, "sdiv w {a:#x} {b:#x}");
        assert_eq!(cpu.x[3], udiv as u64, "udiv w {a:#x} {b:#x}");
        assert_eq!(cpu.x[4], (sw >> (wb & 31)) as u32 as u64, "asr w");
        assert_eq!(cpu.x[5], ((a as i64) >> (b & 63)) as u64, "asr x");
        assert_eq!(
            cpu.x[6],
            ((a as i64 as i128 * b as i64 as i128) >> 64) as u64,
            "smulh"
        );
        assert_eq!(cpu.x[7], ((a as u128 * b as u128) >> 64) as u64, "umulh");
        assert_eq!(cpu.x[8], w as i32 as i64 as u64, "sxtw");
        assert_eq!(cpu.x[9], ((a >> 4) as u8) as i8 as i64 as u64, "sbfx");
        assert_eq!(cpu.x[10], (sw >> 3) as u32 as u64, "asr w imm");
        // Registers 11-17 come from the 32-bit compare, 18-20 from the 64-bit one.
        let w_flags = [
            sw < swb,
            sw >= swb,
            w < wb,
            w > wb,
            sw > swb,
            sw <= swb,
            w <= wb,
        ];
        for (i, want) in w_flags.into_iter().enumerate() {
            assert_eq!(cpu.x[11 + i], want as u64, "w cond {i} {a:#x} {b:#x}");
        }
        let (sa, sb) = (a as i64, b as i64);
        assert_eq!(cpu.x[18], (sa < sb) as u64, "x lt");
        assert_eq!(cpu.x[19], (a < b) as u64, "x lo");
        assert_eq!(cpu.x[20], (sa > sb) as u64, "x gt");
    }
}
#[test]
fn signed_and_unsigned_vector_lanes_select_distinct_semantics() {
    // smax.16b, sabd.8h, shadd.4s, sshl.4s, sqadd.16b, cmgt.2d, cmhi.2d.
    let words = [
        0x4e216403, 0x4e617404, 0x4ea10405, 0x4ea14406, 0x4e210c07, 0x4ee13409, 0x6ee1340a,
    ];
    let block = compile(0, &bytes(&words)).unwrap();
    let a: [u8; 16] = [
        0x80, 0x7f, 0x01, 0xff, 0x10, 0x90, 0x00, 0x7e, 0x85, 0x05, 0xfe, 0x40, 0x7f, 0x80, 0xc0,
        0x20,
    ];
    let b: [u8; 16] = [
        0x01, 0x80, 0xff, 0x01, 0x90, 0x10, 0x00, 0x7f, 0x05, 0x85, 0x02, 0xc0, 0x7f, 0x7f, 0x40,
        0xe0,
    ];
    let pack = |x: [u8; 16]| {
        let v = u128::from_le_bytes(x);
        [v as u64, (v >> 64) as u64]
    };
    let mut cpu = State::default();
    cpu.v[0] = pack(a);
    cpu.v[1] = pack(b);
    block.run(&mut cpu);
    let out = |r: usize| (u128::from(cpu.v[r][1]) << 64 | u128::from(cpu.v[r][0])).to_le_bytes();
    let lanes16 = |x: [u8; 16]| -> Vec<i16> {
        x.chunks(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let lanes32 = |x: [u8; 16]| -> Vec<i32> {
        x.chunks(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    // smax.16b
    let want: Vec<u8> = (0..16)
        .map(|i| (a[i] as i8).max(b[i] as i8) as u8)
        .collect();
    assert_eq!(out(3).to_vec(), want, "smax");
    // sabd.8h
    let want: Vec<u8> = lanes16(a)
        .iter()
        .zip(lanes16(b))
        .flat_map(|(x, y)| ((*x as i32 - y as i32).unsigned_abs() as u16).to_le_bytes())
        .collect();
    assert_eq!(out(4).to_vec(), want, "sabd");
    // shadd.4s
    let want: Vec<u8> = lanes32(a)
        .iter()
        .zip(lanes32(b))
        .flat_map(|(x, y)| (((*x as i64 + y as i64) >> 1) as i32).to_le_bytes())
        .collect();
    assert_eq!(out(5).to_vec(), want, "shadd");
    // sshl.4s: the shift amount is the signed low byte of each lane of b.
    let want: Vec<u8> = lanes32(a)
        .iter()
        .zip(lanes32(b))
        .flat_map(|(x, y)| {
            let n = y as i8 as i32;
            let r = if n >= 32 {
                0
            } else if n >= 0 {
                x << n
            } else if n > -32 {
                x >> -n
            } else if *x < 0 {
                -1
            } else {
                0
            };
            r.to_le_bytes()
        })
        .collect();
    assert_eq!(out(6).to_vec(), want, "sshl");
    // sqadd.16b
    let want: Vec<u8> = (0..16)
        .map(|i| (a[i] as i8).saturating_add(b[i] as i8) as u8)
        .collect();
    assert_eq!(out(7).to_vec(), want, "sqadd");
    // cmgt.2d / cmhi.2d
    for (r, signed) in [(9, true), (10, false)] {
        for lane in 0..2 {
            let (x, y) = (cpu.v[0][lane], cpu.v[1][lane]);
            let gt = if signed {
                (x as i64) > (y as i64)
            } else {
                x > y
            };
            assert_eq!(
                cpu.v[r][lane],
                if gt { u64::MAX } else { 0 },
                "cmp {r} {lane}"
            );
        }
    }
}
#[test]
fn simd_compare_zero_masks_each_lane() {
    // cmeq v1.16b, v0.16b, #0.
    let block = compile(0x1000, &bytes(&[0x4e209801])).unwrap();
    let mut cpu = State::default();
    cpu.v[0][0] = 0x00ff_0000_0000_00ff;
    cpu.v[0][1] = 0xffff_ffff_ffff_ffff;
    block.run(&mut cpu);
    assert_eq!(cpu.v[1][0], 0xff00_ffff_ffff_ff00);
    assert_eq!(cpu.v[1][1], 0);
    // Half-width form operates on v0.8b and clears the top half.
    let block = compile(0x1000, &bytes(&[0x0e209801])).unwrap();
    cpu.v[0][0] = 0x00ff_0000_0000_00ff;
    cpu.v[0][1] = 0x1234_5678_9abc_def0;
    cpu.v[1] = [0xdead_beef_dead_beef, 0xdead_beef_dead_beef];
    block.run(&mut cpu);
    assert_eq!(cpu.v[1][0], 0xff00_ffff_ffff_ff00);
    assert_eq!(cpu.v[1][1], 0);
}
#[test]
fn simd_fmov_moves_bits_between_gpr_and_fpr() {
    // fmov x2, d2 then fmov d3, x2 round-trips the low doubleword.
    let block = compile(0x1000, &bytes(&[0x9e660042, 0x9e670043])).unwrap();
    let mut cpu = State::default();
    cpu.v[2] = [0x1122_3344_5566_7788, 0xdead_beef_dead_beef];
    block.run(&mut cpu);
    assert_eq!(cpu.x[2], 0x1122_3344_5566_7788);
    assert_eq!(cpu.v[3], [0x1122_3344_5566_7788, 0]);
    // 32-bit form truncates to the low word.
    let block = compile(0x1000, &bytes(&[0x1e260022])).unwrap();
    cpu.v[1] = [0xaaaa_bbbb_cccc_dddd, 0];
    block.run(&mut cpu);
    assert_eq!(cpu.x[2] as u32, 0xcccc_dddd);
}
#[test]
fn simd_pairwise_reductions_match_lane_layout() {
    let setup = || {
        let mut cpu = State::default();
        cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
        cpu.v[2] = [0x1112_1314_1516_1718, 0x191a_1b1c_1d1e_1f20];
        cpu
    };
    for (word, want) in [
        (
            0x6e22a420u32,
            [0x0a0c_0e10_0204_0608u64, 0x1a1c_1e20_1214_1618],
        ),
        (0x6e22ac20, [0x090b_0d0f_0103_0507, 0x191b_1d1f_1113_1517]),
        (0x4e22bc20, [0x1317_1b1f_0307_0b0f, 0x3337_3b3f_2327_2b2f]),
        (0x4ea2bc20, [0x1618_1a1c_0608_0a0c, 0x3638_3a3c_2628_2a2c]),
        (0x4ee2bc20, [0x0a0c_0e10_1214_1618, 0x2a2c_2e30_3234_3638]),
    ] {
        let mut cpu = setup();
        let block = compile(0x1000, &bytes(&[word])).unwrap();
        block.run(&mut cpu);
        assert_eq!(cpu.v[0], want, "{word:08x}");
    }
}
#[test]
fn simd_horizontal_add_sums_lanes() {
    for (word, lanes, want) in [
        (0x4e31b820u32, 16usize, 0x88u64),
        (0x4e71b820, 8, 0x4048),
        (0x4eb1b820, 4, 0x1c20_2428),
    ] {
        let block = compile(0x1000, &bytes(&[word])).unwrap();
        let mut cpu = State::default();
        cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
        block.run(&mut cpu);
        let _ = lanes;
        assert_eq!(cpu.v[0], [want, 0], "{word:08x}");
    }
    // The 64-bit form reads only the low half of the source.
    let block = compile(0x1000, &bytes(&[0x0e31b820])).unwrap();
    let mut cpu = State::default();
    cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
    block.run(&mut cpu);
    assert_eq!(cpu.v[0], [0x24, 0]);
}
#[test]
fn simd_dup_from_element_replicates_a_lane() {
    for (word, want) in [
        // dup v0.4h, v1.h[0] -> element 0x0708 in every halfword.
        (0x0e020420u32, [0x0708_0708_0708_0708u64, 0]),
        // dup v0.8h, v1.h[7] -> element 0x090a, both halves.
        (0x4e1e0420, [0x090a_090a_090a_090a, 0x090a_090a_090a_090a]),
        // dup v0.4s, v1.s[1] -> element 0x0102_0304.
        (0x4e0c0420, [0x0102_0304_0102_0304, 0x0102_0304_0102_0304]),
        // dup v0.2d, v1.d[1] -> element 0x090a_0b0c_0d0e_0f10.
        (0x4e180420, [0x090a_0b0c_0d0e_0f10, 0x090a_0b0c_0d0e_0f10]),
        // dup v0.16b, v1.b[3] -> element 0x05.
        (0x4e070420, [0x0505_0505_0505_0505, 0x0505_0505_0505_0505]),
    ] {
        let mut cpu = State::default();
        cpu.v[1] = [0x0102_0304_0506_0708, 0x090a_0b0c_0d0e_0f10];
        let block = compile(0x1000, &bytes(&[word])).unwrap();
        block.run(&mut cpu);
        assert_eq!(cpu.v[0], want, "{word:08x}");
    }
}
#[test]
fn simd_table_lookup_selects_bytes() {
    // tbl v0.16b, {v1.16b, v2.16b}, v3.16b with indices 0,2,4,...
    let block = compile(0x1000, &bytes(&[0x4e032020])).unwrap();
    let mut cpu = State::default();
    cpu.v[1] = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
    cpu.v[2] = [0x1716_1514_1312_1110, 0x1f1e_1d1c_1b1a_1918];
    cpu.v[3] = [0x0604_0200_0e0c_0a08, 0x1614_1210_1e1c_1a18];
    block.run(&mut cpu);
    assert_eq!(cpu.v[0], [0x0604_0200_0e0c_0a08, 0x1614_1210_1e1c_1a18]);
    // Out-of-range indices yield zero for tbl.
    let block = compile(0x1000, &bytes(&[0x4e020020])).unwrap();
    let mut cpu = State::default();
    cpu.v[1] = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
    cpu.v[2] = [0xff01_ff00_ff00_ff00, 0xffff_ffff_ffff_ffff];
    block.run(&mut cpu);
    assert_eq!(cpu.v[0], [0x0001_0000_0000_0000, 0]);
    // tbx keeps the destination byte where the index is out of range.
    let block = compile(0x1000, &bytes(&[0x4e021020])).unwrap();
    let mut cpu = State::default();
    cpu.v[1] = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
    cpu.v[2] = [0xff01_ff00_ff00_ff00, 0xffff_ffff_ffff_ffff];
    cpu.v[0] = [0xdead_beef_dead_beef, 0xdead_beef_dead_beef];
    block.run(&mut cpu);
    assert_eq!(cpu.v[0], [0xde01_be00_de00_be00, 0xdead_beef_dead_beef]);
}
#[test]
fn simd_alu_lane_semantics() {
    // Ground truth computed with the host's own byte arithmetic.
    struct Case {
        word: u32,
        a: [u64; 2],
        b: [u64; 2],
        want: [u64; 2],
    }
    let cases = [
        // add v0.16b, v1.16b, v2.16b wraps per byte
        Case {
            word: 0x4e228420,
            a: [0x0102_03ff_0405_0607, 0x0f10_1120_2122_2324],
            b: [0x0101_0101_0101_0101, 0x0102_0304_0506_0708],
            want: [0x0203_0400_0506_0708, 0x1012_1424_2628_2a2c],
        },
        // sub v0.16b, v1.16b, v2.16b
        Case {
            word: 0x6e228420,
            a: [0x0001_0203_0405_0607, 0],
            b: [0x0101_0101_0101_0101, 0],
            want: [0xff00_0102_0304_0506, 0],
        },
        // cmhs v0.16b, v1.16b, v2.16b (unsigned >=)
        Case {
            word: 0x6e223c20,
            a: [0x00ff_0080_7f7f_8000, 0],
            b: [0x0001_0180_7f80_8000, 0],
            want: [0xffff_00ff_ff00_ffff, 0xffff_ffff_ffff_ffff],
        },
        // umin v0.8h, v1.8h, v2.8h
        Case {
            word: 0x6e626c20,
            a: [0x0003_0002_0001_0000, 0],
            b: [0x0002_0003_0000_0001, 0],
            want: [0x0002_0002_0000_0000, 0],
        },
        // uqadd v0.8h, v1.8h, v2.8h saturates
        Case {
            word: 0x6e620c20,
            a: [0xffff_8000_0001_0000, 0],
            b: [0x0001_9000_0002_0001, 0],
            want: [0xffff_ffff_0003_0001, 0],
        },
        // shadd v0.16b, v1.16b, v2.16b rounds down (arithmetic)
        Case {
            word: 0x4e220420,
            a: [0x0303_fefe_0505_00ff, 0],
            b: [0x0303_fefe_0505_00ff, 0],
            want: [0x0303_fefe_0505_00ff, 0],
        },
        // sshl v0.8h by signed amounts in the low byte of each lane
        Case {
            word: 0x4e624420,
            a: [0x0004_0003_0002_0001, 0],
            b: [0x0000_0002_00ff_0001, 0],
            want: [0x0004_000c_0001_0002, 0],
        },
        // mla/mls fold in the destination *lane*, not the whole half
        Case {
            word: 0x4e229420,
            a: [0x0202_0202_0202_0202, 0],
            b: [0x0505_0505_0505_0505, 0],
            want: [0x0a0a_0a0a_0a0a_0a0a, 0],
        },
        Case {
            word: 0x6e229420,
            a: [0x0202_0202_0202_0202, 0],
            b: [0x0505_0505_0505_0505, 0],
            want: [0xf6f6_f6f6_f6f6_f6f6, 0],
        },
        // bsl: destination starts zero, so the result is v2 unchanged
        Case {
            word: 0x6e621c20,
            a: [0xaaaa_aaaa_aaaa_aaaa, 0],
            b: [0x5555_5555_5555_5555, 0],
            want: [0x5555_5555_5555_5555, 0],
        },
        // bit with a zero destination inserts the masked v1 bits
        Case {
            word: 0x6ea21c20,
            a: [0x0f0f_0f0f_0f0f_0f0f, 0],
            b: [0xaaaa_aaaa_aaaa_aaaa, 0],
            want: [0x0a0a_0a0a_0a0a_0a0a, 0],
        },
        // bif with a zero destination inserts the inverse-masked bits
        Case {
            word: 0x6ee21c20,
            a: [0x0f0f_0f0f_0f0f_0f0f, 0],
            b: [0xaaaa_aaaa_aaaa_aaaa, 0],
            want: [0x0505_0505_0505_0505, 0],
        },
        // shadd of two -1 bytes stays -1 (arithmetic halving)
        Case {
            word: 0x4e220420,
            a: [0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff],
            b: [0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff],
            want: [0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff],
        },
        // shsub of 0 - 1 halves to -1
        Case {
            word: 0x4e222420,
            a: [0, 0],
            b: [0x0101_0101_0101_0101, 0],
            want: [0xffff_ffff_ffff_ffff, 0],
        },
        // srhadd of 1 and 2 rounds to 2
        Case {
            word: 0x4e221420,
            a: [0x0101_0101_0101_0101, 0],
            b: [0x0202_0202_0202_0202, 0],
            want: [0x0202_0202_0202_0202, 0],
        },
        // sshl v0.8h: lane 0 shifts left by 1, lane 1 right by 1
        Case {
            word: 0x4e624420,
            a: [0x0008_0004_0002_0001, 0],
            b: [0x0000_00ff_0001_0001, 0],
            want: [0x0008_0002_0004_0002, 0],
        },
        // uqshl v0.8h saturates an overflow to the lane maximum
        Case {
            word: 0x6e624c20,
            a: [0x0000_0000_0000_8000, 0],
            b: [0x0000_0000_0000_0001, 0],
            want: [0x0000_0000_0000_ffff, 0],
        },
        // sqshl v0.8h saturates a positive overflow to the lane maximum
        Case {
            word: 0x4e624c20,
            a: [0x0000_0000_0000_4000, 0],
            b: [0x0000_0000_0000_0001, 0],
            want: [0x0000_0000_0000_7fff, 0],
        },
        // sqrshl v0.8h rounds before shifting right
        Case {
            word: 0x4e625c20,
            a: [0x0000_0000_0000_0003, 0],
            b: [0x0000_0000_0000_00ff, 0],
            want: [0x0000_0000_0000_0002, 0],
        },
    ];
    for case in cases {
        let block = compile(0x1000, &bytes(&[case.word])).unwrap();
        let mut cpu = State::default();
        cpu.v[1] = case.a;
        cpu.v[2] = case.b;
        block.run(&mut cpu);
        assert_eq!(cpu.v[0], case.want, "{:08x}", case.word);
    }
}

fn sub_flags(a: u64, b: u64) -> u32 {
    let r = a.wrapping_sub(b);
    let (n, z, c) = (r >> 63, r == 0, a >= b);
    let v = ((a ^ b) & (a ^ r)) >> 63;
    (n as u32) << 31 | (z as u32) << 30 | (c as u32) << 29 | (v as u32) << 28
}
fn add_flags32(a: u32, b: u32) -> u32 {
    let (r, c) = a.overflowing_add(b);
    let v = (!(a ^ b) & (a ^ r)) >> 31;
    (r >> 31) << 31 | ((r == 0) as u32) << 30 | (c as u32) << 29 | v << 28
}
#[test]
fn subs_and_adds_set_nzcv() {
    // subs x2,x0,x1; adds w3,w0,w1.
    let block = compile(0, &bytes(&[0xeb010002, 0x2b010003])).unwrap();
    for (a, b) in [
        (5u64, 5u64),
        (3, 5),
        (5, 3),
        (1 << 63, 1),
        (0xffff_ffff, 1),
        (0x7fff_ffff, 1),
        (u64::MAX, u64::MAX),
    ] {
        let mut cpu = State::default();
        cpu.x[0] = a;
        cpu.x[1] = b;
        block.run(&mut cpu);
        assert_eq!(cpu.x[2], a.wrapping_sub(b));
        assert_eq!(cpu.x[3], (a as u32).wrapping_add(b as u32) as u64);
        assert_eq!(
            cpu.flags,
            add_flags32(a as u32, b as u32),
            "adds {a:#x} {b:#x}"
        );
        // The flags of the first instruction were overwritten by the second;
        // check them on their own.
        let sub = compile(0, &bytes(&[0xeb010002])).unwrap();
        sub.run(&mut cpu);
        assert_eq!(cpu.flags, sub_flags(a, b), "subs {a:#x} {b:#x}");
    }
}
#[test]
fn conditional_select_and_compare_read_the_flags() {
    // subs x2,x0,x1; csel x4,x0,x1,lt.
    let block = compile(0, &bytes(&[0xeb010002, 0x9a81b004])).unwrap();
    for (a, b) in [(3u64, 5u64), (5, 3), (5, 5), (1 << 63, 1)] {
        let mut cpu = State::default();
        cpu.x[0] = a;
        cpu.x[1] = b;
        block.run(&mut cpu);
        let want = if (a as i64) < (b as i64) { a } else { b };
        assert_eq!(cpu.x[4], want, "csel lt {a:#x} {b:#x}");
    }
    // ccmp x0,x1,#4,eq: compares when EQ holds, otherwise loads the immediate.
    let block = compile(0, &bytes(&[0xfa410004])).unwrap();
    let mut cpu = State::default();
    cpu.x[0] = 7;
    cpu.x[1] = 9;
    cpu.flags = 0x4000_0000;
    block.run(&mut cpu);
    assert_eq!(cpu.flags, sub_flags(7, 9));
    cpu.flags = 0;
    block.run(&mut cpu);
    assert_eq!(cpu.flags, 0x4000_0000);
}
#[test]
fn zero_register_destinations_do_not_overwrite_sp() {
    // ORR XZR,X0,X0; ADR XZR,. ; ADDS XZR,X0,X0.
    let block = compile(0x1000, &bytes(&[0xaa00001f, 0x1000001f, 0xab00001f])).unwrap();
    let mut cpu = State::default();
    cpu.sp = 0x9000;
    cpu.x[0] = 0x1234;
    block.run(&mut cpu);
    assert_eq!(cpu.sp, 0x9000);
    // CBZ XZR,+8 must take the branch independently of SP.
    let branch = compile(0x2000, &bytes(&[0xb400005f])).unwrap();
    assert_eq!(branch.run(&mut cpu), 0x2008);
}
#[test]
fn branches_return_the_next_pc() {
    let mut cpu = State::default();
    cpu.x[0] = 1;
    cpu.x[5] = 0xdead_0000;
    // b +8; bl +16; br x5; cbnz x0,+8 (taken) and with x0 == 0 (not taken).
    assert_eq!(
        compile(0x1000, &bytes(&[0x14000002]))
            .unwrap()
            .run(&mut cpu),
        0x1008
    );
    assert_eq!(
        compile(0x1000, &bytes(&[0x94000004]))
            .unwrap()
            .run(&mut cpu),
        0x1010
    );
    assert_eq!(cpu.x[30], 0x1004);
    assert_eq!(
        compile(0x1000, &bytes(&[0xd61f00a0]))
            .unwrap()
            .run(&mut cpu),
        0xdead_0000
    );
    let cbnz = compile(0x1000, &bytes(&[0xb5000040])).unwrap();
    assert_eq!(cbnz.run(&mut cpu), 0x1008);
    cpu.x[0] = 0;
    assert_eq!(cbnz.run(&mut cpu), 0x1004);
}
#[test]
fn the_environment_sees_every_instruction_and_every_exit() {
    // add x0,x0,#1 twice, then fall off the end: two instructions, one exit.
    let (_, env) = lift(0x1000, &bytes(&[0x91000400, 0x91000400])).unwrap();
    assert_eq!((env.begun, env.exits), (2, 1));
    // A conditional branch exits once, for both outcomes.
    let (_, env) = lift(0x1000, &bytes(&[0xb5000040, 0x91000400])).unwrap();
    assert_eq!((env.begun, env.exits), (1, 1));
}
#[test]
fn exclusives_and_system_instructions_are_the_environments_business() {
    // ldxr x1,[x0]; mrs x1,nzcv.
    for word in [0xc85f7c01u32, 0xd53b4201] {
        assert_eq!(
            lift(0, &bytes(&[word])).err(),
            Some(LiftError::Unsupported),
            "{word:08x}"
        );
    }
}
#[test]
fn plain_loads_and_stores_reach_the_environment_with_their_address() {
    let mut buf = [0u64; 8];
    let base = buf.as_mut_ptr() as u64;
    let mut cpu = State::default();
    cpu.x[0] = base;
    cpu.x[1] = 0x1122_3344_5566_7788;
    // str x1,[x0,#16]; ldr x3,[x0,#16]; ldr x4,[x0,x2,lsl #3] with x2 = 2.
    cpu.x[2] = 2;
    compile(0, &bytes(&[0xf9000801, 0xf9400803, 0xf8627804]))
        .unwrap()
        .run(&mut cpu);
    assert_eq!(buf[2], 0x1122_3344_5566_7788);
    assert_eq!(cpu.x[3], 0x1122_3344_5566_7788);
    assert_eq!(cpu.x[4], 0x1122_3344_5566_7788);
}
#[test]
fn narrow_loads_extend_as_the_instruction_says() {
    let mut buf = [0u64; 1];
    buf[0] = 0xffff_ffff_8765_f0f1;
    let mut cpu = State::default();
    for (word, want) in [
        (0x39400001u32, 0xf1u64),            // ldrb w1
        (0x39800001, 0xffff_ffff_ffff_fff1), // ldrsb x1
        (0xb9800001, 0xffff_ffff_8765_f0f1), // ldrsw x1
    ] {
        cpu.x[0] = buf.as_mut_ptr() as u64;
        compile(0, &bytes(&[word])).unwrap().run(&mut cpu);
        assert_eq!(cpu.x[1], want, "{word:08x}");
    }
    // strh w1,[x0,#2] stores only the low halfword.
    cpu.x[0] = buf.as_mut_ptr() as u64;
    cpu.x[1] = 0xaaaa_bbbb_cccc_dddd;
    compile(0, &bytes(&[0x79000401])).unwrap().run(&mut cpu);
    assert_eq!(buf[0], 0xffff_ffff_dddd_f0f1);
}
#[test]
fn pair_accesses_apply_the_writeback_after_the_access() {
    let mut buf = [0x11u64, 0x22, 0x33, 0x44];
    let mut cpu = State::default();
    let base = buf.as_mut_ptr() as u64;
    // ldp x1,x2,[x0],#16: loads from the old base, then advances it.
    cpu.x[0] = base;
    compile(0, &bytes(&[0xa8c10801])).unwrap().run(&mut cpu);
    assert_eq!((cpu.x[1], cpu.x[2], cpu.x[0]), (0x11, 0x22, base + 16));
    // ldp x1,x2,[x0,#16]!: the access uses the updated base.
    cpu.x[0] = base - 16;
    compile(0, &bytes(&[0xa9c10801])).unwrap().run(&mut cpu);
    assert_eq!((cpu.x[1], cpu.x[2], cpu.x[0]), (0x11, 0x22, base));
    // stp x1,x2,[x0]
    cpu.x[0] = base + 16;
    cpu.x[1] = 0xaa;
    cpu.x[2] = 0xbb;
    compile(0, &bytes(&[0xa9000801])).unwrap().run(&mut cpu);
    assert_eq!((buf[2], buf[3]), (0xaa, 0xbb));
}
#[test]
fn literal_loads_use_the_pc_relative_address() {
    let buf = [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210];
    let mut cpu = State::default();
    // ldr x1,.+8 at a pc that makes the target buf[1]: pc = &buf[0].
    let pc = buf.as_ptr() as u64;
    compile(pc, &bytes(&[0x58000041])).unwrap().run(&mut cpu);
    assert_eq!(cpu.x[1], 0xfedc_ba98_7654_3210);
}
#[test]
fn vector_loads_and_stores_move_whole_registers() {
    let mut buf = [0x1111_2222_3333_4444u64, 0x5555_6666_7777_8888, 0, 0, 0, 0];
    let mut cpu = State::default();
    cpu.x[0] = buf.as_mut_ptr() as u64;
    cpu.v[1] = [u64::MAX, u64::MAX];
    // ldr q1,[x0]; str q1,[x0,#16]; ldr d2,[x0,#8] zeroes the top half.
    cpu.v[2] = [u64::MAX, u64::MAX];
    compile(0, &bytes(&[0x3dc00001, 0x3d800401, 0xfd400402]))
        .unwrap()
        .run(&mut cpu);
    assert_eq!(cpu.v[1], [0x1111_2222_3333_4444, 0x5555_6666_7777_8888]);
    assert_eq!(
        (buf[2], buf[3]),
        (0x1111_2222_3333_4444, 0x5555_6666_7777_8888)
    );
    assert_eq!(cpu.v[2], [0x5555_6666_7777_8888, 0]);
    // ldp q1,q2,[x0] loads two registers: the original pair, and the copy stored above.
    cpu.v[1] = [0; 2];
    cpu.v[2] = [0; 2];
    compile(0, &bytes(&[0xad400801])).unwrap().run(&mut cpu);
    assert_eq!(cpu.v[1], [0x1111_2222_3333_4444, 0x5555_6666_7777_8888]);
    assert_eq!(cpu.v[2], [0x1111_2222_3333_4444, 0x5555_6666_7777_8888]);
}
#[test]
fn simd_pairwise_ops_match_a_reference_model() {
    // ADDP/SMAXP/SMINP/UMAXP/UMINP, every element size and both vector widths,
    // against random operands. Lane values are compared as the instruction
    // defines them: signed forms order the lanes as signed numbers.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // (base word, name): Q=0, U=0, size=0, Rm=0, Rn=0, Rd=0.
    #[derive(Clone, Copy, PartialEq)]
    enum Op {
        Add,
        SMax,
        SMin,
        UMax,
        UMin,
    }
    let ops = [Op::Add, Op::SMax, Op::SMin, Op::UMax, Op::UMin];
    for op in ops {
        for size in 0..4u32 {
            if size == 3 && op != Op::Add {
                continue;
            }
            for q in [false, true] {
                if size == 3 && !q {
                    continue; // ADDP .1D is reserved
                }
                let (opcode, u) = match op {
                    Op::Add => (0b10111, 0),
                    Op::SMax => (0b10100, 0),
                    Op::SMin => (0b10101, 0),
                    Op::UMax => (0b10100, 1),
                    Op::UMin => (0b10101, 1),
                };
                let word = 0x0e20_0400u32
                    | (q as u32) << 30
                    | u << 29
                    | size << 22
                    | 2 << 16
                    | opcode << 11
                    | 1 << 5;
                let block = compile(0, &bytes(&[word])).unwrap();
                let esize = 8usize << size;
                let lanes = (if q { 128 } else { 64 }) / esize;
                let mask = if esize == 64 {
                    u64::MAX
                } else {
                    (1u64 << esize) - 1
                };
                for _ in 0..200 {
                    let mut cpu = State::default();
                    cpu.v[1] = [next(), next()];
                    cpu.v[2] = [next(), next()];
                    cpu.v[0] = [next(), next()]; // must be fully overwritten
                    let lane = |v: [u64; 2], i: usize| -> u64 {
                        let bit = i * esize;
                        (v[bit / 64] >> (bit % 64)) & mask
                    };
                    let sext = |x: u64| -> i64 { ((x << (64 - esize)) as i64) >> (64 - esize) };
                    // Vm:Vn concatenated, `lanes` lanes each.
                    let all: Vec<u64> = (0..lanes)
                        .map(|i| lane(cpu.v[1], i))
                        .chain((0..lanes).map(|i| lane(cpu.v[2], i)))
                        .collect();
                    let mut want = [0u64; 2];
                    for i in 0..lanes {
                        let (a, b) = (all[2 * i], all[2 * i + 1]);
                        let r = match op {
                            Op::Add => a.wrapping_add(b) & mask,
                            Op::SMax => sext(a).max(sext(b)) as u64 & mask,
                            Op::SMin => sext(a).min(sext(b)) as u64 & mask,
                            Op::UMax => a.max(b),
                            Op::UMin => a.min(b),
                        };
                        let bit = i * esize;
                        want[bit / 64] |= r << (bit % 64);
                    }
                    let input = [cpu.v[1], cpu.v[2]];
                    block.run(&mut cpu);
                    assert_eq!(
                        cpu.v[0], want,
                        "{word:08x} size {size} q {q} inputs {input:x?}"
                    );
                }
            }
        }
    }
}
