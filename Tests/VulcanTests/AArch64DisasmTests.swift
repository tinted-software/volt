#if AArch64
//! Known-answer vectors for the AArch64 disassembler: hand-checked words
//! (verified against objdump in the Zig original) plus the listing shapes.
//! The host cannot execute A64, so these pin the decode text instead.
//!
//! Every multi-register word below is computed from the encoder formulas in
//! `libs/vulcan-target/aarch64/encode.zig` (base | rm<<16 | rn<<5 | rd), not
//! from memory.

import Testing
import VoltTarget

@Suite("AArch64 disassembler")
struct AArch64DisasmTests {
  func expectOne(_ word: UInt32, _ expected: String) {
    #expect(AArch64Disassembler.one(word: word) == expected)
  }

  @Test("integer data processing")
  func integerDataProcessing() {
    expectOne(0xD65F_03C0, "ret")
    expectOne(0x0B02_0020, "add w0, w1, w2")
    expectOne(0x4B05_0083, "sub w3, w4, w5")
    expectOne(0x8B02_0020, "add x0, x1, x2")
    expectOne(0x1B02_8C20, "msub w0, w1, w2, w3")
    expectOne(0x1AC9_0D07, "sdiv w7, w8, w9")
    expectOne(0x1AC3_2041, "lsl w1, w2, w3")
    expectOne(0x2A03_0041, "orr w1, w2, w3")
    expectOne(0xAA06_03E5, "mov x5, x6")
    expectOne(0x72A0_00E0, "movk w0, #7, lsl #16")
    expectOne(0xD100_43FF, "sub sp, sp, #16")
    expectOne(0x9140_0820, "add x0, x1, #2, lsl #12")
    expectOne(0x5280_0540, "mov w0, #0x2a")
    expectOne(0xD2C0_0121, "mov x1, #0x900000000")
  }

  @Test("flags, selects, and branches")
  func flagsSelectsBranches() {
    expectOne(0x6B02_003F, "cmp w1, w2")
    expectOne(0x1A9F_A7E0, "cset w0, lt")
    expectOne(0x1A82_C020, "csel w0, w1, w2, gt")
    expectOne(0xD63F_0100, "blr x8")
    expectOne(0x1400_0004, "b .+16")
    expectOne(0x97FF_FFFE, "bl .-8")
    expectOne(0x3500_0063, "cbnz w3, .+12")
    expectOne(0x3400_0040, "cbz w0, .+8")
    expectOne(0xD400_0001, "svc #0")
  }

  @Test("memory access")
  func memoryAccess() {
    expectOne(0xA9BF_7BFD, "stp x29, x30, [sp, #-16]!")
    expectOne(0xA8C1_7BFD, "ldp x29, x30, [sp], #16")
    expectOne(0xF900_0420, "str x0, [x1, #8]")
    expectOne(0xF940_0020, "ldr x0, [x1]")
    expectOne(0xB900_0420, "str w0, [x1, #4]")
    expectOne(0x3940_0062, "ldrb w2, [x3]")
    expectOne(0x79C0_0062, "ldrsh w2, [x3]")
    expectOne(0x3DC0_0020, "ldr q0, [x1]")
  }

  @Test("system-binary vectors verified against objdump")
  func objdumpVectors() {
    expectOne(0xEB09_0108, "subs x8, x8, x9")
    expectOne(0x6B09_0108, "subs w8, w8, w9")
    expectOne(0xAB09_0108, "adds x8, x8, x9")
    expectOne(0x1200_3C08, "and w8, w0, #0xffff")
    expectOne(0x1200_0108, "and w8, w8, #0x1")
    expectOne(0x3600_0088, "tbz w8, #0, .+16")
    expectOne(0x3400_00E8, "cbz w8, .+28")
    expectOne(0x5400_00C3, "b.cc .+24")
    expectOne(0xB100_0508, "adds x8, x8, #1")
    expectOne(0xF100_0508, "subs x8, x8, #1")
    expectOne(0x93CC_F58C, "ror x12, x12, #61")
    expectOne(0xF85F_83A8, "ldur x8, [x29, #-8]")
    expectOne(0xF81F_03A8, "stur x8, [x29, #-16]")
    expectOne(0x785F_83A8, "ldurh w8, [x29, #-8]")
    expectOne(0xA941_7BFD, "ldp x29, x30, [sp, #16]")
    expectOne(0xA902_7BFD, "stp x29, x30, [sp, #32]")
  }

  @Test("bitfield aliases, movn, mulh, SIMD unscaled")
  func bitfieldMovnMulh() {
    expectOne(0x5300_3D08, "uxth w8, w8")
    expectOne(0x5300_1D08, "uxtb w8, w8")
    expectOne(0x9340_7C00, "sxtw x0, w0")
    expectOne(0x1300_1D08, "sxtb w8, w8")
    expectOne(0x1303_7D08, "asr w8, w8, #3")
    expectOne(0x5304_7D08, "lsr w8, w8, #4")
    expectOne(0x531F_7908, "lsl w8, w8, #1")
    expectOne(0xD340_1128, "ubfx x8, x9, #0, #5")
    expectOne(0x3300_0828, "bfxil w8, w1, #0, #3")
    expectOne(0x9280_0AA1, "mov x1, #0xffffffffffffffaa")
    expectOne(0x1280_0AA1, "mov w1, #0xffffffaa")
    expectOne(0x9BCA_7D28, "umulh x8, x9, x10")
    expectOne(0x9B4A_7D28, "smulh x8, x9, x10")
    expectOne(0x3CDF_03A0, "ldur q0, [x29, #-16]")
    expectOne(0xFC5F_03A0, "ldur d0, [x29, #-16]")
    expectOne(0xBC5F_03A0, "ldur s0, [x29, #-16]")
  }

  @Test("scalar floating point")
  func scalarFloat() {
    expectOne(0x1E22_2820, "fadd s0, s1, s2")
    expectOne(0x1E62_1820, "fdiv d0, d1, d2")
    expectOne(0x1E21_C020, "fsqrt s0, s1")
    expectOne(0x1E27_0020, "fmov s0, w1")
    expectOne(0x9E66_0020, "fmov x0, d1")
    expectOne(0x1E22_0020, "scvtf s0, w1")
    expectOne(0x1E38_0020, "fcvtzs w0, s1")
    expectOne(0x1E22_C020, "fcvt d0, s1")
  }

  @Test("vector floating point and lane moves")
  func vectorFloat() {
    expectOne(0x4E22_D420, "fadd v0.4s, v1.4s, v2.4s")
    expectOne(0x6E22_DC20, "fmul v0.4s, v1.4s, v2.4s")
    expectOne(0x4E22_CC20, "fmla v0.4s, v1.4s, v2.4s")
    expectOne(0x4EA2_CC20, "fmls v0.4s, v1.4s, v2.4s")
    expectOne(0x6EA0_F820, "fneg v0.4s, v1.4s")
    expectOne(0x6E20_5820, "mvn v0.16b, v1.16b")
    expectOne(0x6E62_1C20, "bsl v0.16b, v1.16b, v2.16b")
    expectOne(0x4E04_0C20, "dup v0.4s, w1")
    expectOne(0x5E14_0420, "dup s0, v1.s[2]")
    expectOne(0x6E0C_0420, "ins v0.s[1], v1.s[0]")
    expectOne(0x0E14_3C20, "umov w0, v1.s[2]")
    expectOne(0x4E0C_1C20, "ins v0.s[1], w1")
    expectOne(0x4EB1_B820, "addv s0, v1.4s")
    expectOne(0x6E22_D420, "faddp v0.4s, v1.4s, v2.4s")
    expectOne(0x7E30_D820, "faddp s0, v1.2s")
    expectOne(0x4EA1_1C20, "mov v0.16b, v1.16b")
    expectOne(0x1F02_0C20, "fmadd s0, s1, s2, s3")
    expectOne(0x1F42_0C20, "fmadd d0, d1, d2, d3")
    expectOne(0x1F02_8C20, "fmsub s0, s1, s2, s3")
    expectOne(0x1F62_8C20, "fnmsub d0, d1, d2, d3")
  }

  @Test("unknown words render as .word")
  func unknownWord() {
    #expect(AArch64Disassembler.one(word: 0xFFFF_FFFF) == ".word 0xffffffff")
  }

  @Test("formatModule annotates branch targets")
  func formatModuleAnnotations() {
    // f at word 0 branches to word 2 (f+0x8); word 2 branches to g+0x4.
    let words: [UInt32] = [0x1400_0002, 0xD65F_03C0, 0x1400_0002, 0xD65F_03C0, 0xD65F_03C0]
    var bytes: [UInt8] = []
    for w in words {
      bytes += [
        UInt8(w & 0xFF), UInt8((w >> 8) & 0xFF), UInt8((w >> 16) & 0xFF), UInt8((w >> 24) & 0xFF),
      ]
    }
    let syms = [FuncSymbol(name: "f", offset: 0), FuncSymbol(name: "g", offset: 12)]
    let text = AArch64Disassembler.formatModule(code: bytes, symbols: syms)
    #expect(text.contains("b .+8  <f+0x8>"))
    #expect(text.contains("b .+8  <g+0x4>"))
  }

  @Test("formatWithLines interleaves source markers")
  func formatWithLines() {
    let words: [UInt32] = [0x0B02_0020, 0x1B01_7C00, 0xD65F_03C0]
    let lines = [LineEntry(offset: 0, line: 2), LineEntry(offset: 4, line: 3)]
    let text = AArch64Disassembler.formatWordsWithLines(words, lines: lines)
    #expect(
      text
        == "; line 2\n0000: 0b020020  add w0, w1, w2\n; line 3\n0004: 1b017c00  mul w0, w0, w1\n0008: d65f03c0  ret\n"
    )
  }
}

#endif
