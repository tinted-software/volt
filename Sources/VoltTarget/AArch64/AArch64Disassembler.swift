#if AArch64
//! AArch64 (A64) disassembler: decodes the instruction subset Volt's encoder
//! emits into a readable listing, so codegen debugging does not mean
//! hand-decoding hex. Field positions mirror `encode.zig` bit-for-bit.
//! Anything outside the decoded subset prints as `.word 0x<hex>` so nothing
//! is silently mis-decoded.
//!
//! `disassembleOne` renders a single 32-bit word; `format` renders a whole
//! byte buffer with byte addresses; `formatModule` adds `name:` labels and
//! `<sym>` / `<sym+off>` call annotations. Validated against the same
//! known-answer vectors as the Zig original (see `AArch64DisasmTests`).

import VoltIR

// MARK: - Field extraction

@inline(__always) func rd(_ w: UInt32) -> UInt32 { w & 0x1F }
@inline(__always) func rn(_ w: UInt32) -> UInt32 { (w >> 5) & 0x1F }
@inline(__always) func rm(_ w: UInt32) -> UInt32 { (w >> 16) & 0x1F }
@inline(__always) func ra(_ w: UInt32) -> UInt32 { (w >> 10) & 0x1F }
@inline(__always) func sf(_ w: UInt32) -> Bool { (w >> 31) & 1 != 0 }

/// A 32/64-bit general register (`w`/`x`), rendering 31 as the zero register.
func gp(_ x64: Bool, _ r: UInt32) -> String {
  let c: Character = x64 ? "x" : "w"
  return r == 31 ? "\(c)zr" : "\(c)\(r)"
}

/// A general register in an address/stack context, rendering 31 as `sp`.
func sp(_ x64: Bool, _ r: UInt32) -> String {
  r == 31 ? "sp" : gp(x64, r)
}

/// binutils spelling (cs/cc rather than the hs/lo synonyms) so listings match objdump.
func condName(_ v: UInt32) -> String {
  switch v & 0xF {
  case 0: return "eq"
  case 1: return "ne"
  case 2: return "cs"
  case 3: return "cc"
  case 4: return "mi"
  case 5: return "pl"
  case 6: return "vs"
  case 7: return "vc"
  case 8: return "hi"
  case 9: return "ls"
  case 10: return "ge"
  case 11: return "lt"
  case 12: return "gt"
  case 13: return "le"
  case 14: return "al"
  default: return "nv"
  }
}

func hex(_ v: UInt64, width: Int) -> String {
  var s = String(v, radix: 16)
  while s.count < width { s = "0" + s }
  return s
}

func relText(_ relBytes: Int64) -> String {
  relBytes < 0 ? ".-\(UInt64(bitPattern: -relBytes))" : ".+\(UInt64(bitPattern: relBytes))"
}

// MARK: - Tables

/// Masks for the three-register data-processing form: clear rd, rn, rm; keep the opcode
/// and the op2/shift field in bits[10:15]. The `sf` bit is dropped so one entry covers both
/// the w and x forms.
let rrrMask: UInt32 = 0x7FE0_FC00

/// `rrrMask` with the imm6 shift-amount field (bits[15:10]) dropped, for the add/sub
/// shifted-register form where that field is a real operand and not part of the opcode.
let rrrNoImm6Mask: UInt32 = rrrMask & ~UInt32(0xFC00)

/// The 32-bit-base opcodes of the shared three-register form (sf dropped).
let rrrTable: [(match: UInt32, mnem: String)] = [
  (0x0B00_0000, "add"),
  (0x4B00_0000, "sub"),
  (0x0A00_0000, "and"),
  (0x6A00_0000, "ands"),  // flag-setting and (shift 0), the arith-branch fold's `ands`
  (0x2A00_0000, "orr"),
  (0x4A00_0000, "eor"),
  (0x1AC0_0C00, "sdiv"),
  (0x1AC0_0800, "udiv"),
  (0x1AC0_2000, "lsl"),
  (0x1AC0_2400, "lsr"),
  (0x1AC0_2800, "asr"),
]

struct MemImmEntry {
  var match: UInt32
  var mnem: String
  var x64: Bool
  var scale: UInt32
  var fp: Bool
  var dbl: Bool
}

/// Scaled-unsigned-offset loads/stores (integer and FP).
let memImmTable: [MemImmEntry] = [
  .init(match: 0xF900_0000, mnem: "str", x64: true, scale: 3, fp: false, dbl: false),
  .init(match: 0xF940_0000, mnem: "ldr", x64: true, scale: 3, fp: false, dbl: false),
  .init(match: 0xB900_0000, mnem: "str", x64: false, scale: 2, fp: false, dbl: false),
  .init(match: 0xB940_0000, mnem: "ldr", x64: false, scale: 2, fp: false, dbl: false),
  .init(match: 0x3900_0000, mnem: "strb", x64: false, scale: 0, fp: false, dbl: false),
  .init(match: 0x3940_0000, mnem: "ldrb", x64: false, scale: 0, fp: false, dbl: false),
  .init(match: 0x39C0_0000, mnem: "ldrsb", x64: false, scale: 0, fp: false, dbl: false),
  .init(match: 0x7900_0000, mnem: "strh", x64: false, scale: 1, fp: false, dbl: false),
  .init(match: 0x7940_0000, mnem: "ldrh", x64: false, scale: 1, fp: false, dbl: false),
  .init(match: 0x79C0_0000, mnem: "ldrsh", x64: false, scale: 1, fp: false, dbl: false),
  .init(match: 0xBD00_0000, mnem: "str", x64: false, scale: 2, fp: true, dbl: false),
  .init(match: 0xBD40_0000, mnem: "ldr", x64: false, scale: 2, fp: true, dbl: false),
  .init(match: 0xFD00_0000, mnem: "str", x64: false, scale: 3, fp: true, dbl: true),
  .init(match: 0xFD40_0000, mnem: "ldr", x64: false, scale: 3, fp: true, dbl: true),
  .init(match: 0x3D80_0000, mnem: "str", x64: false, scale: 4, fp: true, dbl: false),  // Q
  .init(match: 0x3DC0_0000, mnem: "ldr", x64: false, scale: 4, fp: true, dbl: false),  // Q
  .init(match: 0x7D00_0000, mnem: "str", x64: false, scale: 1, fp: true, dbl: false),  // H (f16)
  .init(match: 0x7D40_0000, mnem: "ldr", x64: false, scale: 1, fp: true, dbl: false),  // H (f16)
]

// MARK: - Bitmask immediate

/// Decode a logical-immediate `(N:immr:imms)` to its 32/64-bit value, or nil if the encoding is
/// reserved. This is ARM's `DecodeBitMasks` (immediate result only): a run of `s+1` ones,
/// rotated right by `r` within an `esize`-bit element, replicated across the register.
func decodeBitmask(x64: Bool, n: UInt32, imms: UInt32, immr: UInt32) -> UInt64? {
  let width: UInt32 = x64 ? 64 : 32
  // len = position of the highest set bit of (N : NOT(imms)), a 7-bit quantity.
  let combined: UInt32 = (n << 6) | (~imms & 0x3F)
  if combined == 0 { return nil }
  var len: UInt32 = 6
  while len > 0, (combined & (UInt32(1) << len)) == 0 { len -= 1 }
  if len == 0 { return nil }  // element size 1 is not a valid logical immediate
  let esize: UInt32 = UInt32(1) << len
  if esize > width { return nil }  // N == 1 is only valid for the 64-bit form
  let levels: UInt32 = esize - 1
  let s: UInt32 = imms & levels
  let r: UInt32 = immr & levels
  if s == levels { return nil }  // an all-ones element is reserved
  let elem: UInt64 = (UInt64(1) << (s + 1)) - 1
  let rotated = rorInEsize(elem, r, esize)
  var result: UInt64 = 0
  var pos: UInt32 = 0
  while pos < width {
    result |= rotated << pos
    pos += esize
  }
  return x64 ? result : result & 0xFFFF_FFFF
}

/// Rotate the low `esize` bits of `v` right by `r`.
func rorInEsize(_ v: UInt64, _ r: UInt32, _ esize: UInt32) -> UInt64 {
  let mask: UInt64 = esize == 64 ? ~UInt64(0) : (UInt64(1) << esize) - 1
  let rr = r % esize
  if rr == 0 { return v & mask }
  return ((v >> rr) | (v << (esize - rr))) & mask
}

// MARK: - Bitfield aliases

/// Render a bitfield-move (SBFM/UBFM/BFM) using the alias a debugger expects (uxt*/sxt*/lsl/lsr/
/// asr/ubfx/sbfx/bfi/bfxil). `opc` selects the family; `immr`/`imms` drive the alias choice.
func bitfield(_ w: UInt32) -> String {
  let opc = (w >> 29) & 3
  let x64 = sf(w)
  let width: UInt32 = x64 ? 64 : 32
  let immr: UInt32 = (w >> 16) & 0x3F
  let imms: UInt32 = (w >> 10) & 0x3F
  func two(_ mnem: String, _ d: UInt32, _ n: UInt32) -> String {
    "\(mnem) \(gp(x64, d)), \(gp(x64, n))"
  }
  switch opc {
  case 2:  // UBFM
    if imms + 1 == immr, imms != width - 1 {  // lsl #(width-1-imms)
      return "\(two("lsl", rd(w), rn(w))), #\(width - 1 - imms)"
    }
    if imms == width - 1 {  // lsr #immr
      return "\(two("lsr", rd(w), rn(w))), #\(immr)"
    }
    if immr == 0, imms == 7, !x64 { return two("uxtb", rd(w), rn(w)) }
    if immr == 0, imms == 15, !x64 { return two("uxth", rd(w), rn(w)) }
    if imms < immr {  // ubfiz Rd, Rn, #(width-immr), #(imms+1)
      return "\(two("ubfiz", rd(w), rn(w))), #\(width - immr), #\(imms + 1)"
    }
    return "\(two("ubfx", rd(w), rn(w))), #\(immr), #\(imms - immr + 1)"
  case 0:  // SBFM
    if imms == width - 1 {  // asr #immr
      return "\(two("asr", rd(w), rn(w))), #\(immr)"
    }
    if immr == 0, imms == 7 { return two("sxtb", rd(w), rn(w)) }
    if immr == 0, imms == 15 { return two("sxth", rd(w), rn(w)) }
    if immr == 0, imms == 31, x64 {  // sxtw Xd, Wn
      return "sxtw \(gp(true, rd(w))), \(gp(false, rn(w)))"
    }
    if imms < immr {  // sbfiz
      return "\(two("sbfiz", rd(w), rn(w))), #\(width - immr), #\(imms + 1)"
    }
    return "\(two("sbfx", rd(w), rn(w))), #\(immr), #\(imms - immr + 1)"
  default:  // BFM
    if imms < immr {  // bfi Rd, Rn, #(width-immr), #(imms+1)
      return "\(two("bfi", rd(w), rn(w))), #\(width - immr), #\(imms + 1)"
    }
    return "\(two("bfxil", rd(w), rn(w))), #\(immr), #\(imms - immr + 1)"  // bfxil
  }
}

// MARK: - FP / SIMD helpers

/// Scalar floating-point instructions. Returns the text when matched.
func fpDecode(_ w: UInt32) -> String? {
  let dbl = (w >> 22) & 1 != 0
  let fc: Character = dbl ? "d" : "s"
  // Three-register scalar FP arithmetic.
  let fp3: [(match: UInt32, mnem: String)] = [
    (0x1E20_2800, "fadd"),
    (0x1E20_3800, "fsub"),
    (0x1E20_0800, "fmul"),
    (0x1E20_1800, "fdiv"),
  ]
  for e in fp3 where w & 0xFFA0_FC00 == e.match {
    return "\(e.mnem) \(fc)\(rd(w)), \(fc)\(rn(w)), \(fc)\(rm(w))"
  }
  // Floating-point data-processing (3-source): Rd = op(Rn*Rm, Ra). Mask excludes ftype
  // (bit 22) and the four 5-bit register fields.
  let fp3a: [(match: UInt32, mnem: String)] = [
    (0x1F00_0000, "fmadd"),
    (0x1F00_8000, "fmsub"),
    (0x1F20_0000, "fnmadd"),
    (0x1F20_8000, "fnmsub"),
  ]
  for e in fp3a where w & 0xFF20_8000 == e.match {
    return "\(e.mnem) \(fc)\(rd(w)), \(fc)\(rn(w)), \(fc)\(rm(w)), \(fc)\(ra(w))"
  }
  if w & 0xFFA0_0C00 == 0x1E20_0C00 {  // fcsel (mask excludes the cond field [12:15])
    return "fcsel \(fc)\(rd(w)), \(fc)\(rn(w)), \(fc)\(rm(w)), \(condName((w >> 12) & 0xF))"
  }
  if w & 0xFFA0_FC1F == 0x1E20_2000 {  // fcmp
    return "fcmp \(fc)\(rn(w)), \(fc)\(rm(w))"
  }
  if w & 0xFFFF_FC00 == 0x1E60_4000 {  // fmov (reg, 64-bit view)
    return "fmov d\(rd(w)), d\(rn(w))"
  }
  // fmov between a general register and an FP register (both directions, s/d).
  let fmov: [(match: UInt32, toGpr: Bool, x64: Bool)] = [
    (0x1E27_0000, false, false),
    (0x9E67_0000, false, true),
    (0x1E26_0000, true, false),
    (0x9E66_0000, true, true),
  ]
  for e in fmov where w & 0xFFFF_FC00 == e.match {
    let f: Character = e.x64 ? "d" : "s"
    if e.toGpr {
      return "fmov \(gp(e.x64, rd(w))), \(f)\(rn(w))"
    } else {
      return "fmov \(f)\(rd(w)), \(gp(e.x64, rn(w)))"
    }
  }
  // Two-register scalar FP ops (sqrt, rounding, conversions).
  let fp2: [(match: UInt32, mnem: String)] = [
    (0x1E21_C000, "fsqrt"),
    (0x1E24_4000, "frintn"),
    (0x1E24_C000, "frintp"),
    (0x1E25_4000, "frintm"),
    (0x1E25_C000, "frintz"),
  ]
  for e in fp2 where w & 0xFFBF_FC00 == e.match {
    return "\(e.mnem) \(fc)\(rd(w)), \(fc)\(rn(w))"
  }
  if w & 0xFFBF_FC00 == 0x1E22_0000 { return "scvtf \(fc)\(rd(w)), \(gp(false, rn(w)))" }  // scvtf sd, wn
  if w & 0xFFBF_FC00 == 0x1E23_0000 { return "ucvtf \(fc)\(rd(w)), \(gp(false, rn(w)))" }  // ucvtf
  if w & 0xFFBF_FC00 == 0x1E38_0000 { return "fcvtzs \(gp(false, rd(w))), \(fc)\(rn(w))" }  // fcvtzs wd, sn
  if w & 0xFFBF_FC00 == 0x1E39_0000 { return "fcvtzu \(gp(false, rd(w))), \(fc)\(rn(w))" }  // fcvtzu
  if w & 0xFFFF_FC00 == 0x1E22_C000 { return "fcvt d\(rd(w)), s\(rn(w))" }  // fcvt d, s (s->d)
  if w & 0xFFFF_FC00 == 0x1E62_4000 { return "fcvt s\(rd(w)), d\(rn(w))" }  // fcvt s, d (d->s)
  if w & 0xFFFF_FC00 == 0x1EE2_4000 { return "fcvt s\(rd(w)), h\(rn(w))" }  // fcvt s, h (h->s, f16 widen)
  if w & 0xFFFF_FC00 == 0x1E23_C000 { return "fcvt h\(rd(w)), s\(rn(w))" }  // fcvt h, s (s->h, f16 narrow)
  if w & 0xFFFF_FC00 == 0x1E63_C000 { return "fcvt h\(rd(w)), d\(rn(w))" }  // fcvt h, d (d->h, f16 narrow)
  return nil
}

/// NEON (128-bit vector) instructions. Returns the text when matched.
func vecDecode(_ w: UInt32) -> String? {
  // Three-register 4S / 16B forms.
  let v3: [(match: UInt32, mnem: String, b16: Bool)] = [
    (0x4E20_D400, "fadd", false),
    (0x4EA0_D400, "fsub", false),
    (0x6E20_DC00, "fmul", false),
    (0x6E20_FC00, "fdiv", false),
    (0x4EA0_F400, "fmin", false),
    (0x4E20_F400, "fmax", false),
    (0x4E20_E400, "fcmeq", false),
    (0x6EA0_E400, "fcmgt", false),
    (0x6E20_E400, "fcmge", false),
    (0x6E60_1C00, "bsl", true),
    (0x4E20_CC00, "fmla", false),
    (0x4EA0_CC00, "fmls", false),
    (0x6E20_D400, "faddp", false),
  ]
  for e in v3 {
    if w == e.match | (rm(w) << 16) | (rn(w) << 5) | rd(w) {
      let s = e.b16 ? "16b" : "4s"
      return "\(e.mnem) v\(rd(w)).\(s), v\(rn(w)).\(s), v\(rm(w)).\(s)"
    }
  }
  // Two-register vector forms.
  if w & 0xFFFF_FC00 == 0x6EA0_F800 { return "fneg v\(rd(w)).4s, v\(rn(w)).4s" }
  if w & 0xFFFF_FC00 == 0x6EA1_F800 { return "fsqrt v\(rd(w)).4s, v\(rn(w)).4s" }
  if w & 0xFFFF_FC00 == 0x6E20_5800 { return "mvn v\(rd(w)).16b, v\(rn(w)).16b" }
  // Horizontal reduce: addv sums the 4S lanes to a scalar S register; faddpScalar finishes a
  // pairwise float reduction from a 2S pair. Both cross from the Vn.4S/2S view into Sd.
  if w & 0xFFFF_FC00 == 0x4EB1_B800 { return "addv s\(rd(w)), v\(rn(w)).4s" }
  if w & 0xFFFF_FC00 == 0x7E30_D800 { return "faddp s\(rd(w)), v\(rn(w)).2s" }
  // mov vd.16b, vn.16b is orr with rm == rn.
  if w & 0xFFE0_FC00 == 0x4EA0_1C00, rm(w) == rn(w) {
    return "mov v\(rd(w)).16b, v\(rn(w)).16b"
  }
  if w & 0xFFFF_FC00 == 0x4E04_0C00 { return "dup v\(rd(w)).4s, \(gp(false, rn(w)))" }  // dup Vd.4S, Wn
  // dup Vd.4S, Vn.s[i] / dup Sd, Vn.s[i] / ins Vd.s[i], Vn.s[0]: imm5 in bits[16:20].
  let imm5 = (w >> 16) & 0x1F
  let lane = (imm5 >> 3) & 3
  if w & 0xFFE0_FC00 == 0x4E00_0400 { return "dup v\(rd(w)).4s, v\(rn(w)).s[\(lane)]" }
  if w & 0xFFE0_FC00 == 0x5E00_0400 { return "dup s\(rd(w)), v\(rn(w)).s[\(lane)]" }
  if w & 0xFFE0_8C00 == 0x6E00_0400 { return "ins v\(rd(w)).s[\(lane)], v\(rn(w)).s[0]" }
  if w & 0xFFE0_FC00 == 0x0E00_3C00 { return "umov \(gp(false, rd(w))), v\(rn(w)).s[\(lane)]" }  // umov Wd, Vn.s[i]
  if w & 0xFFE0_FC00 == 0x4E00_1C00 { return "ins v\(rd(w)).s[\(lane)], \(gp(false, rn(w)))" }  // ins Vd.s[i], Wn
  return nil
}

/// Scaled-unsigned-offset loads/stores (integer and FP). Returns the text when matched.
func memImm(_ w: UInt32) -> String? {
  for m in memImmTable {
    if w & 0xFFC0_0000 != m.match { continue }
    let off: UInt32 = ((w >> 10) & 0xFFF) << m.scale
    var s = "\(m.mnem) "
    if m.fp {
      // The FP view: Q (128-bit), D (64-bit), H (16-bit, the f16 boundary form), else S.
      let c: Character = m.scale == 4 ? "q" : m.dbl ? "d" : m.scale == 1 ? "h" : "s"
      s += "\(c)\(rd(w))"
    } else {
      s += gp(m.x64, rd(w))
    }
    s += ", [\(sp(true, rn(w)))"
    s += off != 0 ? ", #\(off)]" : "]"
    return s
  }
  return nil
}

// MARK: - Core decoder

/// Decode one word to `mnemonic operands`. Mirrors the Zig `decode` order exactly:
/// exact/special encodings, aliases, the shared three-register table, immediates,
/// branches, pairs, logical-immediates, unscaled and scaled memory, FP, SIMD.
func decodeWord(_ w: UInt32) -> String {
  // Exact, operandless / special encodings first.
  if w == 0xD65F_03C0 { return "ret" }  // encode.ret()
  if w & 0xFFE0_001F == 0xD400_0001 { return "svc #\((w >> 5) & 0xFFFF)" }
  if w & 0xFFFF_FC1F == 0xD63F_0000 { return "blr \(sp(true, rn(w)))" }

  // `mov` (register) is `orr Xd, xzr, Xm` with rn == 31; render it as a move.
  if w & 0x7FE0_FFE0 == 0x2A00_03E0 {
    return "mov \(gp(sf(w), rd(w))), \(gp(sf(w), rm(w)))"
  }

  // neg Rd, Rm = sub Rd, RZR, Rm (rn == 31), the objdump alias.
  if w & 0x7FE0_FC00 == 0x4B00_0000, rn(w) == 31 {
    return "neg \(gp(sf(w), rd(w))), \(gp(sf(w), rm(w)))"
  }

  // The shared three-register form.
  for e in rrrTable where w & rrrMask == e.match {
    return "\(e.mnem) \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w)))"
  }

  // add/sub, shifted-register form with a nonzero LSL shift amount (imm6 at bits[15:10]).
  if w & rrrNoImm6Mask == 0x0B00_0000 || w & rrrNoImm6Mask == 0x4B00_0000 {
    let mnem = w & rrrNoImm6Mask == 0x4B00_0000 ? "sub" : "add"
    return
      "\(mnem) \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w))), lsl #\((w >> 10) & 0x3F)"
  }

  // madd/msub (mul is madd with ra == 31).
  if w & 0x7FE0_8000 == 0x1B00_0000 || w & 0x7FE0_8000 == 0x1B00_8000 {
    let isSub = (w >> 15) & 1 != 0
    if !isSub, ra(w) == 31 {
      return "mul \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w)))"
    }
    if isSub, ra(w) == 31 {  // mneg Rd, Rn, Rm = msub Rd, Rn, Rm, RZR
      return "mneg \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w)))"
    }
    let mnem = isSub ? "msub" : "madd"
    return
      "\(mnem) \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w))), \(gp(sf(w), ra(w)))"
  }

  // add/sub immediate (bit30 = op is sub, bit29 = S sets flags, bit22 = shift-by-12). When S
  // is set and rd == 31 the alias is cmp / cmn. Rn/Rd 31 otherwise mean sp.
  if w & 0x1F80_0000 == 0x1100_0000 {
    let isSub = (w >> 30) & 1 != 0
    let setFlags = (w >> 29) & 1 != 0
    let shift12 = (w >> 22) & 1 != 0
    let imm: UInt32 = (w >> 10) & 0xFFF
    if setFlags, rd(w) == 31 {
      var s = "\(isSub ? "cmp" : "cmn") \(sp(sf(w), rn(w))), #\(imm)"
      if shift12 { s += ", lsl #12" }
      return s
    }
    // mov Rd, Rn (to/from SP) = add Rd, Rn, #0 with no shift.
    if !setFlags, !isSub, imm == 0, !shift12, rd(w) == 31 || rn(w) == 31 {
      return "mov \(sp(sf(w), rd(w))), \(sp(sf(w), rn(w)))"
    }
    let mnem = setFlags ? (isSub ? "subs" : "adds") : (isSub ? "sub" : "add")
    var s = "\(mnem) \(sp(sf(w), rd(w))), \(sp(sf(w), rn(w))), #\(imm)"
    if shift12 { s += ", lsl #12" }
    return s
  }

  // movn (move wide negated): rendered as `mov Rd, #<~imm>`, matching objdump's alias.
  if w & 0x7F80_0000 == 0x1280_0000 {
    let imm: UInt64 = UInt64((w >> 5) & 0xFFFF)
    let shift = ((w >> 21) & 3) * 16
    let val = ~(imm << shift)
    let shown = sf(w) ? val : val & 0xFFFF_FFFF
    return "mov \(gp(sf(w), rd(w))), #0x\(hex(shown, width: 0))"
  }

  // movz / movk (32/64-bit, with an optional lsl of the 16-bit immediate). objdump aliases
  // movz to `mov Rd, #<imm<<shift>>`, so render the assembled value for movz.
  if w & 0x7F80_0000 == 0x5280_0000 || w & 0x7F80_0000 == 0x7280_0000 {
    let keep = (w >> 29) & 1 != 0  // opc 11 -> movk, 10 -> movz
    let imm: UInt64 = UInt64((w >> 5) & 0xFFFF)
    let shift = ((w >> 21) & 3) * 16
    if !keep {  // movz -> mov Rd, #value
      let val = imm << shift
      let shown = sf(w) ? val : val & 0xFFFF_FFFF
      return "mov \(gp(sf(w), rd(w))), #0x\(hex(shown, width: 0))"
    }
    var s = "movk \(gp(sf(w), rd(w))), #\(imm)"
    if shift != 0 { s += ", lsl #\(shift)" }
    return s
  }

  // adds/subs (shifted register, shift 0), with cmn/cmp aliases when rd == 31.
  if w & 0x7FE0_FC00 == 0x2B00_0000 || w & 0x7FE0_FC00 == 0x6B00_0000 {
    let isSub = (w >> 30) & 1 != 0
    if rd(w) == 31 {
      return "\(isSub ? "cmp" : "cmn") \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w)))"
    }
    if isSub, rn(w) == 31 {  // negs Rd, Rm = subs Rd, RZR, Rm
      return "negs \(gp(sf(w), rd(w))), \(gp(sf(w), rm(w)))"
    }
    return
      "\(isSub ? "subs" : "adds") \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w)))"
  }

  // cset (csinc Rd, zr, zr, invert(cond)). The mask excludes the cond field [12:15].
  if w & 0x7FFF_0FE0 == 0x1A9F_07E0 {
    return "cset \(gp(sf(w), rd(w))), \(condName(((w >> 12) & 0xF) ^ 1))"
  }

  // csel Rd, Rn, Rm, cond.
  if w & 0x7FE0_0C00 == 0x1A80_0000 {
    return
      "csel \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w))), \(condName((w >> 12) & 0xF))"
  }

  // cbz / cbnz Rt, label (bit24 selects nz).
  if w & 0x7E00_0000 == 0x3400_0000 {
    let nz = (w >> 24) & 1 != 0
    let imm19 = Int64(Int32(bitPattern: w &<< 8) >> 13) * 4  // sign-extend bits[5:23]
    return "\(nz ? "cbnz" : "cbz") \(gp(sf(w), rd(w))), \(relText(imm19))"
  }

  // tbz / tbnz Rt, #bit, label (bit24 selects nz). Test-bit index is b5:b40.
  if w & 0x7E00_0000 == 0x3600_0000 {
    let nz = (w >> 24) & 1 != 0
    let bit: UInt32 = (((w >> 31) & 1) << 5) | ((w >> 19) & 0x1F)
    let imm14 = Int64(Int32(bitPattern: w &<< 13) >> 18) * 4  // bits[18:5], scaled
    return "\(nz ? "tbnz" : "tbz") \(gp((w >> 31) & 1 != 0, rd(w))), #\(bit), \(relText(imm14))"
  }

  // b.cond label (bit4 == 0 distinguishes it from the consistent-conditional-branch forms).
  if w & 0xFF00_0010 == 0x5400_0000 {
    let imm19 = Int64(Int32(bitPattern: w &<< 8) >> 13) * 4
    return "b.\(condName(w & 0xF)) \(relText(imm19))"
  }

  // b / bl label (imm26 << 2).
  if w & 0xFC00_0000 == 0x1400_0000 || w & 0xFC00_0000 == 0x9400_0000 {
    let link = (w >> 31) & 1 != 0
    let imm26 = Int64(Int32(bitPattern: w &<< 6) >> 6) * 4
    return "\(link ? "bl" : "b") \(relText(imm26))"
  }

  // stp/ldp pair, pre/post-index (64-bit).
  if w & 0xFFC0_0000 == 0xA980_0000 || w & 0xFFC0_0000 == 0xA8C0_0000 {
    let load = (w >> 22) & 1 != 0
    let imm7 = Int64(Int32(bitPattern: w &<< 10) >> 25) * 8  // bits[15:21], scaled by 8
    var s = "\(load ? "ldp" : "stp") \(gp(true, rd(w))), \(gp(true, ra(w))), [\(sp(true, rn(w)))"
    s += load ? "], #\(imm7)" : ", #\(imm7)]!"  // post-index `[base], #imm` / pre-index `[base, #imm]!`
    return s
  }

  // Logical immediate: and/orr/eor/ands, with mov (orr from zr) and tst (ands to zr) aliases.
  if w & 0x1F80_0000 == 0x1200_0000 {
    let opc = (w >> 29) & 3
    let nn: UInt32 = (w >> 22) & 1
    let immr: UInt32 = (w >> 16) & 0x3F
    let imms: UInt32 = (w >> 10) & 0x3F
    if let val = decodeBitmask(x64: sf(w), n: nn, imms: imms, immr: immr) {
      if opc == 1, rn(w) == 31 {  // mov Rd, #imm
        return "mov \(gp(sf(w), rd(w))), #0x\(hex(val, width: 0))"
      }
      if opc == 3, rd(w) == 31 {  // tst Rn, #imm
        return "tst \(gp(sf(w), rn(w))), #0x\(hex(val, width: 0))"
      }
      let mnem: String
      switch opc {
      case 0: mnem = "and"
      case 1: mnem = "orr"
      case 2: mnem = "eor"
      default: mnem = "ands"
      }
      return "\(mnem) \(sp(sf(w), rd(w))), \(gp(sf(w), rn(w))), #0x\(hex(val, width: 0))"
    }
  }

  // extr Rd, Rn, Rm, #lsb (ror when rn == rm). bit23 == 1 separates it from the bitfield ops.
  if w & 0x7F80_0000 == 0x1380_0000 {
    let lsb: UInt32 = (w >> 10) & 0x3F
    if rn(w) == rm(w) {
      return "ror \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), #\(lsb)"
    }
    return "extr \(gp(sf(w), rd(w))), \(gp(sf(w), rn(w))), \(gp(sf(w), rm(w))), #\(lsb)"
  }

  // STUR/LDUR: unscaled signed 9-bit offset (integer). size 00=b 01=h 10=w 11=x; opc picks
  // store / load / sign-extending load.
  if w & 0x3F20_0C00 == 0x3800_0000 {
    let size = w >> 30
    let opc = (w >> 22) & 3
    let imm9 = Int64(Int32(bitPattern: w &<< 11) >> 23)  // bits[20:12], sign-extended
    let load = opc != 0
    let signed = opc >= 2
    let toX = size == 3 || opc == 2  // x-sized access, or sign-extend to a 64-bit reg
    let suffix: String
    switch size {
    case 0: suffix = "b"
    case 1: suffix = "h"
    case 2: suffix = signed ? "w" : ""
    default: suffix = ""
    }
    var s =
      "\(load ? "ldur" : "stur")\(signed ? "s" : "")\(suffix) \(gp(toX, rd(w))), [\(sp(true, rn(w)))"
    s += imm9 != 0 ? ", #\(imm9)]" : "]"
    return s
  }

  // STUR/LDUR (FP/SIMD), unscaled signed 9-bit offset. Width is b/h/s/d by size, or q (128-bit)
  // when opc bit1 is set.
  if w & 0x3F20_0C00 == 0x3C00_0000 {
    let size = w >> 30
    let opc = (w >> 22) & 3
    let imm9 = Int64(Int32(bitPattern: w &<< 11) >> 23)
    let load = opc & 1 != 0
    let c: Character =
      opc & 2 != 0 ? "q" : size == 0 ? "b" : size == 1 ? "h" : size == 2 ? "s" : "d"
    var s = "\(load ? "ldur" : "stur") \(c)\(rd(w)), [\(sp(true, rn(w)))"
    s += imm9 != 0 ? ", #\(imm9)]" : "]"
    return s
  }

  // umulh / smulh (unsigned/signed 64x64 -> high 64). Ra field is 31.
  if w & 0xFFE0_FC00 == 0x9BC0_7C00 || w & 0xFFE0_FC00 == 0x9B40_7C00 {
    let uns = (w >> 23) & 1 != 0
    return "\(uns ? "umulh" : "smulh") x\(rd(w)), x\(rn(w)), x\(rm(w))"
  }

  // Bitfield move (SBFM/UBFM/BFM) with the usual aliases.
  if w & 0x1F80_0000 == 0x1300_0000, (w >> 29) & 3 != 3 {
    return bitfield(w)
  }

  // stp/ldp signed offset (bit31 selects 32/64-bit, bit22 selects load).
  if w & 0x7FC0_0000 == 0x2900_0000 || w & 0x7FC0_0000 == 0x2940_0000 {
    let x64 = (w >> 31) & 1 != 0
    let load = (w >> 22) & 1 != 0
    let scale: UInt32 = x64 ? 3 : 2
    let imm7 = Int64(Int32(bitPattern: w &<< 10) >> 25) << scale  // bits[21:15], scaled
    var s = "\(load ? "ldp" : "stp") \(gp(x64, rd(w))), \(gp(x64, ra(w))), [\(sp(true, rn(w)))"
    s += imm7 != 0 ? ", #\(imm7)]" : "]"
    return s
  }

  // adr / adrp Rd, imm (PC-relative address; adrp is page-scaled).
  if w & 0x1F00_0000 == 0x1000_0000 {
    let page = (w >> 31) & 1 != 0
    let immlo: Int64 = Int64((w >> 29) & 3)
    let immhi = Int64(Int32(bitPattern: (w & 0x00FF_FFE0) &<< 8) >> 13)  // bits[23:5], sign-extended
    var imm = (immhi << 2) | immlo
    if page { imm <<= 12 }
    if imm < 0 {
      return
        "\(page ? "adrp" : "adr") \(gp(true, rd(w))), .-0x\(hex(UInt64(bitPattern: -imm), width: 0))"
    }
    return
      "\(page ? "adrp" : "adr") \(gp(true, rd(w))), .+0x\(hex(UInt64(bitPattern: imm), width: 0))"
  }

  if let s = memImm(w) { return s }
  if let s = fpDecode(w) { return s }
  if let s = vecDecode(w) { return s }

  // Unknown: emit the raw word so nothing is silently lost.
  return ".word 0x\(hex(UInt64(w), width: 8))"
}

// MARK: - Branch targets and listings

/// The target word index of a PC-relative branch (b/bl/b.cond/cbz/cbnz/tbz/tbnz) at `index`,
/// or nil if `w` is not such a branch.
func branchTargetWord(_ w: UInt32, index: Int) -> Int? {
  let base = Int64(index)
  if w & 0x7C00_0000 == 0x1400_0000 {  // b / bl: imm26
    return Int(base + Int64(Int32(bitPattern: w &<< 6) >> 6))
  }
  if w & 0xFF00_0010 == 0x5400_0000 || w & 0x7E00_0000 == 0x3400_0000 {  // b.cond / cbz / cbnz: imm19
    return Int(base + Int64(Int32(bitPattern: w &<< 8) >> 13))
  }
  if w & 0x7E00_0000 == 0x3600_0000 {  // tbz / tbnz: imm14
    return Int(base + Int64(Int32(bitPattern: w &<< 13) >> 18))
  }
  return nil
}

/// The symbol whose range contains word `target` (the greatest symbol at or before it), or nil.
func containingSym(_ syms: [FuncSymbol], _ target: Int) -> FuncSymbol? {
  guard target >= 0 else { return nil }
  var best: FuncSymbol?
  for s in syms where s.offset / 4 <= target && (best == nil || s.offset / 4 > best!.offset / 4) {
    best = s
  }
  return best
}

func readWordLE(_ code: [UInt8], _ offset: Int) -> UInt32 {
  UInt32(code[offset]) | (UInt32(code[offset + 1]) << 8)
    | (UInt32(code[offset + 2]) << 16) | (UInt32(code[offset + 3]) << 24)
}

/// AArch64 disassembler: the `Disassembler` side of the target seam.
public enum AArch64Disassembler: Disassembler {
  /// Decode one instruction at `code[offset]` (little-endian word); nil when truncated.
  /// Unknown encodings return a `.word` directive, never nil.
  public static func disassembleOne(code: [UInt8], at offset: Int) -> (text: String, length: Int)? {
    guard offset + 4 <= code.count else { return nil }
    return (decodeWord(readWordLE(code, offset)), 4)
  }

  /// Render a single word as `mnemonic operands`.
  public static func one(word: UInt32) -> String { decodeWord(word) }

  /// Render a whole word stream (one word per line, `addr: hex  text`).
  public static func formatWords(_ words: [UInt32]) -> String {
    var out = ""
    for (i, w) in words.enumerated() {
      out += "\(hex(UInt64(i * 4), width: 4)): \(hex(UInt64(w), width: 8))  \(decodeWord(w))\n"
    }
    return out
  }

  /// Render a whole instruction stream (one word per line, `addr: hex  text`).
  /// A trailing partial word (fewer than 4 bytes) renders as `.byte` lines.
  public static func format(code: [UInt8]) -> String {
    var out = ""
    var i = 0
    while i + 4 <= code.count {
      let w = readWordLE(code, i)
      out += "\(hex(UInt64(i), width: 4)): \(hex(UInt64(w), width: 8))  \(decodeWord(w))\n"
      i += 4
    }
    while i < code.count {
      out +=
        "\(hex(UInt64(i), width: 4)): \(hex(UInt64(code[i]), width: 2))  .byte 0x\(hex(UInt64(code[i]), width: 2))\n"
      i += 1
    }
    return out
  }

  /// Render a listing with source-line markers interleaved (objdump `-S` style): a `; line N`
  /// header precedes the instructions belonging to that source line. `lines` must be sorted
  /// by offset (as isel produces).
  public static func formatWordsWithLines(_ words: [UInt32], lines: [LineEntry]) -> String {
    var out = ""
    var li = 0
    for (i, w) in words.enumerated() {
      let byteOff = UInt32(i * 4)
      while li < lines.count, lines[li].offset == byteOff {
        out += "; line \(lines[li].line)\n"
        li += 1
      }
      out += "\(hex(UInt64(byteOff), width: 4)): \(hex(UInt64(w), width: 8))  \(decodeWord(w))\n"
    }
    return out
  }

  /// Render a whole linked module: each function gets a `name:` label at its offset, and every
  /// resolved branch is annotated with the function it lands in (`<sym>` / `<sym+0xN>`).
  /// Symbol offsets are byte offsets into `code`.
  public static func formatModule(code: [UInt8], symbols: [FuncSymbol]) -> String {
    let words = stride(from: 0, to: code.count - code.count % 4, by: 4).map { readWordLE(code, $0) }
    var out = ""
    for (i, w) in words.enumerated() {
      for s in symbols where s.offset == i * 4 {
        if i != 0 { out += "\n" }
        out += "\(s.name):\n"
      }
      out += "\(hex(UInt64(i * 4), width: 4)): \(hex(UInt64(w), width: 8))  \(decodeWord(w))"
      // Annotate any PC-relative branch with the function it lands in, objdump-style.
      if let target = branchTargetWord(w, index: i), let s = containingSym(symbols, target) {
        let off = target - s.offset / 4
        out += off == 0 ? "  <\(s.name)>" : "  <\(s.name)+0x\(hex(UInt64(off * 4), width: 0))>"
      }
      out += "\n"
    }
    return out
  }
}

#endif
