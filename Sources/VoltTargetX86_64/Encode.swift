//! x86-64 instruction encoding. Each encoder returns `[UInt8]` bytes.
//! An x86 instruction is at most 15 bytes long.
//! Modeled on `libs/vulcan-target/x86_64/encode.zig`.
//!
//! Encoding model for a 64-bit operand size: a REX prefix `0x48 | R | X | B`
//! (W=1 selects 64-bit, R and B extend the ModRM reg and rm fields to r8..r15),
//! then the opcode, then a register-direct ModRM byte `0xC0 | (reg<<3) | rm`.
//!
//! v1 scope: the GPR core (moves, ALU, shifts, compares, jumps, calls, stack
//! slots, RIP-relative addressing). The SSE/AVX forms land with float codegen.

/// A 64-bit general register. The low 3 bits give the hardware encoding. r8..r15
/// have bit 3 set, so they need a REX prefix bit.
public enum Reg: UInt8, Sendable, Hashable, CaseIterable {
  case rax = 0
  case rcx = 1
  case rdx = 2
  case rbx = 3
  case rsp = 4
  case rbp = 5
  case rsi = 6
  case rdi = 7
  case r8 = 8
  case r9 = 9
  case r10 = 10
  case r11 = 11
  case r12 = 12
  case r13 = 13
  case r14 = 14
  case r15 = 15
}

/// A condition code: the low nibble of the Jcc/SETcc opcode.
public enum Cond: UInt8, Sendable, Hashable, CaseIterable {
  case e = 0x4  // equal / zero
  case ne = 0x5  // not equal
  case l = 0xC  // signed <
  case ge = 0xD  // signed >=
  case le = 0xE  // signed <=
  case g = 0xF  // signed >
  case b = 0x2  // unsigned <
  case ae = 0x3  // unsigned >=
  case be = 0x6  // unsigned <=
  case a = 0x7  // unsigned >
}

/// The logical inverse of a condition. In the x86 condition-code encoding, bit 0
/// toggles the sense, so the inverse is the paired code. This lets an `if` fall
/// through one edge by branching to the other label on the inverted predicate.
public func invertCond(_ cond: Cond) -> Cond {
  switch cond {
  case .e: return .ne
  case .ne: return .e
  case .l: return .ge
  case .ge: return .l
  case .le: return .g
  case .g: return .le
  case .b: return .ae
  case .ae: return .b
  case .be: return .a
  case .a: return .be
  }
}

func le32(_ v: UInt32) -> [UInt8] {
  [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)]
}

func le64(_ v: UInt64) -> [UInt8] {
  (0..<8).map { UInt8((v >> ($0 * 8)) & 0xFF) }
}

/// The REX prefix for a 64-bit op with ModRM `reg` and `rm` register operands.
func rexW(_ reg: Reg, _ rm: Reg) -> UInt8 {
  0x48 | ((reg.rawValue >= 8 ? 1 : 0) << 2) | (rm.rawValue >= 8 ? 1 : 0)
}

/// The REX prefix for a two-register op at the requested operand size, or nil when
/// none is needed. `w` = true selects 64-bit (REX.W). At 32-bit, a REX.B/R bit is
/// still needed to name r8..r15, so this emits a REX with W=0 then. When neither
/// applies (32-bit, both operands rax..rdi), no REX byte at all: a bare 0x40 is
/// legal but non-canonical, and omitting it keeps 32-bit encodings small.
func rexRR(_ reg: Reg, _ rm: Reg, _ w: Bool) -> UInt8? {
  let ext: UInt8 = ((reg.rawValue >= 8 ? 1 : 0) << 2) | (rm.rawValue >= 8 ? 1 : 0)
  if w { return 0x48 | ext }
  if ext != 0 { return 0x40 | ext }
  return nil
}

/// A register-direct ModRM byte (mod = 11).
func modrm(_ reg: Reg, _ rm: Reg) -> UInt8 {
  0xC0 | ((reg.rawValue & 7) << 3) | (rm.rawValue & 7)
}

/// A two-register ALU op: `opcode` with `reg` as the source and `rm` as the
/// destination (the `/r` form, e.g. ADD r/m, r). `w` picks the operand size: true
/// selects 64-bit (REX.W), false selects 32-bit. A 32-bit op auto-zeroes the upper
/// 32 bits of its destination, so an i32 result stays clean.
func aluRR(_ opcode: UInt8, _ src: Reg, _ dst: Reg, _ w: Bool) -> [UInt8] {
  if let rex = rexRR(src, dst, w) { return [rex, opcode, modrm(src, dst)] }
  return [opcode, modrm(src, dst)]
}

/// `mov dst, src` (89 /r). Always 64-bit: a plain register copy does not depend
/// on width, so this always uses REX.W to keep every move byte-identical.
public func movReg(_ dst: Reg, _ src: Reg) -> [UInt8] { aluRR(0x89, src, dst, true) }

/// `add dst, src` (01 /r).
public func addRR(_ dst: Reg, _ src: Reg, _ w: Bool) -> [UInt8] { aluRR(0x01, src, dst, w) }

/// `sub dst, src` (29 /r).
public func subRR(_ dst: Reg, _ src: Reg, _ w: Bool) -> [UInt8] { aluRR(0x29, src, dst, w) }

/// `and dst, src` (21 /r).
public func andRR(_ dst: Reg, _ src: Reg, _ w: Bool) -> [UInt8] { aluRR(0x21, src, dst, w) }

/// `or dst, src` (09 /r).
public func orRR(_ dst: Reg, _ src: Reg, _ w: Bool) -> [UInt8] { aluRR(0x09, src, dst, w) }

/// `xor dst, src` (31 /r).
public func xorRR(_ dst: Reg, _ src: Reg, _ w: Bool) -> [UInt8] { aluRR(0x31, src, dst, w) }

/// `imul dst, src` (0F AF /r): two-operand signed multiply. A 32-bit imul
/// zero-extends its 32-bit product into the destination.
public func imulRR(_ dst: Reg, _ src: Reg, _ w: Bool) -> [UInt8] {
  if let rex = rexRR(dst, src, w) { return [rex, 0x0F, 0xAF, modrm(dst, src)] }
  return [0x0F, 0xAF, modrm(dst, src)]
}

/// `cmp a, b` (39 /r): compute `a - b` and set flags. `w` picks the operand size,
/// so an i32 compare sets flags from the low 32 bits only.
public func cmpRR(_ a: Reg, _ b: Reg, _ w: Bool) -> [UInt8] { aluRR(0x39, b, a, w) }

/// `test a, b` (85 /r): compute `a & b` and set flags. `test r, r` tests zero.
public func testRR(_ a: Reg, _ b: Reg, _ w: Bool) -> [UInt8] { aluRR(0x85, b, a, w) }

/// `mov dst, imm` loads an immediate. `w` = true selects `mov r64, imm32`
/// (REX.W C7 /0), sign-extending imm32 to 64 bits. `w` = false selects
/// `mov r32, imm32` (B8+rd), zero-extending into the full register.
public func movImm(_ dst: Reg, _ imm: Int32, _ w: Bool) -> [UInt8] {
  let u = UInt32(bitPattern: imm)
  if w {
    return [0x48 | (dst.rawValue >= 8 ? 1 : 0), 0xC7, 0xC0 | (dst.rawValue & 7)] + le32(u)
  }
  // A 32-bit `mov r32, imm32` zero-extends. r8..r15 need REX.B (W=0).
  if dst.rawValue >= 8 {
    return [0x41, 0xB8 | (dst.rawValue & 7)] + le32(u)
  }
  return [0xB8 | (dst.rawValue & 7)] + le32(u)
}

/// `mov dst, imm64` (REX.W B8+r): a full 64-bit immediate.
public func movImm64(_ dst: Reg, _ imm: UInt64) -> [UInt8] {
  [0x48 | (dst.rawValue >= 8 ? 1 : 0), 0xB8 | (dst.rawValue & 7)] + le64(imm)
}

/// An ALU op against a 32-bit immediate (81 /digit id): ADD=/0, OR=/1, AND=/4,
/// SUB=/5, XOR=/6, CMP=/7. `w` picks the operand size.
public enum AluDigit: UInt8 {
  case add = 0
  case orr = 1
  case adc = 2
  case sbb = 3
  case and = 4
  case sub = 5
  case xor = 6
  case cmp = 7
}

public func aluImm(_ digit: AluDigit, _ dst: Reg, _ imm: Int32, _ w: Bool) -> [UInt8] {
  let u = UInt32(bitPattern: imm)
  let mrm: UInt8 = 0xC0 | (digit.rawValue << 3) | (dst.rawValue & 7)
  if let rex = rexOne(dst.rawValue >= 8, w) { return [rex, 0x81, mrm] + le32(u) }
  return [0x81, mrm] + le32(u)
}

/// The optional REX prefix for a single-register op at operand size `w`: REX.W for
/// 64-bit; REX.B only when the register is r8..r15 at 32-bit; else none.
func rexOne(_ ext: Bool, _ w: Bool) -> UInt8? {
  if w { return 0x48 | (ext ? 1 : 0) }
  if ext { return 0x41 }
  return nil
}

/// `imul dst, src, imm32` (69 /r id): three-operand multiply.
public func imulImm(_ dst: Reg, _ src: Reg, _ imm: Int32, _ w: Bool) -> [UInt8] {
  let u = UInt32(bitPattern: imm)
  if let rex = rexRR(dst, src, w) { return [rex, 0x69, modrm(dst, src)] + le32(u) }
  return [0x69, modrm(dst, src)] + le32(u)
}

/// A shift of `dst` by an immediate count (C1 /digit ib): SHL=/4, SHR=/5, SAR=/7.
/// A 32-bit shr/sar shifts within 32 bits, filling from bit 31.
public enum ShiftDigit: UInt8 {
  case shl = 4
  case shr = 5
  case sar = 7
}

public func shiftImm(_ digit: ShiftDigit, _ dst: Reg, _ count: UInt8, _ w: Bool) -> [UInt8] {
  let mrm: UInt8 = 0xC0 | (digit.rawValue << 3) | (dst.rawValue & 7)
  if let rex = rexOne(dst.rawValue >= 8, w) { return [rex, 0xC1, mrm, count] }
  return [0xC1, mrm, count]
}

/// A shift of `dst` by the count held in CL (D3 /digit): SHL=/4, SHR=/5 (logical),
/// SAR=/7 (arithmetic).
public func shiftCl(_ digit: UInt8, _ dst: Reg, _ w: Bool) -> [UInt8] {
  let mrm: UInt8 = 0xC0 | (digit << 3) | (dst.rawValue & 7)
  var out: [UInt8] = []
  if w || dst.rawValue >= 8 {
    out.append(0x48 | (dst.rawValue >= 8 ? 1 : 0))
  }
  out += [0xD3, mrm]
  return out
}

/// `shl dst, cl`: logical shift left by CL.
public func shlCl(_ dst: Reg, _ w: Bool) -> [UInt8] { shiftCl(4, dst, w) }

/// `shr dst, cl`: logical shift right by CL.
public func shrCl(_ dst: Reg, _ w: Bool) -> [UInt8] { shiftCl(5, dst, w) }

/// `sar dst, cl`: arithmetic shift right by CL.
public func sarCl(_ dst: Reg, _ w: Bool) -> [UInt8] { shiftCl(7, dst, w) }

/// `cqo` (REX.W 99): sign-extend RAX into RDX:RAX, the dividend for `idiv`.
public func cqo() -> [UInt8] { [0x48, 0x99] }

/// `cdq` (99): sign-extend EAX into EDX:EAX, the dividend for 32-bit `idiv`.
public func cdq() -> [UInt8] { [0x99] }

/// `idiv src` (F7 /7): signed divide of RDX:RAX (or EDX:EAX). Quotient to
/// (R/E)AX, remainder to (R/E)DX. `src` must not be RAX or RDX.
public func idiv(_ src: Reg, _ w: Bool) -> [UInt8] {
  let mrm: UInt8 = 0xC0 | (7 << 3) | (src.rawValue & 7)
  if let rex = rexOne(src.rawValue >= 8, w) { return [rex, 0xF7, mrm] }
  return [0xF7, mrm]
}

/// `div src` (F7 /6): unsigned divide. Clear the high half first. A 32-bit div
/// reads only EAX as the dividend, correctly ignoring a dirty upper half of RAX.
public func divu(_ src: Reg, _ w: Bool) -> [UInt8] {
  let mrm: UInt8 = 0xC0 | (6 << 3) | (src.rawValue & 7)
  if let rex = rexOne(src.rawValue >= 8, w) { return [rex, 0xF7, mrm] }
  return [0xF7, mrm]
}

/// `cmovcc dst, src` (0F 40+cc /r): move `src` into `dst` if the condition holds.
public func cmovcc(_ dst: Reg, _ src: Reg, _ cond: Cond) -> [UInt8] {
  [rexW(dst, src), 0x0F, 0x40 | cond.rawValue, modrm(dst, src)]
}

/// `setcc dst8` (0F 90+cc): set the low byte of `dst` from the flags. Always
/// emits a REX prefix so the encoding addresses the low byte (spl/sil/... for
/// regs 4..7, r8b.. for 8..15), never the legacy ah/ch/dh/bh.
public func setcc(_ dst: Reg, _ cond: Cond) -> [UInt8] {
  [0x40 | (dst.rawValue >= 8 ? 1 : 0), 0x0F, 0x90 | cond.rawValue, 0xC0 | (dst.rawValue & 7)]
}

/// `movzx dst, src8` (REX.W 0F B6 /r): zero-extend the low byte.
public func movzxByte(_ dst: Reg, _ src: Reg) -> [UInt8] {
  [rexW(dst, src), 0x0F, 0xB6, modrm(dst, src)]
}

/// `movsx dst, src8` (REX.W 0F BE /r): sign-extend the low byte.
public func movsxByte(_ dst: Reg, _ src: Reg) -> [UInt8] {
  [rexW(dst, src), 0x0F, 0xBE, modrm(dst, src)]
}

/// `movzx dst, src16` (REX.W 0F B7 /r): zero-extend the low word.
public func movzxWord(_ dst: Reg, _ src: Reg) -> [UInt8] {
  [rexW(dst, src), 0x0F, 0xB7, modrm(dst, src)]
}

/// `movsx dst, src16` (REX.W 0F BF /r): sign-extend the low word.
public func movsxWord(_ dst: Reg, _ src: Reg) -> [UInt8] {
  [rexW(dst, src), 0x0F, 0xBF, modrm(dst, src)]
}

/// `movsxd dst, src32` (REX.W 63 /r): sign-extend the low 32 bits. Needed because
/// a 32-bit ALU result auto-zero-extends (an architectural rule), which is wrong
/// for a signed widening convert: a negative i32 must sign-extend into an i64.
public func movsxdReg(_ dst: Reg, _ src: Reg) -> [UInt8] {
  [rexW(dst, src), 0x63, modrm(dst, src)]
}

/// `mov dst32, src32` (89 /r, no REX.W): a 32-bit copy. The architectural 32-bit
/// side effect zero-extends `dst`, which is the correct unsigned 32-to-64 widening.
public func movReg32(_ dst: Reg, _ src: Reg) -> [UInt8] { aluRR(0x89, src, dst, false) }

/// `jcc rel32` (0F 80+cc cd): branch when the condition holds. `rel` is measured
/// from the end of the instruction.
public func jcc(_ cond: Cond, _ rel: Int32) -> [UInt8] {
  [0x0F, 0x80 | cond.rawValue] + le32(UInt32(bitPattern: rel))
}

/// `jmp rel32` (E9 cd): unconditional relative jump.
public func jmp(_ rel: Int32) -> [UInt8] {
  [0xE9] + le32(UInt32(bitPattern: rel))
}

/// `mov dst, [rsp + disp32]` (REX.W 8B /r): load a stack slot. The RSP base
/// needs a SIB byte, 0x24.
public func movFromStack(_ dst: Reg, _ disp: Int32) -> [UInt8] {
  [rexW(dst, .rax), 0x8B, 0x84 | ((dst.rawValue & 7) << 3), 0x24] + le32(UInt32(bitPattern: disp))
}

/// `mov [rsp + disp32], src` (REX.W 89 /r): store to a stack slot.
public func movToStack(_ disp: Int32, _ src: Reg) -> [UInt8] {
  [rexW(src, .rax), 0x89, 0x84 | ((src.rawValue & 7) << 3), 0x24] + le32(UInt32(bitPattern: disp))
}

/// A general `mov` to or from `[base + disp32]` with an arbitrary base register
/// (mod=10, disp32). A base of rsp(4) or r12 needs the 0x24 SIB byte. rm=rbp(5)
/// is fine at mod=10 since disp32 is explicit. `op` is 0x8B for a load or 0x89
/// for a store.
func movMem(_ op: UInt8, _ data: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  let modrmByte: UInt8 = 0x80 | ((data.rawValue & 7) << 3) | (base.rawValue & 7)
  let prefix: [UInt8] = [0x48 | ((data.rawValue >= 8 ? 1 : 0) << 2) | (base.rawValue >= 8 ? 1 : 0)]
  let tail = le32(UInt32(bitPattern: disp))
  if (base.rawValue & 7) == 4 {  // rsp/r12 base needs a SIB byte (base=rm, no index)
    return prefix + [op, modrmByte, 0x24] + tail
  }
  return prefix + [op, modrmByte] + tail
}

/// `mov dst, [base + disp32]` (REX.W 8B /r): load 64 bits.
public func movFromMem(_ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  movMem(0x8B, dst, base, disp)
}

/// `mov [base + disp32], src` (REX.W 89 /r): store 64 bits.
public func movToMem(_ base: Reg, _ disp: Int32, _ src: Reg) -> [UInt8] {
  movMem(0x89, src, base, disp)
}

/// A memory mov without REX.W (32-bit operand): 32-bit loads zero-extend into
/// the full register. A base of rsp/r12 needs the 0x24 SIB byte.
func movMem32(_ op: UInt8, _ data: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  let modrmByte: UInt8 = 0x80 | ((data.rawValue & 7) << 3) | (base.rawValue & 7)
  var out: [UInt8] = []
  if data.rawValue >= 8 || base.rawValue >= 8 {
    out.append(0x40 | ((data.rawValue >= 8 ? 1 : 0) << 2) | (base.rawValue >= 8 ? 1 : 0))
  }
  out.append(op)
  out.append(modrmByte)
  if (base.rawValue & 7) == 4 { out.append(0x24) }
  out += le32(UInt32(bitPattern: disp))
  return out
}

/// 32-bit integer load `mov r32, [base+disp]` (zero-extends to the 64-bit register).
public func movFromMem32(_ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  movMem32(0x8B, dst, base, disp)
}

/// 32-bit integer store `mov [base+disp], r32` (writes exactly 4 bytes).
public func movToMem32(_ base: Reg, _ disp: Int32, _ src: Reg) -> [UInt8] {
  movMem32(0x89, src, base, disp)
}

/// Sign-extending 32-bit load `movsxd r64, [base+disp]` (63 /r, REX.W).
public func movsxdFromMem(_ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  movMem(0x63, dst, base, disp)
}

/// A sign- or zero-extending narrow load, `movsx`/`movzx dst32, byte/word ptr
/// [base+disp32]` (0F op). The 32-bit destination auto-zeroes the upper half,
/// keeping an i8/i16 result clean. A base of rsp/r12 needs the SIB byte.
func extMem(_ op: UInt8, _ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  let modrmByte: UInt8 = 0x80 | ((dst.rawValue & 7) << 3) | (base.rawValue & 7)
  var out: [UInt8] = []
  if dst.rawValue >= 8 || base.rawValue >= 8 {
    out.append(0x40 | ((dst.rawValue >= 8 ? 1 : 0) << 2) | (base.rawValue >= 8 ? 1 : 0))
  }
  out += [0x0F, op, modrmByte]
  if (base.rawValue & 7) == 4 { out.append(0x24) }
  out += le32(UInt32(bitPattern: disp))
  return out
}

/// `movsx r32, byte ptr [base+disp]` (0F BE): signed i8 load.
public func movsxByteFromMem(_ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  extMem(0xBE, dst, base, disp)
}

/// `movzx r32, byte ptr [base+disp]` (0F B6): unsigned i8 load.
public func movzxByteFromMem(_ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  extMem(0xB6, dst, base, disp)
}

/// `movsx r32, word ptr [base+disp]` (0F BF): signed i16 load.
public func movsxWordFromMem(_ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  extMem(0xBF, dst, base, disp)
}

/// `movzx r32, word ptr [base+disp]` (0F B7): unsigned i16 load.
public func movzxWordFromMem(_ dst: Reg, _ base: Reg, _ disp: Int32) -> [UInt8] {
  extMem(0xB7, dst, base, disp)
}

/// `lea dst, [rsp + disp32]` (REX.W 8D /r): materialize a stack address, e.g. an
/// alloca slot. The rsp base needs the 0x24 SIB byte, mod=10 for disp32.
public func leaFromStack(_ dst: Reg, _ disp: Int32) -> [UInt8] {
  [rexW(dst, .rax), 0x8D, 0x84 | ((dst.rawValue & 7) << 3), 0x24] + le32(UInt32(bitPattern: disp))
}

/// `lea dst, [rip + disp32]` (REX.W 8D /r): materialize a PC-relative symbol
/// address, a `global_addr`. Mod=00, rm=101 is the RIP-relative special case (no
/// SIB, no base register), so the `rm` side of the REX contributes nothing.
public func leaRipRel(_ dst: Reg, _ disp: Int32) -> [UInt8] {
  let modrmByte: UInt8 = 0x05 | ((dst.rawValue & 7) << 3)
  return [rexW(dst, .rax), 0x8D, modrmByte] + le32(UInt32(bitPattern: disp))
}

/// `mov dst, qword [rip + disp32]` (REX.W 8B /r): load the 64-bit value at a
/// PC-relative address — a GOT-indirect `global_addr` (`via_got`).
public func movRipRel(_ dst: Reg, _ disp: Int32) -> [UInt8] {
  let modrmByte: UInt8 = 0x05 | ((dst.rawValue & 7) << 3)
  return [rexW(dst, .rax), 0x8B, modrmByte] + le32(UInt32(bitPattern: disp))
}

/// `call rel32` (E8 cd): a relative call. `rel` is measured from the end of the
/// instruction. The 4-byte displacement is the last 4 bytes — the relocation site.
public func callRel(_ rel: Int32) -> [UInt8] {
  [0xE8] + le32(UInt32(bitPattern: rel))
}

/// `call r64` (FF /2): an indirect call through a register. No REX.W needed (call
/// defaults to 64-bit in long mode); r8..r15 need REX.B.
public func callReg(_ reg: Reg) -> [UInt8] {
  let mrm: UInt8 = 0xD0 | (reg.rawValue & 7)  // ModRM mod=11, reg field=/2, rm=reg
  if reg.rawValue >= 8 { return [0x41, 0xFF, mrm] }
  return [0xFF, mrm]
}

/// `push r64` (0x50+rd): rsp -= 8. No REX.W needed; r8..r15 need REX.B.
public func pushReg(_ r: Reg) -> [UInt8] {
  if r.rawValue >= 8 { return [0x41, 0x50 | (r.rawValue & 7)] }
  return [0x50 | r.rawValue]
}

/// `pop r64` (0x58+rd): rsp += 8.
public func popReg(_ r: Reg) -> [UInt8] {
  if r.rawValue >= 8 { return [0x41, 0x58 | (r.rawValue & 7)] }
  return [0x58 | r.rawValue]
}

/// `ret` (near return).
public func x86Ret() -> [UInt8] { [0xC3] }
