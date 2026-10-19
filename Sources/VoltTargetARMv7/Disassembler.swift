import VoltTarget

/// A useful ARM-state (A32) decoder for common integer, branch, and memory instructions.
/// Unsupported words are preserved as `.word` directives. Input is little-endian.
public enum ARMv7Disassembler: Disassembler {
  private static let conditions = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "", "nv",
  ]
  private static func reg(_ n: UInt32) -> String {
    n == 13 ? "sp" : n == 14 ? "lr" : n == 15 ? "pc" : "r\(n)"
  }
  private static func hex(_ n: UInt32, _ width: Int) -> String {
    var s = String(n, radix: 16)
    while s.count < width { s = "0" + s }
    return "0x" + s
  }
  private static func sign(_ value: UInt32, bits: UInt32) -> Int64 {
    let shift = 64 - bits
    return Int64(bitPattern: UInt64(value) << shift) >> shift
  }
  private static func decode(_ w: UInt32) -> String {
    let cond = conditions[Int(w >> 28)]
    let suffix = cond.isEmpty ? "" : cond
    if w == 0xFFFF_FFFF { return ".word 0xffffffff" }
    if w & 0x0FFF_FFF0 == 0x012F_FF10 { return "bx\(suffix) \(reg(w & 15))" }
    if w & 0x0F00_0000 == 0x0F00_0000 { return "swi\(suffix) #\(hex(w & 0x00FF_FFFF, 0))" }
    if w & 0x0E00_0000 == 0x0A00_0000 {
      let link = w & 0x0100_0000 != 0
      let displacement = sign(w & 0x00FF_FFFF, bits: 24) * 4 + 8
      let target = UInt32(truncatingIfNeeded: displacement)
      let magnitude = displacement < 0 ? 0 &- target : target
      return "\(link ? "bl" : "b")\(suffix) \(displacement < 0 ? "-" : "+")\(hex(magnitude, 0))"
    }
    if w & 0x0FC0_00F0 == 0x0000_0090 {
      let accumulate = w & 0x0020_0000 != 0
      let setFlags = w & 0x0010_0000 != 0
      let rd = (w >> 16) & 15
      let rn = (w >> 12) & 15
      let rs = (w >> 8) & 15
      let rm = w & 15
      return
        "\(accumulate ? "mla" : "mul")\(suffix)\(setFlags ? "s" : "") \(reg(rd)), \(reg(rm)), \(reg(rs))\(accumulate ? ", \(reg(rn))" : "")"
    }
    if w & 0x0C00_0000 == 0x0400_0000 {
      let load = w & 0x0010_0000 != 0
      let byte = w & 0x0040_0000 != 0
      let up = w & 0x0080_0000 != 0
      let pre = w & 0x0100_0000 != 0
      let rn = (w >> 16) & 15
      let rd = (w >> 12) & 15
      let imm = w & 0xFFF
      let offset = w & 0x0200_0000 != 0 ? "\(reg(w & 15)), lsl #\((w >> 7) & 31)" : "#\(imm)"
      let sign = up ? "" : "-"
      let address =
        pre
        ? "[\(reg(rn)), \(sign)\(offset)]\(w & 0x0020_0000 != 0 ? "!" : "")"
        : "[\(reg(rn))], \(sign)\(offset)"
      return "\(load ? "ldr" : "str")\(byte ? "b" : "")\(suffix) \(reg(rd)), \(address)"
    }
    if w & 0x0C00_0000 == 0 {
      let opcode = (w >> 21) & 15
      let names = [
        "and", "eor", "sub", "rsb", "add", "adc", "sbc", "rsc", "tst", "teq", "cmp", "cmn", "orr",
        "mov", "bic", "mvn",
      ]
      let name = names[Int(opcode)]
      let setFlags = w & 0x0010_0000 != 0
      let rd = (w >> 12) & 15
      let rn = (w >> 16) & 15
      let imm = w & 0x0200_0000 != 0
      let op2: String
      if imm {
        let value = (w & 255).rotatedRight(by: Int(((w >> 8) & 15) * 2))
        op2 = "#\(hex(value, 0))"
      } else {
        op2 =
          reg(w & 15)
          + ((w >> 4) & 1 != 0
            ? ", " + ["lsl", "lsr", "asr", "ror"][Int((w >> 5) & 3)] + " #\((w >> 7) & 31)" : "")
      }
      if opcode >= 8 && opcode <= 11 { return "\(name)\(suffix) \(reg(rn)), \(op2)" }
      if opcode == 13 || opcode == 15 {
        return "\(name)\(suffix)\(setFlags ? "s" : "") \(reg(rd)), \(op2)"
      }
      return "\(name)\(suffix)\(setFlags ? "s" : "") \(reg(rd)), \(reg(rn)), \(op2)"
    }
    return ".word \(hex(w, 8))"
  }
  private static func typedText(_ instruction: A32Instruction) -> String {
    func suffix(_ s: Bool) -> String { s ? "s" : "" }
    switch instruction {
    case .movImmediate(let rd, let value, let s):
      return "mov\(suffix(s)) \(reg(UInt32(rd))), #\(hex(value, 0))"
    case .movRegister(let rd, let rm, let s):
      return "mov\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rm)))"
    case .mvnImmediate(let rd, let value, let s):
      return "mvn\(suffix(s)) \(reg(UInt32(rd))), #\(hex(value, 0))"
    case .mvnRegister(let rd, let rm, let s):
      return "mvn\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rm)))"
    case .andImmediate(let rd, let rn, let value, let s):
      return "and\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), #\(hex(value, 0))"
    case .andRegister(let rd, let rn, let rm, let s):
      return "and\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), \(reg(UInt32(rm)))"
    case .eorImmediate(let rd, let rn, let value, let s):
      return "eor\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), #\(hex(value, 0))"
    case .eorRegister(let rd, let rn, let rm, let s):
      return "eor\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), \(reg(UInt32(rm)))"
    case .orrImmediate(let rd, let rn, let value, let s):
      return "orr\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), #\(hex(value, 0))"
    case .orrRegister(let rd, let rn, let rm, let s):
      return "orr\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), \(reg(UInt32(rm)))"
    case .addImmediate(let rd, let rn, let value, let s):
      return "add\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), #\(hex(value, 0))"
    case .addRegister(let rd, let rn, let rm, let s):
      return "add\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), \(reg(UInt32(rm)))"
    case .subImmediate(let rd, let rn, let value, let s):
      return "sub\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), #\(hex(value, 0))"
    case .subRegister(let rd, let rn, let rm, let s):
      return "sub\(suffix(s)) \(reg(UInt32(rd))), \(reg(UInt32(rn))), \(reg(UInt32(rm)))"
    case .bxLR: return "bx lr"
    case .svc(let immediate): return "swi #\(hex(immediate, 0))"
    }
  }
  public static func disassembleOne(code: [UInt8], at offset: Int) -> (text: String, length: Int)? {
    guard offset >= 0, offset + 4 <= code.count else { return nil }
    let w =
      UInt32(code[offset]) | UInt32(code[offset + 1]) << 8 | UInt32(code[offset + 2]) << 16
      | UInt32(code[offset + 3]) << 24
    if let typed = try? ARMv7A32Decoder.decode(code, at: offset) {
      return (typedText(typed.instruction), 4)
    }
    return (decode(w), 4)
  }
  public static func format(code: [UInt8]) -> String {
    var result = ""
    var i = 0
    while i + 4 <= code.count {
      let item = disassembleOne(code: code, at: i)!
      let w =
        UInt32(code[i]) | UInt32(code[i + 1]) << 8 | UInt32(code[i + 2]) << 16 | UInt32(code[i + 3])
        << 24
      let address = String(i, radix: 16)
      let word = String(w, radix: 16)
      result += address + ": " + word + "  " + item.text + "\n"
      i += 4
    }
    while i < code.count {
      let b = String(code[i], radix: 16)
      let address = String(i, radix: 16)
      result += address + ": " + b + "  .byte 0x" + b + "\n"
      i += 1
    }
    return result
  }
  public static func formatModule(code: [UInt8], symbols: [FuncSymbol]) -> String {
    format(code: code)
  }
}

extension UInt32 {
  fileprivate func rotatedRight(by count: Int) -> UInt32 {
    count == 0 ? self : (self >> count) | (self << (32 - count))
  }
}
