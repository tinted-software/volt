#if ARMv7
  import Foundation

  public enum A32Instruction: Sendable, Equatable {
    case movImmediate(rd: UInt8, value: UInt32, s: Bool)
    case movRegister(rd: UInt8, rm: UInt8, s: Bool)
    case mvnImmediate(rd: UInt8, value: UInt32, s: Bool)
    case mvnRegister(rd: UInt8, rm: UInt8, s: Bool)
    case andImmediate(rd: UInt8, rn: UInt8, value: UInt32, s: Bool)
    case andRegister(rd: UInt8, rn: UInt8, rm: UInt8, s: Bool)
    case eorImmediate(rd: UInt8, rn: UInt8, value: UInt32, s: Bool)
    case eorRegister(rd: UInt8, rn: UInt8, rm: UInt8, s: Bool)
    case orrImmediate(rd: UInt8, rn: UInt8, value: UInt32, s: Bool)
    case orrRegister(rd: UInt8, rn: UInt8, rm: UInt8, s: Bool)
    case addImmediate(rd: UInt8, rn: UInt8, value: UInt32, s: Bool)
    case addRegister(rd: UInt8, rn: UInt8, rm: UInt8, s: Bool)
    case subImmediate(rd: UInt8, rn: UInt8, value: UInt32, s: Bool)
    case subRegister(rd: UInt8, rn: UInt8, rm: UInt8, s: Bool)
    case bxLR
    case svc(immediate: UInt32)
  }

  public struct A32DecodedInstruction: Sendable, Equatable {
    public let pc: UInt32
    public let word: UInt32
    public let instruction: A32Instruction
    /// Carry-out from the immediate shifter, when one is encoded and nonzero.
    public let shifterCarry: Bool?
  }

  public enum A32DecodeError: Error, Sendable, Equatable {
    case truncated(pc: UInt32, remainingBytes: Int)
    case unsupported(pc: UInt32, word: UInt32)
  }

  public enum ARMv7A32Decoder {
    public static func decode(_ bytes: [UInt8], at offset: Int = 0, pc: UInt32 = 0) throws
      -> A32DecodedInstruction
    {
      guard offset >= 0, offset <= bytes.count, bytes.count - offset >= 4 else {
        throw A32DecodeError.truncated(pc: pc, remainingBytes: max(0, bytes.count - max(offset, 0)))
      }
      let w =
        UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16
        | UInt32(bytes[offset + 3]) << 24
      guard w >> 28 == 0xE else { throw A32DecodeError.unsupported(pc: pc, word: w) }
      let instruction: A32Instruction
      if w & 0x0FFF_FFF0 == 0x012F_FF10, w & 15 == 14 {
        instruction = .bxLR
      } else if w & 0x0F00_0000 == 0x0F00_0000 {
        instruction = .svc(immediate: w & 0x00FF_FFFF)
      } else if w & 0x0C00_0000 == 0 && w & 0x0200_0000 != 0 {
        let opcode = (w >> 21) & 15
        let s = w & 0x0010_0000 != 0
        let rd = UInt8((w >> 12) & 15)
        let rn = UInt8((w >> 16) & 15)
        let imm8 = w & 255
        let rotate = Int((w >> 8) & 15) * 2
        let imm = rotate == 0 ? imm8 : (imm8 >> rotate) | (imm8 << (32 - rotate))
        switch opcode {
        case 0: instruction = .andImmediate(rd: rd, rn: rn, value: imm, s: s)
        case 1: instruction = .eorImmediate(rd: rd, rn: rn, value: imm, s: s)
        case 2: instruction = .subImmediate(rd: rd, rn: rn, value: imm, s: s)
        case 4: instruction = .addImmediate(rd: rd, rn: rn, value: imm, s: s)
        case 12: instruction = .orrImmediate(rd: rd, rn: rn, value: imm, s: s)
        case 13 where rn == 0: instruction = .movImmediate(rd: rd, value: imm, s: s)
        case 15 where rn == 0: instruction = .mvnImmediate(rd: rd, value: imm, s: s)
        default: throw A32DecodeError.unsupported(pc: pc, word: w)
        }
      } else if w & 0x0C00_0000 == 0 && w & 0x0200_0000 == 0
        && w & 0x0000_0FF0 == 0
      {
        let opcode = (w >> 21) & 15
        let s = w & 0x0010_0000 != 0
        let rd = UInt8((w >> 12) & 15)
        let rn = UInt8((w >> 16) & 15)
        let rm = UInt8(w & 15)
        switch opcode {
        case 0: instruction = .andRegister(rd: rd, rn: rn, rm: rm, s: s)
        case 1: instruction = .eorRegister(rd: rd, rn: rn, rm: rm, s: s)
        case 2: instruction = .subRegister(rd: rd, rn: rn, rm: rm, s: s)
        case 4: instruction = .addRegister(rd: rd, rn: rn, rm: rm, s: s)
        case 12: instruction = .orrRegister(rd: rd, rn: rn, rm: rm, s: s)
        case 13 where rn == 0: instruction = .movRegister(rd: rd, rm: rm, s: s)
        case 15 where rn == 0: instruction = .mvnRegister(rd: rd, rm: rm, s: s)
        default: throw A32DecodeError.unsupported(pc: pc, word: w)
        }
      } else {
        throw A32DecodeError.unsupported(pc: pc, word: w)
      }
      let immediateShifterCarry: Bool? = {
        guard w & 0x0C00_0000 == 0, w & 0x0200_0000 != 0 else { return nil }
        let rotate = Int((w >> 8) & 15) * 2
        guard rotate != 0 else { return nil }
        let imm8 = w & 0xFF
        let imm = (imm8 >> rotate) | (imm8 << (32 - rotate))
        return imm & 0x8000_0000 != 0
      }()
      let writesPC: Bool
      switch instruction {
      case .movImmediate(let rd, _, _), .mvnImmediate(let rd, _, _): writesPC = rd == 15
      case .movRegister(let rd, let rm, _), .mvnRegister(let rd, let rm, _):
        writesPC = rd == 15 || rm == 15
      case .andImmediate(let rd, let rn, _, _), .eorImmediate(let rd, let rn, _, _),
        .orrImmediate(let rd, let rn, _, _):
        writesPC = rd == 15 || rn == 15
      case .andRegister(let rd, let rn, let rm, _), .eorRegister(let rd, let rn, let rm, _),
        .orrRegister(let rd, let rn, let rm, _):
        writesPC = rd == 15 || rn == 15 || rm == 15
      case .addImmediate(let rd, let rn, _, _), .subImmediate(let rd, let rn, _, _):
        writesPC = rd == 15 || rn == 15
      case .addRegister(let rd, let rn, let rm, _), .subRegister(let rd, let rn, let rm, _):
        writesPC = rd == 15 || rn == 15 || rm == 15
      case .bxLR, .svc: writesPC = false
      }
      guard !writesPC else { throw A32DecodeError.unsupported(pc: pc, word: w) }
      return A32DecodedInstruction(
        pc: pc, word: w, instruction: instruction, shifterCarry: immediateShifterCarry)
    }
  }
#endif
