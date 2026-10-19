import VoltIR
import VoltTarget
import VoltTargetARMv7
import VoltTargetX86_64

/// Guest-visible A32 state with sixteen 32-bit registers and CPSR.
public struct ARMv7A32State {
  public var registers:
    (
      UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
      UInt32, UInt32, UInt32, UInt32, UInt32
    )
  public var cpsr: UInt32
  public init(registers: [UInt32] = Array(repeating: 0, count: 16), cpsr: UInt32 = 0) {
    precondition(registers.count == 16)
    self.registers = (
      registers[0], registers[1], registers[2], registers[3], registers[4], registers[5],
      registers[6], registers[7], registers[8], registers[9], registers[10], registers[11],
      registers[12], registers[13], registers[14], registers[15]
    )
    self.cpsr = cpsr
  }
  public subscript(_ register: Int) -> UInt32 {
    get {
      withUnsafeBytes(of: registers) { $0.load(fromByteOffset: register * 4, as: UInt32.self) }
    }
    set {
      withUnsafeMutableBytes(of: &registers) {
        $0.storeBytes(of: newValue, toByteOffset: register * 4, as: UInt32.self)
      }
    }
  }
}

public enum ARMv7EmulatorError: Error, Sendable, Equatable {
  case unsupportedInstruction(pc: UInt32, word: UInt32)
  case unsupportedSyscall(pc: UInt32, immediate: UInt32)
}

public enum ARMv7A32Lifter {
  public static func lift(_ code: [UInt8], pc: UInt32 = 0) throws -> Function {
    var decoded: [A32DecodedInstruction] = []
    var offset = 0
    while offset < code.count {
      let item: A32DecodedInstruction
      do {
        item = try ARMv7A32Decoder.decode(code, at: offset, pc: pc &+ UInt32(offset))
      } catch let error as A32DecodeError {
        if case .unsupported(let guestPC, let word) = error {
          throw ARMv7EmulatorError.unsupportedInstruction(pc: guestPC, word: word)
        }
        throw error
      }
      if case .svc(let imm) = item.instruction {
        throw ARMv7EmulatorError.unsupportedSyscall(pc: item.pc, immediate: imm)
      }
      decoded.append(item)
      offset += 4
    }
    let f = Function()
    let ptr = f.types.ptrGlobal()
    let i32 = f.types.intern(.int(signedness: false, bits: 32))
    let entry = f.appendBlock()
    let state = f.appendBlockParam(entry, ptr)
    let b = Builder(function: f, block: entry)
    for d in decoded {
      let regOffset: (UInt8) -> Int64 = { Int64($0) * 4 }
      switch d.instruction {
      case .movImmediate(let rd, let value):
        b.store(b.iconst(i32, Int64(value)), b.ptrAdd(state, byteOffset: regOffset(rd)))
      case .addImmediate(let rd, let rn, let value), .subImmediate(let rd, let rn, let value):
        let op: BinOp = {
          if case .addImmediate = d.instruction { return .add } else { return .sub }
        }()
        let lhs = b.load(i32, b.ptrAdd(state, byteOffset: regOffset(rn)))
        let result = b.arithImm(op, lhs, Int64(value))
        b.store(result, b.ptrAdd(state, byteOffset: regOffset(rd)))
      case .addRegister(let rd, let rn, let rm), .subRegister(let rd, let rn, let rm):
        let op: BinOp = {
          if case .addRegister = d.instruction { return .add } else { return .sub }
        }()
        let lhs = b.load(i32, b.ptrAdd(state, byteOffset: regOffset(rn)))
        let rhs = b.load(i32, b.ptrAdd(state, byteOffset: regOffset(rm)))
        b.store(b.arith(op, lhs, rhs), b.ptrAdd(state, byteOffset: regOffset(rd)))
      case .bxLR:
        let lr = b.load(i32, b.ptrAdd(state, byteOffset: regOffset(14)))
        b.store(lr, b.ptrAdd(state, byteOffset: regOffset(15)))
      case .svc: break
      }
      if case .bxLR = d.instruction { break }
      b.store(b.iconst(i32, Int64(d.pc &+ 4)), b.ptrAdd(state, byteOffset: regOffset(15)))
    }
    b.ret()
    return f
  }
}

public enum ARMv7A32Runner {
  public static func run(_ code: [UInt8], state: inout ARMv7A32State, pc: UInt32 = 0) throws {
    let f = try ARMv7A32Lifter.lift(code, pc: pc)
    let jit = try jitCompileModule(
      JITModule(functions: [NamedFunction(name: "guest", function: f)]))
    typealias GuestFn = @convention(c) (UnsafeMutableRawPointer) -> Void
    let entry: GuestFn = jit.entry(as: GuestFn.self, name: "guest")!
    withUnsafeMutablePointer(to: &state) { entry(UnsafeMutableRawPointer($0)) }
  }
}
