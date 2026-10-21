#if ARMv7Emulator
  import VoltIR
  import VoltTarget

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
      let boolTy = f.types.intern(.bool)
      let entry = f.appendBlock()
      let state = f.appendBlockParam(entry, ptr)
      let b = Builder(function: f, block: entry)
      let cpsrOffset: Int64 = 16 * 4
      // Condition flags in the guest CPSR: N=31, Z=30, C=29, V=28.
      func storeNZCV(result: Value, carry: Value, overflow: Value) {
        let zero = b.iconst(i32, 0)
        let one = b.iconst(i32, 1)
        let nBit = b.arithImm(.shr, result, 31)
        let zBit = b.select(b.icmp(.eq, result, zero, boolTy: boolTy), one, zero)
        let nz = b.arith(.bitOr, b.arithImm(.shl, nBit, 31), b.arithImm(.shl, zBit, 30))
        let nzc = b.arith(.bitOr, nz, b.arithImm(.shl, carry, 29))
        let flags = b.arith(.bitOr, nzc, b.arithImm(.shl, overflow, 28))
        let cpsrPtr = b.ptrAdd(state, byteOffset: cpsrOffset)
        let old = b.load(i32, cpsrPtr)
        let kept = b.arithImm(.bitAnd, old, Int64(0x0FFF_FFFF))
        b.store(b.arith(.bitOr, kept, flags), cpsrPtr)
      }
      // Logical S-form: N/Z from the result, C/V preserved.
      func storeNZ(result: Value) {
        let cpsrPtr = b.ptrAdd(state, byteOffset: cpsrOffset)
        let old = b.load(i32, cpsrPtr)
        let zero = b.iconst(i32, 0)
        let one = b.iconst(i32, 1)
        let nBit = b.arithImm(.shr, result, 31)
        let zBit = b.select(b.icmp(.eq, result, zero, boolTy: boolTy), one, zero)
        let nz = b.arith(.bitOr, b.arithImm(.shl, nBit, 31), b.arithImm(.shl, zBit, 30))
        let nzMasked = b.arithImm(.bitAnd, nz, Int64(0xC000_0000))
        let cv = b.arithImm(.bitAnd, old, Int64(0x3000_0000))
        let low = b.arithImm(.bitAnd, old, Int64(0x0FFF_FFFF))
        b.store(b.arith(.bitOr, low, b.arith(.bitOr, cv, nzMasked)), cpsrPtr)
      }
      // Unsigned carry out of bit 31 for `a + b = r`:
      // C = bit31((a & b) | ((a ^ b) & ~r)).
      func carryAdd(a: Value, bb: Value, r: Value) -> Value {
        let ab = b.arith(.bitAnd, a, bb)
        let axb = b.arith(.bitXor, a, bb)
        let notR = b.arithImm(.bitXor, r, Int64(0xFFFF_FFFF))
        let c = b.arith(.bitOr, ab, b.arith(.bitAnd, axb, notR))
        return b.arithImm(.bitAnd, b.arithImm(.shr, c, 31), 1)
      }
      // Unsigned carry (not-borrow) for `a - b = r`:
      // C = bit31((a & ~b) | (~(a ^ b) & ~r)).
      func carrySub(a: Value, bb: Value, r: Value) -> Value {
        let notB = b.arithImm(.bitXor, bb, Int64(0xFFFF_FFFF))
        let aNotB = b.arith(.bitAnd, a, notB)
        let axb = b.arith(.bitXor, a, bb)
        let notAxb = b.arithImm(.bitXor, axb, Int64(0xFFFF_FFFF))
        let notR = b.arithImm(.bitXor, r, Int64(0xFFFF_FFFF))
        let c = b.arith(.bitOr, aNotB, b.arith(.bitAnd, notAxb, notR))
        return b.arithImm(.bitAnd, b.arithImm(.shr, c, 31), 1)
      }
      // Signed overflow for addition: V = bit31(~(a ^ b) & (a ^ r)).
      func overflowAdd(a: Value, bb: Value, r: Value) -> Value {
        let axb = b.arith(.bitXor, a, bb)
        let notAxb = b.arithImm(.bitXor, axb, Int64(0xFFFF_FFFF))
        let axr = b.arith(.bitXor, a, r)
        let v = b.arith(.bitAnd, notAxb, axr)
        return b.arithImm(.bitAnd, b.arithImm(.shr, v, 31), 1)
      }
      // Signed overflow for subtraction: V = bit31((a ^ b) & (a ^ r)).
      func overflowSub(a: Value, bb: Value, r: Value) -> Value {
        let axb = b.arith(.bitXor, a, bb)
        let axr = b.arith(.bitXor, a, r)
        let v = b.arith(.bitAnd, axb, axr)
        return b.arithImm(.bitAnd, b.arithImm(.shr, v, 31), 1)
      }
      // ISel lowers i32 arithmetic as 64-bit ops, so clear the high half:
      // shl 32 then logical shr 32 keeps the low 32 bits.
      func mask32(_ v: Value) -> Value { b.arithImm(.shr, b.arithImm(.shl, v, 32), 32) }
      func emitAdd(rd: UInt8, a: Value, bb: Value, s: Bool) {
        let r = mask32(b.arith(.add, a, bb))
        b.store(r, b.ptrAdd(state, byteOffset: Int64(rd) * 4))
        if s {
          storeNZCV(
            result: r, carry: carryAdd(a: a, bb: bb, r: r),
            overflow: overflowAdd(a: a, bb: bb, r: r))
        }
      }
      func emitSub(rd: UInt8, a: Value, bb: Value, s: Bool) {
        let r = mask32(b.arith(.sub, a, bb))
        b.store(r, b.ptrAdd(state, byteOffset: Int64(rd) * 4))
        if s {
          storeNZCV(
            result: r, carry: carrySub(a: a, bb: bb, r: r),
            overflow: overflowSub(a: a, bb: bb, r: r))
        }
      }
      func emitLogic(rd: UInt8, r: Value, s: Bool) {
        let r = mask32(r)
        b.store(r, b.ptrAdd(state, byteOffset: Int64(rd) * 4))
        if s { storeNZ(result: r) }
      }
      for d in decoded {
        let regOffset: (UInt8) -> Int64 = { Int64($0) * 4 }
        func loadReg(_ r: UInt8) -> Value { b.load(i32, b.ptrAdd(state, byteOffset: regOffset(r))) }
        switch d.instruction {
        case .movImmediate(let rd, let value, let s):
          emitLogic(rd: rd, r: b.iconst(i32, Int64(value)), s: s)
        case .movRegister(let rd, let rm, let s):
          emitLogic(rd: rd, r: loadReg(rm), s: s)
        case .mvnImmediate(let rd, let value, let s):
          emitLogic(
            rd: rd, r: b.arithImm(.bitXor, b.iconst(i32, Int64(value)), Int64(0xFFFF_FFFF)), s: s)
        case .mvnRegister(let rd, let rm, let s):
          emitLogic(rd: rd, r: b.arithImm(.bitXor, loadReg(rm), Int64(0xFFFF_FFFF)), s: s)
        case .andImmediate(let rd, let rn, let value, let s):
          emitLogic(rd: rd, r: b.arithImm(.bitAnd, loadReg(rn), Int64(value)), s: s)
        case .andRegister(let rd, let rn, let rm, let s):
          emitLogic(rd: rd, r: b.arith(.bitAnd, loadReg(rn), loadReg(rm)), s: s)
        case .eorImmediate(let rd, let rn, let value, let s):
          emitLogic(rd: rd, r: b.arithImm(.bitXor, loadReg(rn), Int64(value)), s: s)
        case .eorRegister(let rd, let rn, let rm, let s):
          emitLogic(rd: rd, r: b.arith(.bitXor, loadReg(rn), loadReg(rm)), s: s)
        case .orrImmediate(let rd, let rn, let value, let s):
          emitLogic(rd: rd, r: b.arithImm(.bitOr, loadReg(rn), Int64(value)), s: s)
        case .orrRegister(let rd, let rn, let rm, let s):
          emitLogic(rd: rd, r: b.arith(.bitOr, loadReg(rn), loadReg(rm)), s: s)
        case .addImmediate(let rd, let rn, let value, let s):
          emitAdd(rd: rd, a: loadReg(rn), bb: b.iconst(i32, Int64(value)), s: s)
        case .addRegister(let rd, let rn, let rm, let s):
          emitAdd(rd: rd, a: loadReg(rn), bb: loadReg(rm), s: s)
        case .subImmediate(let rd, let rn, let value, let s):
          emitSub(rd: rd, a: loadReg(rn), bb: b.iconst(i32, Int64(value)), s: s)
        case .subRegister(let rd, let rn, let rm, let s):
          emitSub(rd: rd, a: loadReg(rn), bb: loadReg(rm), s: s)
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
#endif
