#if ARMv7Emulator
  import Testing
  import VoltEmulator
  import VoltTargetARMv7

  @Suite("ARMv7 A32 emulator")
  struct ARMv7EmulatorTests {
    @Test("demo block executes and returns through BX LR")
    func demoAndBX() throws {
      let bytes: [UInt8] = [0x05, 0x00, 0xA0, 0xE3, 0x07, 0x00, 0x80, 0xE2, 0x1E, 0xFF, 0x2F, 0xE1]
      var state = ARMv7A32State()
      state[14] = 0x1234
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[0] == 12)
      #expect(state[15] == 0x1234)
    }

    @Test("register-source arithmetic executes")
    func registerArithmetic() throws {
      let bytes: [UInt8] = [
        0x04, 0x00, 0xA0, 0xE3, 0x03, 0x10, 0xA0, 0xE3, 0x01, 0x00, 0x80, 0xE0, 0x1E, 0xFF, 0x2F,
        0xE1,
      ]
      var state = ARMv7A32State()
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[0] == 7)
    }

    @Test("logical operations execute with immediates and registers")
    func logicalOperations() throws {
      let bytes: [UInt8] = [
        0xF0, 0x20, 0xA0, 0xE3, 0x0F, 0x10, 0xA0, 0xE3, 0x02, 0x30, 0x01, 0xE2, 0x01, 0x30, 0x23,
        0xE2, 0x03, 0x40, 0x02, 0xE0, 0x03, 0x50, 0x82, 0xE1, 0x05, 0x60, 0xA0, 0xE1, 0x06, 0x70,
        0xE0, 0xE1, 0x1E, 0xFF, 0x2F, 0xE1,
      ]
      var state = ARMv7A32State()
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[3] == 0x0000_0003)
      #expect(state[4] == 0x0000_0000)
      #expect(state[5] == 0x0000_00F3)
      #expect(state[6] == 0x0000_00F3)
      #expect(state[7] == 0xFFFF_FF0C)
    }

    @Test("arithmetic wraps at 32 bits")
    func wrappingArithmetic() throws {
      let bytes: [UInt8] = [
        0x01, 0x10, 0x80, 0xE2, 0x01, 0x10, 0x41, 0xE2, 0x1E, 0xFF, 0x2F, 0xE1,
      ]
      var state = ARMv7A32State()
      state[0] = 0xFFFF_FFFF
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[1] == 0xFFFF_FFFF)
    }

    @Test("ADDS carry sets Z and C")
    func addsCarry() throws {
      let bytes: [UInt8] = [0x01, 0x10, 0x90, 0xE2, 0x1E, 0xFF, 0x2F, 0xE1]
      var state = ARMv7A32State()
      state[0] = 0xFFFF_FFFF
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[1] == 0x0000_0000)
      #expect(state.cpsr == 0x6000_0000)
    }

    @Test("ADDS signed overflow sets N and V")
    func addsOverflow() throws {
      let bytes: [UInt8] = [0x01, 0x20, 0x90, 0xE2, 0x1E, 0xFF, 0x2F, 0xE1]
      var state = ARMv7A32State()
      state[0] = 0x7FFF_FFFF
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[2] == 0x8000_0000)
      #expect(state.cpsr == 0x9000_0000)
    }

    @Test("SUBS borrow clears C and sets N")
    func subsBorrow() throws {
      let bytes: [UInt8] = [0x01, 0x20, 0x51, 0xE2, 0x1E, 0xFF, 0x2F, 0xE1]
      var state = ARMv7A32State()
      state[1] = 0x0000_0000
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[2] == 0xFFFF_FFFF)
      #expect(state.cpsr == 0x8000_0000)
    }

    @Test("MOVS zero sets Z, non-flag forms preserve CPSR")
    func movsAndPreservation() throws {
      let bytes: [UInt8] = [
        0x00, 0x30, 0xB0, 0xE3, 0x05, 0x40, 0xA0, 0xE3, 0x1E, 0xFF, 0x2F, 0xE1,
      ]
      var state = ARMv7A32State()
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[3] == 0x0000_0000)
      #expect(state[4] == 0x0000_0005)
      #expect(state.cpsr == 0x4000_0000)
    }

    @Test("ANDS preserves carry from earlier ADDS")
    func andsPreservesCarry() throws {
      let bytes: [UInt8] = [
        0x01, 0x10, 0x90, 0xE2, 0x02, 0x40, 0x11, 0xE0, 0x1E, 0xFF, 0x2F, 0xE1,
      ]
      var state = ARMv7A32State()
      state[0] = 0xFFFF_FFFF
      state[1] = 0xFFFF_FFFF
      try ARMv7A32Runner.run(bytes, state: &state)
      #expect(state[1] == 0x0000_0000)
      #expect(state[4] == 0x0000_0000)
      #expect(state.cpsr == 0x6000_0000)
    }

    @Test("truncated final word is rejected")
    func truncated() throws {
      do {
        _ = try ARMv7A32Lifter.lift([0x01, 0x02], pc: 0x8000)
        Issue.record("expected truncated instruction")
      } catch let error as A32DecodeError {
        #expect(error == .truncated(pc: 0x8000, remainingBytes: 2))
      }
    }

    @Test("non-AL instruction is rejected with guest location")
    func conditionRejected() throws {
      do {
        _ = try ARMv7A32Lifter.lift([0x01, 0x00, 0x80, 0x12], pc: 0x4000)
        Issue.record("expected unsupported instruction")
      } catch let error as ARMv7EmulatorError {
        #expect(error == .unsupportedInstruction(pc: 0x4000, word: 0x1280_0001))
      }
    }

    @Test("SVC reports guest PC and immediate")
    func svcRejected() throws {
      do {
        _ = try ARMv7A32Lifter.lift([0x34, 0x12, 0x00, 0xEF], pc: 0x9000)
        Issue.record("expected unsupported syscall")
      } catch let error as ARMv7EmulatorError {
        #expect(error == .unsupportedSyscall(pc: 0x9000, immediate: 0x1234))
      }
    }
  }
#endif
