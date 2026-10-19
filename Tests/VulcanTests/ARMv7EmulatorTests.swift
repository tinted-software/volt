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
