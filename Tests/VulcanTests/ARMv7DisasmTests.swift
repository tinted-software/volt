#if ARMv7Emulator
  import Testing
  import VoltTargetARMv7

  @Suite("ARMv7 disassembler")
  struct ARMv7DisasmTests {
    @Test func commonInstructions() {
      #expect(
        ARMv7Disassembler.disassembleOne(code: [0x01, 0x00, 0x80, 0xE2], at: 0)?.text
          == "add r0, r0, #0x1")
      #expect(
        ARMv7Disassembler.disassembleOne(code: [0xFE, 0xFF, 0xFF, 0xEA], at: 0)?.text == "b +0x0")
      #expect(
        ARMv7Disassembler.disassembleOne(code: [0x1E, 0xFF, 0x2F, 0xE1], at: 0)?.text == "bx lr")
      #expect(
        ARMv7Disassembler.disassembleOne(code: [0xFF, 0xFF, 0xFF, 0xFF], at: 0)?.text
          == ".word 0xffffffff")
      #expect(ARMv7Disassembler.disassembleOne(code: [1, 2, 3], at: 0) == nil)
    }
  }
#endif
