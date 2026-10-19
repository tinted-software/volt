import Foundation
import VoltEmulator

#if canImport(Glibc)
  import Glibc
#elseif canImport(Darwin)
  import Darwin
#endif

@main
struct VoltEmulate {
  static func main() {
    guard CommandLine.arguments.count == 2, CommandLine.arguments[1] == "--demo" else {
      print("usage: volt-emulate --demo")
      exit(2)
    }
    let code: [UInt8] = [0x05, 0x00, 0xA0, 0xE3, 0x07, 0x00, 0x80, 0xE2, 0x1E, 0xFF, 0x2F, 0xE1]
    var state = ARMv7A32State()
    do {
      try ARMv7A32Runner.run(code, state: &state)
      print("r0 = \(state[0])")
    } catch {
      print("volt-emulate: \(error)")
      exit(1)
    }
  }
}
