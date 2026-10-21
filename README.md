# Volt

A from-scratch port of Vulcan's JIT to pure Swift (Swift 6.4, x86_64 Linux
host), mirroring the existing `../vulcan/libs/vulcan-target` seam. Ships:

- **AArch64 disassembler** (`VoltTarget` with the `AArch64` trait): the encoder's integer/FP/SIMD
  subset to objdump-style listings; unknown words render as `.word 0x<hex>`.
- **x86_64 JIT codegen** (`VoltTarget` with the `X86_64` trait + `volt-jit`): encoder, System V
  instruction selector, in-memory linker (rel32 call resolution), and W^X
  mapping (`ExecutableMemory`: mmap RW → mprotect R+X).
- **Optional A32 decoder/emulator** (`VoltTarget` with the `ARMv7` trait; `VoltEmulator` +
  `volt-emulate` with `ARMv7Emulator`): `swift run --traits default,ARMv7Emulator volt-emulate --demo`.
  It JIT-lifts non-flag-setting `MOV`/`MVN`/`AND`/`EOR`/`ORR`/`ADD`/`SUB` (immediate and
  unshifted register forms) plus `BX LR`; all 32-bit arithmetic wraps, the remaining
  instructions and `SVC` fail closed.
- **Allocator** (`VoltTarget/LinearScan.swift`): classic linear scan with
  splitting — furthest-ending victim keeps a register prefix, tail spills
  with one store; reloads implicit at use; call/div-spanning values forced to
  slots. No re-homing yet (tails take slots directly).

Out of scope: SPIR-V/GPU, RISC-V, C/Zig frontends, float/vector codegen,
interval holes + re-homing, entry-pin coalescing.

The backend traits are exposed through the single `VoltTarget` product. x86_64 and AArch64
are enabled by default; `ARMv7Emulator` enables ARMv7 and x86_64 as needed.

Build and test with `swift build` / `swift test` (see `Package.swift`). To run all suites with
the emulator enabled, use `swift test --traits default,ARMv7Emulator`.

### Environment note: SwiftBuild linker probe

Swift 6.4's SwiftBuild probes `ld -version_details` expecting Apple ld64 JSON
and fails closed on Linux linkers. A temporary shim answering the probe from
lld's real version unblocks local builds (see vulcan/README.md); `swift
build`/`swift test` also need `--disable-sandbox` here so the probe can exec.
