# Volt

A from-scratch port of Vulcan's JIT to pure Swift (Swift 6.4, x86_64 Linux
host), mirroring the existing `../vulcan/libs/vulcan-target` seam. Ships:

- **AArch64 disassembler** (`VoltTargetAArch64`): the encoder's integer/FP/SIMD
  subset to objdump-style listings; unknown words render as `.word 0x<hex>`.
- **x86_64 JIT codegen** (`VoltTargetX86_64` + `volt-jit`): encoder, System V
  instruction selector, in-memory linker (rel32 call resolution), and W^X
  mapping (`ExecutableMemory`: mmap RW → mprotect R+X).
- **Allocator** (`VoltTarget/LinearScan.swift`): classic linear scan with
  splitting — furthest-ending victim keeps a register prefix, tail spills
  with one store; reloads implicit at use; call/div-spanning values forced to
  slots. No re-homing yet (tails take slots directly).

Out of scope: SPIR-V/GPU, RISC-V, C/Zig frontends, float/vector codegen,
interval holes + re-homing, entry-pin coalescing.

Build and test with `swift build` / `swift test` (see `Package.swift`).

### Environment note: SwiftBuild linker probe

Swift 6.4's SwiftBuild probes `ld -version_details` expecting Apple ld64 JSON
and fails closed on Linux linkers. A temporary shim answering the probe from
lld's real version unblocks local builds (see vulcan/README.md); `swift
build`/`swift test` also need `--disable-sandbox` here so the probe can exec.
