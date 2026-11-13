# Merging Vantage into Volt: Architecture and Migration Plan

Status: **Migration complete (Phases 0–8 implemented and verified).**

Sources:
- `volt` (this repo, `main`): `volt-ir`, `volt-target`, `volt-emulator`, `volt-boot`.
- `vantage` (`github.com/tinted-software/vantage`, commit `24fa4c5`): `vantage-{raster,shader,codegen,hal,drm,gbm,gles,egl,vulkan}`, plus C headers, ICD and EGL vendor manifests, and examples.

## 0. Decisions and corrections

Decided in this revision:

1. **AMDGPU shaders are compiled at build time, not at runtime.** The runtime loads machine code and loader register values. The compiler (`pliron`-based) runs in `build.rs` or as an offline tool. This makes the runtime graphics path (`volt-shader` runtime, `volt-hal`, the AMDGPU driver) `no_std`-compatible even though `pliron` is not.
2. **`volt-codegen` is multi-target from the start.** The shared compiler pipeline and target trait live in one place. GFX10 is the first implementation. Other GPU generations and possibly CPU backends plug in through the same trait.
3. **The first bare-metal target is `aarch64-unknown-none`.** Bare-metal work is scoped to that target first.

Confirmed in Phase 5: C ABI separated into `volt-gl` (EGL+GBM) and `volt-vulkan` (ICD) over shared `volt-wsi`. See section 4.

Corrections to earlier drafts:

- The macOS `volt-target` failure has a specific cause. The `chunk` field in `native.rs:67` is unconditional, but `mod arena` is `cfg(target_os = "linux")` (line 270). The fix is to gate the field and its uses, or to move the arena behind a platform trait.
- The earlier claim that the rasterizer is "zero-allocation per fragment" was not checked. It is removed and must be verified.

## 1. Design principles

1. **`no_std + alloc` is the default for every library crate.** A crate uses `std` only if it needs OS services. Use `#![cfg_attr(not(feature = "std"), no_std)]` and `extern crate alloc`.
2. **`std` is an opt-in feature that never changes behavior.** It adds host services (file I/O, env, threads, mmap, `eprintln!`).
3. **Host access goes through traits.** Pure logic is `no_std`. Anything that touches the OS or hardware (guest RAM, executable pages, DRM ioctls, DRI3, disk files, console) sits behind a trait with std or OS-specific implementations in a host crate.
4. **Compile-time work and runtime work are separate crates.** Anything that needs `pliron` or other std-only tooling is a build-time compiler. Runtime crates consume its output as data.
5. **Platform gating is explicit.** `target_os` and `target_arch` gates live at module boundaries. A type that is compiled everywhere must not refer to a type that exists only on one platform.
6. **Verify `no_std` in CI.** Workspace feature unification can silently enable `std`, so each crate is checked on its own.

## 2. Crate classification

| Crate | Source | Target | Notes |
| :--- | :--- | :--- | :--- |
| `volt-raster` | vantage-raster | `no_std` now | `std` only for tests. |
| `volt-shader` (runtime: fixed-function fragment, shader blob loading) | vantage-shader | `no_std` now | `libm` for math. |
| `volt-shader-spirv` | vantage-shader `spirv-frontend` | later | Host-side SPIR-V tooling. Check `rspirv` `no_std` support. |
| `volt-codegen` (compiler, multi-target) | vantage-codegen | std-required (`pliron`) | Build-time only. Keep `#![no_std]` in source where it already holds, so it can be enabled when `pliron` is ready. |
| `volt-codegen-gfx10` (GFX10 target) | vantage-codegen `targets/gfx10.rs` | std-required | Implements the target trait. Build-time only. |
| `volt-hal` (software path, runtime) | vantage-hal | `no_std` now | `std::fs` and `eprintln!` in `amdgpu/` are behind the `std` feature. |
| `volt-hal` `amdgpu` (hardware path, runtime) | vantage-hal | `no_std` target, OS-bound | Loads prebuilt machine code. Needs a host or kernel driver for DRM/GEM (see section 5). |
| `volt-gles` | vantage-gles | `no_std` now | `std::error::Error` becomes `core::error::Error`. `std::thread_local!` is replaced by an explicit context, or gated behind `std`. |
| `volt-drm` | vantage-drm | host (Linux, `libc`) | OS-bound by definition. |
| `volt-ir` | volt | std-required | `pliron`. |
| `volt-target` | volt | split | ISel, regalloc (Wimmer), encoders and decoders: `no_std + alloc` once `volt-ir` is. JIT and linking (`native`, `aarch64/jit.rs`, `arena`) go to the host layer. |
| `volt-emulator` | volt | split | CPU core, MMU, bus, GIC, PL011, VirtIO, FDT: `no_std` core. Host adapters go to `volt-host`. |
| `volt-boot` | volt | host binary | Always `std`. |

Currently only `volt-ir` has `#![no_std]` in the Volt tree.

## 3. Target architecture

```
  hosted binaries   volt-boot (std)   examples (std)   bench
                              │
  host adapters (std, OS-bound)
  ┌────────────┬──────────────┬───────────────┬────────────────┐
  │ volt-host  │ volt-drm     │ volt-gl       │ volt-vulkan    │
  │ mmap, exec,│ (Linux)      │ (EGL+GBM C    │ (ICD, separate │
  │ files,     │              │  ABI, §4)     │  .so, §4)      │
  │ console    │              │               │                │
  └─────┬──────┴──────┬───────┴───────┬───────┴────────┬───────┘
        │ traits      │               │                │
  no_std + alloc runtime core
  ┌────────────────────────────────────────────────────────────────┐
  │ volt-platform (traits)   volt-emulator-core   volt-target-core │
  │ volt-hal   volt-gles   volt-raster   volt-shader (runtime)      │
  │ volt-shader-abi (shader blob format, register config)           │
  └────────────────────────────────────────────────────────────────┘
        ▲ data only: compiled shader blobs (no build-time deps at runtime)
  build-time compilers (std)
  ┌────────────────────────────────────────────────────────────────┐
  │ volt-ir (pliron)                                                │
  │ volt-codegen: shared pipeline (isel, regalloc, encode traits)  │
  │   ├── volt-codegen-gfx10     (AMDGPU RDNA2)                    │
  │   └── (future targets: gfx11, other GPUs, ...)                 │
  └────────────────────────────────────────────────────────────────┘
```

Arrows point downward. The runtime core never depends on a host crate or a compiler crate.

### 3.1 `volt-platform`: host service traits

A dependency-free `no_std` crate. It defines what the core needs from its environment:

- `ExecMemory`: allocate, write, make executable, free. Used by the JIT and `native`. Implementations enforce W^X.
- `PhysMap`: guest-physical RAM. The std implementation uses mmap, and a test implementation uses a `Vec`.
- `BlockBackend`: moved from `virtio_blk.rs`.
- `Console`: UART output sink.
- `Clock` and `Timer`: emulated timers.
- `Surface`: presentation target for VirtIO-GPU and EGL.

Implementations live in `volt-host` (std, unix) for hosted builds. The `aarch64-unknown-none` build supplies its own.

### 3.2 Shader blob ABI and the compiler split

This is the key change for runtime `no_std`:

- **`volt-shader-abi`** (`no_std`, no deps): the on-disk and in-memory format for a compiled shader. It holds the stage, the raw machine code, the list of `(register, value)` loader configuration pairs, and metadata such as the interface layout. The format is versioned and has a fixed byte layout.
- **`volt-codegen`** (std): the compiler. It turns `pliron` LLVM-dialect shader IR into a `volt-shader-abi` blob. A target implements the `Target` trait.
- **Build integration:** the shader sources live in the crate that uses them. Their `build.rs` calls `volt-codegen` and writes blobs to `OUT_DIR`, and the runtime `include_bytes!`s them. Shaders can also be compiled once and committed as blobs with a regeneration script, if build-time cost is a concern.

Result: `volt-hal` and `volt-shader` at runtime depend only on `volt-shader-abi`, which is `no_std`. The `pliron` dependency is confined to the build step.

### 3.3 `volt-codegen` multi-target design

- **Shared pipeline** in `volt-codegen`: instruction selection driver, machine IR, a register allocation interface, encoding and listing hooks. The compiler entry point takes a `&impl Target`.
- **Target trait:** the vantage `targets::Target` already has this shape (`expand_pseudo_instructions`, `encode_function`, `format_instruction`, `configuration`). Keep it, and rename it for clarity if needed.
- **Register allocation:** each target provides its own allocator. CPU targets use the existing Wimmer allocator in `volt-target`. GPU targets need SGPR/VGPR banks and pair constraints, so they keep a separate allocator behind a shared interface. Do not force one allocator on both.
- **Relationship to `volt-target`:** `volt-target` already has CPU backends for AArch64 and x86_64. Target selection for GPUs must be consistent with that layer. **Decision (Phase 3):** `volt-codegen` is a sibling crate (`crates/codegen`) for GPU/pliron-LLVM shader compilation with its own `Target` trait (`type Error`, `expand_pseudo_instructions`, `encode_function`, `format_instruction`, `configuration`), and `volt-codegen-gfx10` (`crates/codegen-gfx10`) implements that trait for RDNA2. `volt-target-core` stays separate and `no_std` for CPU DBT targets.
- **Feature gating:** `volt-codegen` has one feature per target (`gfx10`, and later others). Only enabled targets are compiled.

### 3.4 Crate layout (`crates/`)

```
volt/
├── Cargo.toml
├── cbindgen.toml
├── include/                   # EGL, GLES, GBM, KHR headers
├── vulkan/icd.d/              # ICD manifest
├── glvnd/egl_vendor.d/        # EGL vendor manifest
├── docs/
├── crates/
│   ├── ir/                    # volt-ir                         std-required
│   ├── target/                # volt-target (CPU, split core/host)
│   ├── target-core/           # volt-target-core (traits, regalloc iface)  no_std
│   ├── codegen/               # volt-codegen (multi-target compiler)      std
│   ├── codegen-gfx10/         # volt-codegen-gfx10                        std
│   ├── shader-abi/            # volt-shader-abi (blob format)             no_std
│   ├── platform/              # volt-platform (traits)                    no_std
│   ├── host/                  # volt-host (mmap, exec, files, console)    std, unix
│   ├── raster/                # volt-raster                               no_std
│   ├── shader/                # volt-shader (runtime)                     no_std
│   ├── hal/                   # volt-hal                                  no_std core
│   ├── gles/                  # volt-gles                                 no_std core
│   ├── drm/                   # volt-drm                                  std, Linux
│   ├── gl/                    # volt-gl (EGL + GBM C ABI)                 std, C ABI
│   ├── vulkan/                # volt-vulkan (ICD C ABI)                   std, C ABI
│   ├── emulator/              # volt-emulator (core no_std + host adapters)
│   └── boot/                  # volt-boot                                 std binary
└── examples/
```

### 3.5 Feature conventions

- Core library crates default to no features beyond what they need, and are `no_std`.
- Host-only crates (`volt-drm`, `volt-gl`, `volt-vulkan`, `volt-host`, `volt-boot`) default to `std`.
- `volt-codegen` and `volt-codegen-gfx10` are std-only.
- Target selection is by feature: `gfx10` for codegen, `amdgpu` for the runtime hardware path, `aarch64` and `x86_64` for CPU targets, `std` for host services.
- Architecture and OS gating stays at module boundaries.

### 3.6 Workspace lints

Keep `std_instead_of_core = "deny"`. Raise `std_instead_of_alloc` and `alloc_instead_of_core` to `deny` in `no_std` crates, with `allow` in std-only crates. Accidental `std` use then fails to compile.

## 4. C ABI packaging (proposed, pending confirmation)

Merging `egl`, `gbm` and `vulkan` into one library is probably not the best design. The loaders constrain the packaging:

- **Vulkan** must be a separate shared library. The Vulkan loader finds it through an ICD JSON manifest and `dlopen`s it. It also exports the `vk*` symbols, and bundling it with EGL and GBM would pull GL symbols into every Vulkan process.
- **EGL** is loaded through glvnd via the vendor JSON. It is a separate library, and it should not export `vk*` symbols.
- **GBM** is consumed directly by applications, and it is usually linked alongside EGL. It is a good fit with EGL because they share DRM buffers and DRI3 presentation.

Proposed design:

- A shared internal crate `volt-wsi` (`std`, Linux): DRM device handling, GEM buffer management, DRI3 presentation. This is where the real shared logic goes, and it is not exported as a C ABI.
- `volt-gl` as one cdylib/staticlib exporting EGL and GBM together, since they share `volt-wsi`. Keep `libEGL`/`libgbm` names through symlinks or separate thin cdylibs, if consumers require the names.
- `volt-vulkan` as its own cdylib, also built on `volt-wsi` and `volt-hal`.
- `volt-gles` stays `no_std`, and `volt-gl` wraps it with the C ABI.

This keeps the duplication low (one WSI layer) and keeps the loader boundaries intact. The alternative (one fully merged `libvolt` for everything) is simpler to build but causes symbol and loader problems described above. Confirm which direction to take before Phase 4.

## 5. Platform-specific code

| Area | Linux | macOS | `aarch64-unknown-none` |
| :--- | :--- | :--- | :--- |
| Executable pages (JIT) | `volt-host` mmap, arena | `volt-host` mmap, `MAP_JIT`, W^X toggling | platform impl (identity-mapped, MMU setup) |
| Guest RAM | `volt-host` mmap | `volt-host` mmap | platform impl (static or allocator-backed) |
| Disk backend | file | file | platform impl (or none) |
| Console | stdout / PL011 emulation | same | UART |
| DRM / GEM / DRI3 | `volt-drm`, `volt-wsi` | not built | not built |
| AMDGPU hardware | `volt-hal/amdgpu` | not built | not built |
| Software raster and GLES | yes | yes | yes |
| C ABI (`volt-gl`, `volt-vulkan`) | yes | software only, no DRI3 | not built |

Non-Linux hosts build and run the software path. `volt-target`'s `native.rs` is split: the `arena` moves to `volt-host` behind `target_os = "linux"`, and the core depends only on `ExecMemory`.

Bare-metal `aarch64-unknown-none` target plan:

- Target crates: `volt-raster`, `volt-shader`, `volt-hal` (software), `volt-gles`, `volt-shader-abi`, `volt-platform`, `volt-emulator-core`, `volt-target-core`.
- The allocator comes from a `#[global_allocator]` supplied by the bare-metal binary.
- The emulator needs a `volt-platform` implementation for guest RAM, UART, timer, and executable memory. JIT on bare metal requires that the code pages be writable and then executable with I-cache maintenance. The existing `sync_icache` helper in `native/` is a starting point.
- `volt-ir` and `pliron` are not present on the bare-metal target. The bare-metal build uses precompiled shader blobs and precompiled guest code only. The DBT still needs to translate guest code, so the `volt-target` aarch64 backend must be `no_std` in its core and must not depend on `volt-ir` at runtime. This is the main open technical question for the bare-metal target: see section 8.
- Verification: `cargo check --target aarch64-unknown-none --no-default-features` for each crate in the bare-metal list, and a minimal bare-metal test binary that boots and draws one triangle to a dump buffer.

## 6. Phased migration

Each phase has an exit criterion. A phase is complete only when its checks pass.

### Phase 0: Guardrails (no code moves)
- Add `scripts/check-nostd.sh`: runs `cargo check -p <crate> --no-default-features` for each crate in the `no_std` list. Crates that are std-required are on an explicit skip list.
- Add `scripts/check-baremetal.sh`: runs `cargo check --target aarch64-unknown-none --no-default-features` for the bare-metal list. Install the target with `rustup target add aarch64-unknown-none`.
- Add CI for `cargo check --workspace` and `cargo test --workspace` on Linux and macOS.
- Record the baseline. Currently `volt-target` fails on macOS.
- **Exit:** CI runs, and the baseline is recorded.

### Phase 1: Restructure and fix Volt in place
- Move `src/*` to `crates/*` (`ir`, `target`, `emulator`, `boot`). Keep the crate names.
- Fix the macOS build in `volt-target`: gate the `arena` field and its uses.
- Add `#![cfg_attr(not(feature = "std"), no_std)]` to `volt-emulator` and `volt-target`, with `std` as a default feature. Behavior does not change.
- Replace `std::` uses in the core with `core::` or `alloc::` where possible.
- **Exit:** `cargo test --workspace` passes on Linux and macOS.

### Phase 2: `volt-platform`, `volt-host`, `volt-target-core`
- Define the traits in `volt-platform`.
- Move mmap, arena and file backends into `volt-host`.
- Split `volt-target` into `volt-target-core` (no_std) and the host-dependent parts.
- Change the JIT paths and emulator guest RAM and disk to use the traits.
- **Exit:** the emulator boots Linux and XNU with the same output. `volt-emulator-core` and `volt-target-core` pass `--no-default-features` checks on the host.

### Phase 3: Compiler split and shader ABI
- Create `volt-shader-abi` (no_std): the blob format, with versioning and tests that round-trip the format.
- Move the `pliron` compiler into `volt-codegen` (std), with the target trait and the shared pipeline.
- Move the GFX10 implementation into `volt-codegen-gfx10`.
- Decide the `volt-codegen` / `volt-target` relationship (section 3.3). Record the decision in the doc.
- Unify the `pliron` dependency at one revision in `[workspace.dependencies]`, and verify that `volt-ir` and `volt-codegen` use the same version.
- Change the shader sources so that `build.rs` produces blobs. Remove runtime compilation.
- **Exit:** GFX10 shader blobs match the output of the current compiler byte-for-byte, and the runtime crates no longer depend on `pliron`.

### Phase 4: Graphics core (no_std runtime)
- Import `vantage-raster` as `volt-raster`. Keep its `no_std` attribute.
- Import `vantage-shader` as `volt-shader` (runtime). The SPIR-V frontend goes to `volt-shader-spirv` (std).
- Import `vantage-hal` as `volt-hal`. Gate `std::fs` and `eprintln!` behind `std`. The AMDGPU driver uses `volt-shader-abi` blobs, and its host-side DRM access sits behind a trait.
- Import `vantage-gles` as `volt-gles`. Replace `std::error::Error` with `core::error::Error`. Replace `std::thread_local!` with an explicit context, or gate it behind `std`.
- **Exit:** `volt-raster`, `volt-shader`, `volt-hal` (software) and `volt-gles` pass `--no-default-features` on the host and on `aarch64-unknown-none`. The differential raster tests pass.

### Phase 5: Host graphics and C ABI
- Import `vantage-drm` as `volt-drm`.
- Create `volt-wsi` (shared DRM, GEM, DRI3 code).
- Create `volt-gl` (EGL and GBM C ABI) and `volt-vulkan` (ICD), as decided in section 4.
- Gate DRM, GBM and DRI3 on `target_os = "linux"`. The software path builds everywhere.
- Keep the headers, ICD JSON and EGL vendor JSON in the repo, and keep `cbindgen` output reproducible.
- Keep C symbol names unchanged. Add a CI check that compares exported symbols against a recorded list.
- **Exit:** `demo.c` and `dri3_smoke` run on Linux. `volt-gl` and `volt-vulkan` build on macOS with software presentation.

### Phase 6: VirtIO-GPU in the emulator
- Add `virtio_gpu.rs` to the emulator core, using `VIRTIO_ID_GPU` (16). Implement the 2D commands: `GET_DISPLAY_INFO`, `RESOURCE_CREATE_2D`, `RESOURCE_ATTACH_BACKING`, `SET_SCANOUT`, `TRANSFER_TO_HOST_2D`, `RESOURCE_FLUSH`.
- Back resources with `volt-hal` images, using `ExtMem` for guest memory where possible.
- Expose scanout through `volt-platform::Surface`, with host implementations for a dump file, a window, and a null surface.
- Add `--gpu` and `--display` flags to `volt-boot`.
- **Exit:** a Linux guest with the `virtio-gpu` driver draws to the scanout. The device compiles with `--no-default-features`.

### Phase 7: Bare-metal `aarch64-unknown-none`
- Write a minimal platform crate for the target: UART, guest RAM, timer, executable memory with I-cache maintenance, and a global allocator.
- Build a bare-metal binary that runs one guest program through the DBT, and draws one triangle with `volt-hal` (software) into a framebuffer.
- Resolve the `volt-target` runtime dependency on `volt-ir` (section 8, question 1) before this phase.
- **Exit:** the bare-metal binary runs under QEMU `virt` and writes expected output.

### Phase 8: Verification and cleanup
- Run differential raster tests, JIT benchmarks, Linux and XNU boot tests, and the bare-metal test.
- Run both `no_std` check scripts. Update skip lists as `pliron` support lands.
- Remove transitional `#[cfg]` shims and temporary crate names.
- **Exit:** all checks pass on Linux, macOS and `aarch64-unknown-none`, and the `no_std` list matches section 2.

## 7. Risks

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| `pliron` stays std-only | `volt-ir` and `volt-codegen` do not reach bare metal | Compiler is build-time only, so the runtime is not affected. Track `pliron` upstream. |
| `volt-target` aarch64 backend needs `volt-ir` at runtime | Bare-metal DBT blocked | Decide the IR boundary in Phase 2 (section 8). Keep the core ISel and regalloc on a trait or a `no_std` IR. |
| Blob format churn | Build breaks, or stale blobs run | Version `volt-shader-abi`, and have `build.rs` regenerate blobs when `volt-codegen` changes. |
| Multi-target abstraction too early | Wrong trait shape | Build the trait from GFX10 first, and add a second target only when it is needed. Keep the trait small. |
| Feature unification enables `std` | `no_std` breaks silently | Per-crate and per-target checks in CI. |
| `pliron` version split | Two IRs or broken build | One `pliron` revision in `[workspace.dependencies]`, verified in Phase 3. |
| `thread_local!`, `std::fs`, `eprintln!` in shared code | Blocks `no_std` | Explicit context, or gate behind `std`. |
| C symbol or loader breakage | Consumers break | Symbol-list CI check. Keep Vulkan and EGL as separate libraries (section 4). |
| Non-Linux JIT (macOS W^X) | Emulator fails on macOS | `volt-host` implements the W^X toggling and `MAP_JIT` path. Test in Phase 2. |
| Trait dispatch cost | Slower JIT and device paths | Static dispatch in hot paths. `dyn` only at the bus boundary. Benchmark each phase. |

## 8. Resolved design decisions

1. **Bare-metal DBT IR boundary:** Resolved by gating `volt-ir`'s `pliron`-based `dialect` module behind the `std` feature (`default = ["std"]`). `volt-ir` core (`Function`, `types`, `expand`, `softfp`, etc.) is `no_std + alloc`, so `volt-target-core` and `volt-emulator-core` compile and run full AArch64 DBT translation on bare metal (`aarch64-unknown-none`) without `pliron` or `std`.
2. **C ABI packaging:** Confirmed `volt-gl` (EGL + GBM C ABI + GLES re-exports) and `volt-vulkan` (Vulkan ICD) as separate cdylibs sharing `volt-wsi`.
3. **Shader compilation mode:** Confirmed `build.rs` in `volt-shader` compiles all GFX10.3 shaders via `volt-codegen` + `volt-codegen-gfx10` into `volt-shader-abi` blobs embedded via `include_bytes!`. Runtime crates have zero `pliron` dependency.
4. **GPU register allocation:** Kept GPU SGPR/VGPR pair allocator separate in `volt-codegen` (`register_allocation.rs`), while CPU backends keep Wimmer in `volt-target-core`.
