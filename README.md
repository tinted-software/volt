# volt

Independent Rust workspace for modular dynamic binary translation.

## AArch64 Linux boot

`volt-boot` launches a raw arm64 Linux `Image` with a PL011 console, GICv2,
and optional initramfs or VirtIO block root filesystem.

```sh
cargo run --release --bin volt-boot -- \
  ~/src/linux/arch/arm64/boot/Image \
  --initrd initramfs.cpio.zst \
  20 'console=ttyAMA0 rdinit=/sbin/init nokaslr'
```

`--initrd` (also `--initramfs`) places the image in guest RAM and publishes
`linux,initrd-start` and `linux,initrd-end` in `/chosen`. Raw and zstd-compressed
images are accepted. The default RAM size grows for an initramfs; override it with
`--memory 512M` when needed.

For a disk-backed root filesystem, pass `--disk` (aliases: `--drive`, `--rootfs`):

```sh
cargo run --release --bin volt-boot -- \
  ~/src/linux/arch/arm64/boot/Image \
  --disk rootfs.ext4 \
  20
```

The disk form supplies `root=/dev/vda rw` by default. `--cpus`, `--idle`,
`--fast`, and `--spin` control the virtual CPU count and WFI policy. Use
`volt-boot --help` for the complete generated interface.

## XNU boot

`volt-boot` also boots an XNU kernel. A Mach-O image is detected by its magic, so no
flag is needed:

```sh
cargo run --release --bin volt-boot -- \
  path/to/kernel.development.qemu 20 --fast
```

The loader follows the `bootxnu` U-Boot command: it places the Mach-O segments at a
`physBase` that is congruent to the link-time `virtBase` modulo a 32 MiB L2 block,
builds an Apple flattened device tree (the equivalent of `mkafdt.py`) and a `boot_args`
block, and enters at the kernel entry with the MMU off and `x0` pointing at
`boot_args`. The machine is the QEMU `virt` layout XNU's QEMU platform expects:
PL011 `uart0` at `0x09000000`, and a GICv3 (distributor `0x08000000`, redistributor
`0x080a0000`, `ICC_*` system registers) that forwards the virtual timer as a FIQ. The
last 512 KiB of RAM is the panic log (`chosen/pram`), and `memSize` excludes it.

Defaults for a Mach-O kernel are 1 GiB of RAM and the boot-args
`-v serial=3 debug=0x14e keepsyms=1 serial-device-name=uart0`. To pass your own, the
first word must not start with a dash (`serial=3 -v ...`), or it is read as a flag.
`--initrd`, `--disk` and `--cpus` above 1 are rejected for XNU.

The bare kernel boots through the pmap and VM bootstrap, zone and IOKit start-up, the
scheduler and IONVRAM, and then panics in `read_random`: the corecrypto kext that
registers the kernel PRNG is not part of the kernel image. That is the kernel's own
behavior on any machine, and the panic message and backtrace appear on the console.
Going further needs a kernelcache with corecrypto prelinked.

Not modelled: privileged-access-never is stored and saved in `SPSR_EL1` but not
enforced by translation, the physical timer is stored but never fires, floating point
rounds to nearest even and sets no `FPSR` flags, and the GICv3 carries only the
virtual timer (no SPIs, SGIs or group 1).

## Apple M1 (t8103) SoC

`--dtb` selects the M1 machine in place of the QEMU `virt` board. The device tree is
the Asahi kernel's board tree, which the kernel tree builds as
`arch/arm64/boot/dts/apple/t8103-j274.dtb`:

```sh
cargo run --release --bin volt-boot -- \
  ~/src/linux/arch/arm64/boot/Image 60 \
  --dtb ~/src/linux/arch/arm64/boot/dts/apple/t8103-j274.dtb
```

The SoC addresses come from the M1 Mac mini's own device tree (`DeviceTree.j274ap`
in a macOS restore image; `ipsw dtree` reads it). DRAM is at `0x8_0000_0000`, the
interrupt controller (AIC) at `0x2_3b10_0000`, the power-state blocks at
`0x2_3b70_0000` and `0x2_3d28_0000`, and the console (`apple,s5l-uart`, `stdout-path`)
at `0x2_3520_0000`. The machine is one vCPU with the M1 Icestorm identity
(`MIDR_EL1` 0x610f0220) and a 16 KiB granule, which the Asahi kernel requires.

The Linux kernel boots through the AIC and the S5L console, probes its power domains,
and panics on the missing root filesystem (`VFS: Unable to mount root fs`), which is
the expected result without one. `--initrd`, `--disk` and `--smp` above 1 are rejected
on this board. A kernelcache from the restore image (`MH_FILESET`) loads, with its ADT
as `--dtb`, and enters at its reset trampoline with the cold-boot reset type. It runs
several hundred blocks of early boot, then executes `genter` (guarded execution) before
it has set `VBAR_EL1`. The kernel expects iBoot to have enabled GXF and set its monitor
entry, which volt does not provide; chained fixups are still not applied.

The EL2 regime is modelled: `CurrentEl` reports EL2, exceptions and `ERET` target
`VBAR_EL2`/`ELR_EL2`/`SPSR_EL2`, translation walks `TTBR0_EL2`/`TCR_EL2`, and
`HCR_EL2.E2H` (VHE) redirects EL1 register names accessed from EL2 to their EL2 banks.
`system::monitor` loads Apple's SPTM monitor (`sptm.t8103.release`) the way the boot
firmware does: the image placed in DRAM at its physical address, the MMU off, EL2, and
`x0` a boot structure (DRAM's virtual and physical base and size, the device tree, a
scratch buffer). The monitor builds its own translation tables, turns the MMU on, and
checks what iBoot would have recorded in `chosen`: the `chosen/memory-map` regions, and
`dram-base` and `dram-size`. Its strings suggest it applies the kernelcache's chained
fixups itself (`sptm_fixup`); that is inferred, not confirmed.

The region table is walked in a fixed order, and each present region must start where the
previous present one ended. In order: `TXM-ro`, `TXM-rx`, `TXM-bx`, `TrustCache`,
`AuxKC-ro/rx` (optional), `BootKC-rx`, `BootKC-bx`, `BootKC-ro`, `BootKC-rs`,
`CL4-rx/ro` (optional), `DeviceTree`, `SPTM-ro`, `SPTM-rx`, `SPTM-rw`. Each part is a
contiguous link span (`monitor::image_parts`), copied whole (`monitor::place_parts`). The
monitor's own image stays linear, because it runs position-independently before its MMU
is on. With that packing, validation passes and the monitor reaches `/chosen/dram-base`.

It then stops in a data abort. It zeroes a page at `VA & 0x3fff_ffff_ffff` of a linear-map
address, so its convention for converting a linear-map address to a physical one is not
the one volt's boot structure implies (`VA = PA + x22 - x23`). Which convention it uses, and
where iBoot places DRAM relative to the image, is not yet established. GXF (`genter`,
`gexit`, `GXF_*`, `VBAR_GL1`) and SPRR are not modelled; `genter` raises an undefined
instruction.

Two debugging hooks make this tractable. `Machine::set_step_trace` runs one instruction
per block and reports each step with the system registers it changed, exception entries
included. `Machine::set_memory_trace` reports every translated load and store (atomics
and SIMD structure accesses excepted). Installing either turns the inline data TLB off,
so a traced run is much slower than an untraced one.

Not modelled: any arm-io device other than the AIC, the UART and the power-state
blocks. Those addresses read as zero and writes are dropped, and the boot report counts
them as `unmodeled device accesses`, so driver probes of missing hardware (DART, GPU,
SMC, PCIe) fail rather than fault. Secondary cores stay powered off (the device tree
gives them no release address), the physical timer never fires, and fast IPIs are
storage only.

## Performance

Translating a block costs much more than running it once, and a kernel boot runs most
of its code only a few times. While the vCPU runs a block it has just built, spare cores
build the blocks that statically follow it (same-page branch targets and fall-throughs)
into the shared block cache. By default up to three helper threads are used, leaving one
core for the vCPU; set `VOLT_JIT_THREADS=0` to translate on demand only, or a number to
choose the count. Speculative blocks are keyed by their bytes like any other, so they
never change what the guest executes.
