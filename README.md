# volt

Independent Rust workspace for modular dynamic binary translation.

## Building & Testing

```sh
cargo build --release
cargo test
```

## AArch64 Linux boot

`volt-boot` launches a raw arm64 Linux `Image` with a PL011 console, GICv2,
and optional initramfs or VirtIO block root filesystem.

```sh
cargo run --release --bin volt-boot -- \
  ~/src/linux/arch/arm64/boot/Image \
  --initrd initramfs.cpio.zst \
  20 'console=ttyAMA0 rdinit=/sbin/init nokaslr'
```

`--initrd` (also `--initramfs`) places raw or zstd-compressed images in guest RAM
and publishes `linux,initrd-start` and `linux,initrd-end` in `/chosen`. Use
`--memory 512M` to override the default RAM size if needed.

For a disk-backed root filesystem, pass `--disk` (aliases: `--drive`, `--rootfs`):

```sh
cargo run --release --bin volt-boot -- \
  ~/src/linux/arch/arm64/boot/Image \
  --disk rootfs.ext4 \
  20
```

The disk form supplies `root=/dev/vda rw` by default. `--cpus`, `--idle`,
`--fast`, and `--spin` control the virtual CPU count and WFI policy. Run
`volt-boot --help` for the full CLI.

## XNU boot

`volt-boot` also boots Mach-O XNU kernels (detected automatically by magic):

```sh
cargo run --release --bin volt-boot -- \
  path/to/kernel.development.qemu 20 --fast
```

The loader places Mach-O segments, builds an Apple flattened device tree and
`boot_args`, and enters with MMU off and `x0` pointing at `boot_args` using
the QEMU `virt` layout (PL011 `uart0` at `0x09000000`, GICv3 forwarding the
virtual timer as a FIQ, and 512 KiB of panic log at the top of RAM).

Defaults for XNU are 1 GiB of RAM and `-v serial=3 debug=0x14e keepsyms=1
serial-device-name=uart0`. Custom boot-args must not start with `-`.
`--initrd`, `--disk`, and `--cpus` > 1 are not supported for XNU.

### Current state

- **QEMU XNU kernelcache (`MH_FILESET`):** Boots through VM bootstrap, zone and
  IOKit start-up, MAC policies and BSD init, up to the root-device wait (the same
  point QEMU reaches without a rootfs). Pass the kernelcache, not the bare
  `mach_kernel`.
- **Apple Silicon / `vmapple` XNU:** Loads Mach-O segments, but currently faults
  on unimplemented ARMv8.3 Pointer Authentication instructions (e.g. `xpaci`).
- **Advanced SIMD:** The ARMv8.0 integer and floating-point NEON set is
  implemented in vector and scalar forms and checked bit-for-bit against real
  hardware. Half-precision, crypto, dot-product and complex-number extensions
  are not (the ID registers don't advertise them).
- **Not modelled:** PAN enforcement in translation, physical timer interrupts,
  FP `FPSR` flags / non-nearest rounding, and GICv3 SPIs/SGIs/group 1.

## Performance

`volt-boot` uses up to three helper threads by default to speculatively
translate reachable blocks into the shared cache while the vCPU runs. Set
`VOLT_JIT_THREADS=0` to translate on demand only, or a number to choose the
thread count.
