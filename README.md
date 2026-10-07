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

## Performance

Translating a block costs much more than running it once, and a kernel boot runs most
of its code only a few times. While the vCPU runs a block it has just built, spare cores
build the blocks that statically follow it (same-page branch targets and fall-throughs)
into the shared block cache. By default up to three helper threads are used, leaving one
core for the vCPU; set `VOLT_JIT_THREADS=0` to translate on demand only, or a number to
choose the count. Speculative blocks are keyed by their bytes like any other, so they
never change what the guest executes.
