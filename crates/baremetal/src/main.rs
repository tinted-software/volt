//! Bare-metal `aarch64-unknown-none` smoke binary for QEMU `virt`.
//!
//! 1. Drops from EL2 to EL1 (if needed) and enables a 1:1 identity-mapped MMU
//!    (Normal WB cacheable RAM + Device-nGnRnE MMIO) + FP/NEON.
//! 2. Runs an AArch64 guest program through `volt-emulator-core`'s DBT.
//! 3. Draws a triangle with `volt-hal` (software) into a 64x64 BGRA framebuffer.
//! 4. Reports results over PL011 UART and shuts down QEMU via PSCI.

#![no_std]
#![no_main]

extern crate alloc;

mod platform;

use alloc::vec;
use core::fmt::Write;
use core::panic::PanicInfo;
use platform::{ArmTimer, BareExecMemory, UartConsole, psci_system_off};
use volt_emulator_core::devices::{Gicv2, Pl011};
use volt_emulator_core::memory::{GuestMemory, PhysicalMemory, Region};
use volt_emulator_core::system::{Exit, Machine};
use volt_hal::{Cmd, CommandBuffer, Device, Format, Pipeline, Vertex};
use volt_platform::Clock;
use volt_raster::gl;

#[repr(C, align(4096))]
struct PageTable([u64; 512]);

static mut L1_TABLE: PageTable = PageTable([0; 512]);

core::arch::global_asm!(
    r#"
.section .text.boot, "ax"
.global _start
_start:
    // Drop from EL2 to EL1 if QEMU booted us at EL2.
    mrs x0, CurrentEL
    and x0, x0, #12
    cmp x0, #8
    b.ne 1f
    mrs x0, hcr_el2
    orr x0, x0, #(1 << 31) // RW = 1 (AArch64 EL1)
    msr hcr_el2, x0
    mov x0, #3
    msr cnthctl_el2, x0
    msr cntvoff_el2, xzr
    mov x0, #0x3c5         // EL1h, DAIF masked
    msr spsr_el2, x0
    adr x0, 1f
    msr elr_el2, x0
    eret
1:
    // Enable FP/SIMD (CPACR_EL1.FPEN = 0b11).
    mov x0, #(3 << 20)
    msr cpacr_el1, x0
    isb

    // Zero .bss.
    ldr x0, =__bss_start
    ldr x1, =__bss_end
2:
    cmp x0, x1
    b.hs 3f
    stp xzr, xzr, [x0], #16
    b 2b
3:
    // Set stack pointer.
    ldr x0, =__stack_top
    mov sp, x0

    // Jump to Rust entry.
    bl rust_main
4:
    wfe
    b 4b
"#
);

unsafe fn enable_identity_mmu() {
    // MAIR: attr0 = Device-nGnRnE (0x00), attr1 = Normal WBWA (0xff)
    let mair: u64 = 0x0000_0000_0000_ff00;
    // TCR: T0SZ = 25 (39-bit VA -> page walk starts at L1, where 1 GiB blocks are valid),
    // 4KB granule (TG0=0b00), inner shareable (SH0=0b11), WBWA (ORGN0=0b01, IRGN0=0b01).
    let tcr: u64 = 25 | (0b01 << 8) | (0b01 << 10) | (0b11 << 12);

    unsafe {
        let tt = &raw mut L1_TABLE.0;
        // 1 GiB blocks at L1:
        // Entry 0: 0x0000_0000..0x4000_0000 -> Device-nGnRnE (attr0, AF=1, PXN/UXN=1)
        (*tt)[0] = 0x0000_0000 | (1 << 10) | 1 | (1 << 53) | (1 << 54);
        // Entry 1: 0x4000_0000..0x8000_0000 -> Normal WBWA (attr1, AF=1, SH=IS, executable)
        (*tt)[1] = 0x4000_0000 | (1 << 10) | (1 << 2) | (0b11 << 8) | 1;

        let ttbr0 = tt as usize as u64;
        core::arch::asm!(
            "msr mair_el1, {mair}",
            "msr tcr_el1, {tcr}",
            "msr ttbr0_el1, {ttbr0}",
            "isb",
            "mrs {sctlr}, sctlr_el1",
            "orr {sctlr}, {sctlr}, #1",       // M = 1
            "orr {sctlr}, {sctlr}, #(1 << 2)", // C = 1
            "orr {sctlr}, {sctlr}, #(1 << 12)",// I = 1
            "bic {sctlr}, {sctlr}, #(1 << 1)", // A = 0 (allow unaligned)
            "msr sctlr_el1, {sctlr}",
            "isb",
            mair = in(reg) mair,
            tcr = in(reg) tcr,
            ttbr0 = in(reg) ttbr0,
            sctlr = out(reg) _,
        );
    }
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    let mut uart = UartConsole::virt_default();
    let _ = writeln!(uart, "volt-baremetal: enabling MMU...");
    unsafe { enable_identity_mmu() };
    let timer = ArmTimer;
    let t0 = timer.now();

    let _ = writeln!(uart, "volt-baremetal: booted on aarch64-unknown-none");

    // 1. Run a guest program through the DBT (adds 19 + 23 -> x0, prints "OK", HVC shutdown).
    let guest_code: [u32; 8] = [
        0xd2800260, // movz x0, #19
        0x91005c00, // add  x0, x0, #23   -> x0 = 42
        0xd2a12001, // movz x1, #0x900, lsl #16 (PL011)
        0x528009e2, // movz w2, #'O'
        0x39000022, // strb w2, [x1]
        0x52800962, // movz w2, #'K'
        0x39000022, // strb w2, [x1]
        0xd4000002, // hvc #0
    ];
    let mut mem = PhysicalMemory::new(vec![Region::ram(0x4000_0000, vec![0; 0x10000]).unwrap()])
        .expect("physical memory");
    for (i, word) in guest_code.iter().enumerate() {
        mem.write(0x4000_0000 + (i as u64) * 4, &word.to_le_bytes())
            .unwrap();
    }

    let mut machine = Machine::new_with(mem, BareExecMemory, true);
    machine.cpu.pc = 0x4000_0000;
    static GUEST_LEN: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
    static GUEST_BUF: [core::sync::atomic::AtomicU8; 16] =
        [const { core::sync::atomic::AtomicU8::new(0) }; 16];
    {
        let mut serial = Pl011::new(|b| {
            let idx = GUEST_LEN.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            if idx < GUEST_BUF.len() {
                GUEST_BUF[idx].store(b, core::sync::atomic::Ordering::Relaxed);
            }
        });
        let mut gic = Gicv2::default();
        loop {
            match machine.run(&mut serial, &mut gic).expect("DBT run") {
                Exit::Interrupted | Exit::Timer => continue,
                Exit::Psci { .. } => break,
                other => panic!("unexpected DBT exit: {other:?}"),
            }
        }
    }
    assert_eq!(machine.cpu.x[0], 42);
    assert_eq!(GUEST_LEN.load(core::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(GUEST_BUF[0].load(core::sync::atomic::Ordering::Relaxed), b'O');
    assert_eq!(GUEST_BUF[1].load(core::sync::atomic::Ordering::Relaxed), b'K');
    let _ = writeln!(
        uart,
        "volt-baremetal: DBT OK (x0={}, serial=OK)",
        machine.cpu.x[0]
    );

    // 2. Render one green triangle with volt-hal (software) into a 64x64 BGRA framebuffer.
    let mut dev = Device::new();
    let fb = dev.create_image(Format::B8G8R8A8Unorm, 64, 64);
    let pipeline = Pipeline {
        topology: 0,
        cull_mode: 0,
        front_face_ccw: true,
        blend_enabled: false,
        src_factor: 1,
        dst_factor: 0,
        depth_test: false,
        depth_write: false,
        depth_func: gl::ALWAYS,
        color_mask: 0x0f,
    };
    let pipe = dev.create_pipeline(pipeline.clone());
    let v0 = Vertex {
        pos: [-0.8, -0.8, 0.5, 1.0],
        color: [0.0, 1.0, 0.0, 1.0],
        tex0: [0.0, 0.0],
        tex1: [0.0, 0.0],
        fog: 0.0,
        _pad: 0.0,
    };
    let v1 = Vertex {
        pos: [0.8, -0.8, 0.5, 1.0],
        color: [0.0, 1.0, 0.0, 1.0],
        tex0: [0.0, 0.0],
        tex1: [0.0, 0.0],
        fog: 0.0,
        _pad: 0.0,
    };
    let v2 = Vertex {
        pos: [0.0, 0.8, 0.5, 1.0],
        color: [0.0, 1.0, 0.0, 1.0],
        tex0: [0.0, 0.0],
        tex1: [0.0, 0.0],
        fog: 0.0,
        _pad: 0.0,
    };
    let mut cmd = CommandBuffer::default();
    cmd.push(Cmd::BindPipeline { pipeline: pipe });
    cmd.push(Cmd::BindAttachments {
        color: Some(fb),
        depth: None,
        stencil: None,
    });
    cmd.push(Cmd::ClearAttachments {
        color: [0.0, 0.0, 0.0, 1.0],
        depth: 1.0,
        stencil: 0,
        mask: 1,
    });
    cmd.push(Cmd::SetViewport {
        x: 0,
        y: 0,
        w: 64,
        h: 64,
    });
    cmd.push(Cmd::DrawMesh {
        vertices: vec![v0, v1, v2],
        indices: None,
        pipeline,
    });
    dev.submit(&cmd);

    let img = dev.image(fb).expect("framebuffer");
    let px = img.slice();
    let corner = &px[0..4];
    let center_off = (32 * 64 + 32) * 4;
    let center = &px[center_off..center_off + 4];
    assert_eq!(corner, [0, 0, 0, 255]);
    assert_eq!(center, [0, 255, 0, 255]);

    let elapsed_us = timer.now().saturating_sub(t0).as_micros();
    let _ = writeln!(
        uart,
        "volt-baremetal: HAL triangle OK (corner={:02x}{:02x}{:02x}{:02x}, center={:02x}{:02x}{:02x}{:02x}, {}us)",
        corner[0],
        corner[1],
        corner[2],
        corner[3],
        center[0],
        center[1],
        center[2],
        center[3],
        elapsed_us
    );

    psci_system_off();
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    let mut uart = UartConsole::virt_default();
    let _ = writeln!(uart, "volt-baremetal PANIC: {info}");
    psci_system_off();
}
