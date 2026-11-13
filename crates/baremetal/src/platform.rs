//! Minimal `volt-platform` implementations for bare-metal `aarch64-unknown-none` (QEMU `virt`).

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;
use volt_platform::{Clock, Console, ExecImage, ExecMemory, Timer};

const HEAP_SIZE: usize = 16 * 1024 * 1024;
const JIT_ARENA_SIZE: usize = 2 * 1024 * 1024;

#[repr(align(16))]
struct AlignedHeap([u8; HEAP_SIZE]);
static mut HEAP: AlignedHeap = AlignedHeap([0; HEAP_SIZE]);
static HEAP_OFF: AtomicUsize = AtomicUsize::new(0);

pub struct BumpAlloc;

unsafe impl GlobalAlloc for BumpAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align().max(1);
        let mut off = HEAP_OFF.load(Ordering::Relaxed);
        let base = (&raw const HEAP) as usize;
        loop {
            let aligned = (base + off + align - 1) & !(align - 1);
            let next = aligned - base + layout.size();
            if next > HEAP_SIZE {
                return core::ptr::null_mut();
            }
            if HEAP_OFF
                .compare_exchange_weak(off, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return aligned as *mut u8;
            }
            off = HEAP_OFF.load(Ordering::Relaxed);
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static GLOBAL_ALLOC: BumpAlloc = BumpAlloc;

/// PL011 UART console at QEMU `virt` MMIO base `0x0900_0000`.
pub struct UartConsole {
    base: usize,
}

impl UartConsole {
    pub const fn virt_default() -> Self {
        Self { base: 0x0900_0000 }
    }
}

impl Console for UartConsole {
    fn write_byte(&mut self, byte: u8) {
        if byte == b'\n' {
            self.write_byte(b'\r');
        }
        unsafe {
            let dr = self.base as *mut u32;
            let fr = (self.base + 0x18) as *const u32;
            while core::ptr::read_volatile(fr) & (1 << 5) != 0 {}
            core::ptr::write_volatile(dr, u32::from(byte));
        }
    }
}

impl fmt::Write for UartConsole {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_bytes(s.as_bytes());
        Ok(())
    }
}

/// ARM Generic Timer (`CNTVCT_EL0` / `CNTFRQ_EL0`).
#[derive(Clone, Copy, Default)]
pub struct ArmTimer;

impl ArmTimer {
    fn counter() -> u64 {
        let v: u64;
        unsafe {
            core::arch::asm!("mrs {v}, cntvct_el0", v = out(reg) v, options(nomem, nostack));
        }
        v
    }

    fn freq() -> u64 {
        let f: u64;
        unsafe {
            core::arch::asm!("mrs {f}, cntfrq_el0", f = out(reg) f, options(nomem, nostack));
        }
        f.max(1)
    }
}

impl Clock for ArmTimer {
    fn now(&self) -> Duration {
        let cnt = Self::counter();
        let hz = Self::freq();
        let secs = cnt / hz;
        let rem = cnt % hz;
        let nanos = ((rem as u128 * 1_000_000_000) / hz as u128) as u32;
        Duration::new(secs, nanos)
    }
}

impl Timer for ArmTimer {
    fn sleep(&self, duration: Duration) {
        let hz = Self::freq() as u128;
        let ticks = (duration.as_nanos() * hz) / 1_000_000_000;
        let target = Self::counter().saturating_add(ticks as u64);
        while Self::counter() < target {
            core::hint::spin_loop();
        }
    }
}

#[repr(align(4096))]
struct JitArena(UnsafeCell<[u8; JIT_ARENA_SIZE]>);
unsafe impl Sync for JitArena {}

static JIT_ARENA: JitArena = JitArena(UnsafeCell::new([0; JIT_ARENA_SIZE]));
static JIT_OFF: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
pub enum BareExecError {
    Empty,
    OutOfMemory,
}

impl fmt::Display for BareExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("cannot map empty code"),
            Self::OutOfMemory => f.write_str("baremetal JIT arena exhausted"),
        }
    }
}

impl core::error::Error for BareExecError {}

pub struct BareExecImage {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for BareExecImage {}
unsafe impl Sync for BareExecImage {}

impl ExecImage for BareExecImage {
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Executable memory allocator with AArch64 I-cache / D-cache maintenance.
#[derive(Clone, Copy, Default)]
pub struct BareExecMemory;

impl ExecMemory for BareExecMemory {
    type Image = BareExecImage;
    type Error = BareExecError;

    fn map_code(&self, code: &[u8]) -> Result<Self::Image, Self::Error> {
        if code.is_empty() {
            return Err(BareExecError::Empty);
        }
        let aligned_len = (code.len() + 15) & !15;
        let off = JIT_OFF.fetch_add(aligned_len, Ordering::SeqCst);
        if off + aligned_len > JIT_ARENA_SIZE {
            return Err(BareExecError::OutOfMemory);
        }
        let base = JIT_ARENA.0.get() as *mut u8;
        let dst = unsafe { base.add(off) };
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), dst, code.len());
            sync_icache(dst, code.len());
        }
        Ok(BareExecImage {
            ptr: dst,
            len: code.len(),
        })
    }

    fn instruction_barrier(&self) {
        unsafe {
            core::arch::asm!("isb", options(nostack, preserves_flags));
        }
    }
}

unsafe fn sync_icache(start_ptr: *const u8, len: usize) {
    let ctr: u64;
    unsafe {
        core::arch::asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nomem, nostack, preserves_flags));
        let start = start_ptr as usize;
        let end = start + len;
        if ctr & (1 << 28) == 0 {
            let line = 4usize << ((ctr >> 16) & 15);
            let mut at = start & !(line - 1);
            while at < end {
                core::arch::asm!("dc cvau, {at}", at = in(reg) at, options(nostack, preserves_flags));
                at += line;
            }
        }
        core::arch::asm!("dsb ish", options(nostack, preserves_flags));
        if ctr & (1 << 29) == 0 {
            let line = 4usize << (ctr & 15);
            let mut at = start & !(line - 1);
            while at < end {
                core::arch::asm!("ic ivau, {at}", at = in(reg) at, options(nostack, preserves_flags));
                at += line;
            }
            core::arch::asm!("dsb ish", options(nostack, preserves_flags));
        }
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }
}

pub fn psci_system_off() -> ! {
    unsafe {
        core::arch::asm!(
            "movz w0, #0x0008",
            "movk w0, #0x8400, lsl #16",
            "hvc #0",
            options(noreturn)
        );
    }
}
