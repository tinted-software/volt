//! AArch64 Linux and XNU full-system execution on the Volt native JIT.
//! Ports Mirage's JIT Machine, raw Image launch and one-vCPU serial/GIC runner; a
//! Mach-O kernel boots through [`xnu`] on a GICv3 instead of the Linux GICv2.

pub mod afdt;
pub mod boot;
pub mod esr;
pub mod fdt;
pub mod machine;
pub mod psci;
pub mod smp;
pub mod xnu;

use crate::aarch64::{Cpu, decode, translate};
use crate::devices::bus::Device;
use crate::devices::{gicv2::Gicv2, gicv3::Gicv3, pl011::Pl011};
use crate::memory::{GuestMemory, PhysicalMemory, Region, SharedMemory};
use alloc::sync::Arc;
use core::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use parking_lot::Mutex;
use std::{io::Write, time::Instant};

pub const RAM_BASE: u64 = 0x4000_0000;
pub const RAM_SIZE: usize = 128 << 20;
pub const DEFAULT_CMDLINE: &str = "console=ttyAMA0 earlycon=pl011,0x9000000 nokaslr loglevel=8";
/// Boot arguments for an XNU kernel: verbose boot, console and debug output on the
/// PL011 that the device tree describes as `uart0`.
pub const DEFAULT_XNU_CMDLINE: &str = "-v serial=3 debug=0x14e keepsyms=1 serial-device-name=uart0";
/// XNU sizes its zones from RAM; 128 MiB leaves it little to work with.
pub const XNU_RAM_SIZE: usize = 1 << 30;

#[derive(Debug)]
pub enum Error {
    Boot(boot::Error),
    Memory(crate::memory::MemoryError),
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl core::error::Error for Error {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Shutdown,
    Reset,
    Timeout,
    LostStack,
    Halt,
    Fault(String),
}

/// CPU idle policy when the guest issues `WFI` (Wait For Interrupt).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IdleMode {
    /// Advance virtual time to the next timer deadline and sleep the host thread
    /// to pace execution with real time and conserve CPU/battery.
    #[default]
    Paced,
    /// Fast-forward virtual time directly to the timer deadline without sleeping.
    /// Ideal for headless batch testing, benchmarks, and CI.
    FastForward,
    /// Busy-spin without advancing virtual timer or sleeping (legacy behavior).
    Spin,
}
#[derive(Debug)]
pub struct InstructionDiagnostic {
    pub virtual_address: u64,
    pub physical_address: u64,
    pub word: u32,
    pub decoded: bool,
}
#[derive(Debug)]
pub struct WalkEntry {
    pub level: u8,
    pub table: u64,
    pub index: u64,
    pub descriptor: Option<u64>,
}
#[derive(Debug)]
pub struct BootReport {
    pub reason: StopReason,
    pub layout: boot::Layout,
    pub cpu: Cpu,
    pub console: Vec<u8>,
    pub exits: u64,
    pub compiled_blocks: usize,
    pub executed_blocks: u64,
    pub timer_interrupts: u64,
    pub unmapped_accesses: u64,
    pub stalled_at: Option<u64>,
    pub instruction: Option<InstructionDiagnostic>,
    pub last_fault_walk: Vec<WalkEntry>,
    pub trace: Vec<machine::TraceEntry>,
    pub elapsed: Duration,
}
impl core::fmt::Display for BootReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "--- stopped: {:?}", self.reason)?;
        writeln!(
            f,
            "entry {:x}, boot info {:x}",
            self.layout.entry, self.layout.boot_info
        )?;
        if let Some(at) = self.stalled_at {
            writeln!(f, "refused at {at:x}")?;
        }
        let c = &self.cpu;
        let s = &c.system;
        writeln!(
            f,
            "pc {:x}  sp {:x}  el {}  flags {:x}  trap {:?}",
            c.pc, c.sp, s.el, c.flags, c.trap
        )?;
        writeln!(
            f,
            "esr {:x}  elr {:x}  far {:x}  spsr {:x}  sctlr {:x}",
            s.esr_el1, s.elr_el1, s.far_el1, s.spsr_el1, s.sctlr_el1
        )?;
        writeln!(
            f,
            "ttbr0 {:x}  ttbr1 {:x}  tcr {:x}  mair {:x}",
            s.ttbr0_el1, s.ttbr1_el1, s.tcr_el1, s.mair_el1
        )?;
        writeln!(
            f,
            "timer counter {}  compare {}  ctl {:x}",
            s.cntvct_el0, s.cntv_cval_el0, s.cntv_ctl_el0
        )?;
        for (index, value) in c.x.iter().enumerate() {
            write!(f, "x{index:<2} {value:016x}  ")?;
            if index % 4 == 3 {
                writeln!(f)?;
            }
        }
        writeln!(f)?;
        if let Some(i) = &self.instruction {
            writeln!(
                f,
                "{}word at {:x} (physical {:x}): {:08x}",
                if i.decoded { "" } else { "first undecoded " },
                i.virtual_address,
                i.physical_address,
                i.word
            )?;
        }
        if !self.last_fault_walk.is_empty() {
            writeln!(f, "page tables for last fault at {:x}:", s.far_el1)?;
        }
        for entry in &self.last_fault_walk {
            match entry.descriptor {
                Some(descriptor) => writeln!(
                    f,
                    "  level {}: table {:x}  index {:x}  entry {:016x}",
                    entry.level, entry.table, entry.index, descriptor
                )?,
                None => writeln!(
                    f,
                    "  level {}: table {:x} is outside memory",
                    entry.level, entry.table
                )?,
            }
        }
        writeln!(f, "last blocks, oldest first:")?;
        for entry in &self.trace {
            writeln!(f, "  pc {:x}  sp {:x}", entry.pc, entry.sp)?;
        }
        writeln!(
            f,
            "exits {}, blocks compiled {}, blocks executed {}, timer interrupts {}, unmapped accesses {}, elapsed {:.3}s",
            self.exits,
            self.compiled_blocks,
            self.executed_blocks,
            self.timer_interrupts,
            self.unmapped_accesses,
            self.elapsed.as_secs_f64()
        )
    }
}
#[derive(Debug, Clone)]
pub struct SystemConfig<B = Vec<u8>> {
    pub timeout: Duration,
    pub cmdline: String,
    pub idle_mode: IdleMode,
    pub ram_size: usize,
    pub initrd: Option<Vec<u8>>,
    pub disk: Option<B>,
}

impl<B> Default for SystemConfig<B> {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(20),
            cmdline: DEFAULT_CMDLINE.into(),
            idle_mode: IdleMode::default(),
            ram_size: RAM_SIZE,
            initrd: None,
            disk: None,
        }
    }
}

/// The interrupt controller a boot configures: the GICv2 a Linux device tree
/// describes, or the GICv3 XNU's platform expects. Shared peripherals (UART, virtio)
/// are wired to the GICv2 only; the GICv3 carries the virtual timer, which the
/// machine evaluates live from the timer registers.
enum Intc {
    V2(Gicv2),
    V3(Gicv3),
}
impl Intc {
    fn raise(&mut self, intid: u32) {
        if let Self::V2(gic) = self {
            gic.raise(intid);
        }
    }
    fn lower(&mut self, intid: u32) {
        if let Self::V2(gic) = self {
            gic.lower(intid);
        }
    }
    fn raise_on(&mut self, cpu: u32, intid: u32) {
        if let Self::V2(gic) = self {
            gic.raise_on(cpu, intid);
        }
    }
    fn signalled(&self, cpu: u32) -> bool {
        match self {
            Self::V2(gic) => gic.signalled(cpu),
            Self::V3(_) => false,
        }
    }
    /// See [`machine::Machine::vtimer_fiq`].
    fn vtimer_fiq(&self) -> Option<u8> {
        match self {
            Self::V2(_) => None,
            Self::V3(gic) => gic.vtimer_fiq_priority(),
        }
    }
}

struct SystemDevices<'a, V> {
    serial: &'a mut Pl011,
    gic: &'a mut Intc,
    virtio: Option<&'a mut V>,
}

impl<V: crate::devices::VirtioIo> machine::DeviceIo for SystemDevices<'_, V> {
    fn read_uart(&mut self, offset: u64, size: u8) -> u64 {
        self.serial.read(offset, size)
    }
    fn write_uart(&mut self, offset: u64, size: u8, value: u64) {
        self.serial.write(offset, size, value);
    }
    fn read_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        match self.gic {
            Intc::V2(gic) => gic.read_distributor_for(cpu_id, offset, size),
            Intc::V3(gic) => gic.read_distributor(offset, size),
        }
    }
    fn write_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        match self.gic {
            Intc::V2(gic) => {
                gic.write_distributor_for(cpu_id, offset, size, value);
            }
            Intc::V3(gic) => gic.write_distributor(offset, size, value),
        }
    }
    fn read_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        match self.gic {
            Intc::V2(gic) => gic.read_cpu_for(cpu_id, offset, size),
            // The GICv3 CPU interface is the `ICC_*` system registers.
            Intc::V3(_) => 0,
        }
    }
    fn write_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        if let Intc::V2(gic) = self.gic {
            gic.write_cpu_for(cpu_id, offset, size, value);
        }
    }
    fn read_gic_redistributor(&mut self, offset: u64, size: u8) -> u64 {
        match self.gic {
            Intc::V3(gic) => gic.read_redistributor(offset, size),
            Intc::V2(_) => 0,
        }
    }
    fn write_gic_redistributor(&mut self, offset: u64, size: u8, value: u64) {
        if let Intc::V3(gic) = self.gic {
            gic.write_redistributor(offset, size, value);
        }
    }
    fn read_virtio(&mut self, offset: u64, size: u8) -> u64 {
        if let Some(v) = &mut self.virtio {
            v.read(offset, size)
        } else {
            0
        }
    }
    fn write_virtio(&mut self, offset: u64, size: u8, value: u64) {
        if let Some(v) = &mut self.virtio {
            v.write(offset, size, value);
        }
    }
}

/// Boot a raw Linux Image with live serial output on stdout and a bounded wall-clock run.
/// Execution faults are returned in the report with the full guest state, not discarded.
pub fn boot(image: &[u8], timeout: Duration, cmdline: &str) -> Result<BootReport, Error> {
    boot_options(image, timeout, cmdline, IdleMode::default())
}

/// Same as [`boot`], with a selectable [`IdleMode`].
pub fn boot_options(
    image: &[u8],
    timeout: Duration,
    cmdline: &str,
    idle_mode: IdleMode,
) -> Result<BootReport, Error> {
    let mut output = std::io::BufWriter::new(std::io::stdout());
    boot_with_options(image, timeout, cmdline, idle_mode, move |byte| {
        // The complete output also lives in BootReport if a host console cannot accept it.
        let _ = output.write_all(&[byte]);
        if byte == b'\n' {
            let _ = output.flush();
        }
    })
}

/// Same machine as [`boot`], with an application-owned serial destination.
pub fn boot_with_serial(
    image: &[u8],
    timeout: Duration,
    cmdline: &str,
    sink: impl FnMut(u8) + Send + 'static,
) -> Result<BootReport, Error> {
    boot_with_options(image, timeout, cmdline, IdleMode::default(), sink)
}

/// Same machine as [`boot_with_serial`], with a selectable [`IdleMode`].
pub fn boot_with_options(
    image: &[u8],
    timeout: Duration,
    cmdline: &str,
    idle_mode: IdleMode,
    sink: impl FnMut(u8) + Send + 'static,
) -> Result<BootReport, Error> {
    boot_system(
        image,
        SystemConfig::<Vec<u8>> {
            timeout,
            cmdline: cmdline.to_string(),
            idle_mode,
            ram_size: RAM_SIZE,
            initrd: None,
            disk: None,
        },
        sink,
    )
}

/// Boot with full system configuration including optional initrd and configurable RAM size.
pub fn boot_system<B: crate::devices::BlockBackend>(
    image: &[u8],
    config: SystemConfig<B>,
    mut sink: impl FnMut(u8) + Send + 'static,
) -> Result<BootReport, Error> {
    let ram_size = if config.ram_size > 0 {
        config.ram_size
    } else {
        RAM_SIZE
    };
    let timeout = config.timeout;
    let idle_mode = config.idle_mode;
    let region = Region::ram(RAM_BASE, vec![0; ram_size]).map_err(Error::Memory)?;
    let memory = PhysicalMemory::new(vec![region]).map_err(Error::Memory)?;
    let mut shared_mem = SharedMemory::new(memory);
    let has_virtio = config.disk.is_some();
    let is_xnu = xnu::is_macho(image);
    let layout = if is_xnu {
        if config.initrd.is_some() || has_virtio {
            return Err(Error::Boot(boot::Error::Unsupported(
                "XNU boot has no initrd or virtio disk support",
            )));
        }
        xnu::prepare_boot(
            &mut shared_mem,
            image,
            &config.cmdline,
            RAM_BASE,
            ram_size as u64,
        )
    } else {
        boot::prepare_boot(
            &mut shared_mem,
            image,
            config.initrd.as_deref(),
            &config.cmdline,
            RAM_BASE,
            ram_size as u64,
            1,
            has_virtio,
        )
    }
    .map_err(Error::Boot)?;
    let console = Arc::new(Mutex::new(Vec::new()));
    let serial_console = console.clone();
    let uart_line = Arc::new(AtomicBool::new(false));
    let serial_line = uart_line.clone();
    let mut serial = Pl011::new(move |byte| {
        serial_console.lock().push(byte);
        sink(byte);
    })
    .with_line(move |level| serial_line.store(level, Ordering::Release));
    let mut gic = if is_xnu {
        Intc::V3(Gicv3::default())
    } else {
        Intc::V2(Gicv2::default())
    };
    let virtio_line = Arc::new(AtomicBool::new(false));
    let virtio_notify = virtio_line.clone();
    let mut virtio = config.disk.map(|backend| {
        crate::devices::VirtioBlock::new(backend, shared_mem.clone())
            .with_line(move |level| virtio_notify.store(level, Ordering::Release))
    });
    let mut machine = machine::Machine::new(shared_mem);
    machine.cpu.pc = layout.entry;
    machine.cpu.x[0] = layout.boot_info;
    machine.cpu.x[1..4].fill(0);
    let started = Instant::now();
    let mut exits = 0;
    let mut timer_interrupts = 0;
    let reason = loop {
        if started.elapsed() >= timeout {
            break StopReason::Timeout;
        }
        if uart_line.load(Ordering::Acquire) {
            gic.raise(fdt::UART_INTID);
        } else {
            gic.lower(fdt::UART_INTID);
        }
        if virtio_line.load(Ordering::Acquire) {
            gic.raise(fdt::VIRTIO_INTID);
        } else {
            gic.lower(fdt::VIRTIO_INTID);
        }
        machine.irq_line = gic.signalled(0);
        machine.vtimer_fiq = gic.vtimer_fiq();
        let mut dev = SystemDevices {
            serial: &mut serial,
            gic: &mut gic,
            virtio: virtio.as_mut(),
        };
        match machine.run_with_devices(&mut dev) {
            Ok(machine::Exit::Timer) => {
                gic.raise_on(0, fdt::VIRTUAL_TIMER_INTID);
                timer_interrupts += 1;
            }
            Ok(machine::Exit::Psci { function, args }) => match psci::handle(function, args) {
                psci::Outcome::PowerOff | psci::Outcome::CpuOff => break StopReason::Shutdown,
                psci::Outcome::Reset => break StopReason::Reset,
                psci::Outcome::Value(value) => machine.cpu.x[0] = value,
                // The FDT advertises exactly one CPU. There is no target to start.
                psci::Outcome::StartCpu { .. } => machine.cpu.x[0] = psci::NOT_SUPPORTED,
            },
            Ok(machine::Exit::Wait) => match idle_mode {
                IdleMode::Paced => {
                    let ctl = machine.cpu.system.cntv_ctl_el0;
                    let enabled = ctl & 1 != 0;
                    let masked = ctl & 2 != 0;
                    let cval = machine.cpu.system.cntv_cval_el0;
                    let vct = machine.cpu.system.cntvct_el0;
                    if enabled && !masked {
                        if cval > vct {
                            let delta = cval - vct;
                            let nanos =
                                (delta as u128 * 1_000_000_000 / fdt::TIMER_FREQ as u128) as u64;
                            let remaining = timeout.saturating_sub(started.elapsed());
                            let sleep_dur = Duration::from_nanos(nanos).min(remaining);
                            if !sleep_dur.is_zero() {
                                std::thread::sleep(sleep_dur);
                            }
                            machine.cpu.system.cntvct_el0 = cval;
                        }
                        gic.raise_on(0, fdt::VIRTUAL_TIMER_INTID);
                        timer_interrupts += 1;
                    } else if (!enabled || masked) && !machine.irq_line && !gic.signalled(0) {
                        let remaining = timeout.saturating_sub(started.elapsed());
                        let sleep_dur = Duration::from_millis(1).min(remaining);
                        if !sleep_dur.is_zero() {
                            std::thread::sleep(sleep_dur);
                        }
                    }
                }
                IdleMode::FastForward => {
                    let ctl = machine.cpu.system.cntv_ctl_el0;
                    let enabled = ctl & 1 != 0;
                    let masked = ctl & 2 != 0;
                    let cval = machine.cpu.system.cntv_cval_el0;
                    let vct = machine.cpu.system.cntvct_el0;
                    if enabled && !masked {
                        if cval > vct {
                            machine.cpu.system.cntvct_el0 = cval;
                        }
                        gic.raise_on(0, fdt::VIRTUAL_TIMER_INTID);
                        timer_interrupts += 1;
                    } else if (!enabled || masked) && !machine.irq_line && !gic.signalled(0) {
                        break StopReason::Halt;
                    }
                }
                IdleMode::Spin => {}
            },
            Ok(machine::Exit::Interrupted) => {}
            Err(error) => break StopReason::Fault(error.to_string()),
        }
        exits += 1;
        let sp = machine.cpu.sp;
        // XNU points `sp` at a kernel virtual address before it enables the MMU.
        if !is_xnu
            && machine.cpu.system.sctlr_el1 & 1 == 0
            && sp != 0
            && !(RAM_BASE..=RAM_BASE + ram_size as u64).contains(&sp)
        {
            break StopReason::LostStack;
        }
    };
    let elapsed = started.elapsed();
    let stalled_at = machine.stalled.as_ref().map(|_| machine.stalled_at);
    let at = stalled_at.unwrap_or(machine.cpu.pc);
    let instruction = instruction_diagnostic(&mut machine, at, stalled_at.is_some());
    let last_fault_walk =
        walk_diagnostic(&machine.cpu, &machine.memory, machine.cpu.system.far_el1);
    let trace = machine.recent_trace(24);
    let report = BootReport {
        reason,
        layout,
        cpu: machine.cpu,
        console: core::mem::take(&mut *console.lock()),
        exits,
        compiled_blocks: machine.cache.len(),
        executed_blocks: machine.blocks_run,
        timer_interrupts,
        unmapped_accesses: machine.unmapped,
        stalled_at,
        instruction,
        last_fault_walk,
        trace,
        elapsed,
    };
    Ok(report)
}
fn instruction_diagnostic<M: GuestMemory>(
    machine: &mut machine::Machine<M>,
    at: u64,
    stalled: bool,
) -> Option<InstructionDiagnostic> {
    let mut last = None;
    for step in 0..64 {
        let address = at.checked_add(step * 4)?;
        let physical = translate::translate(
            &mut machine.tlb,
            &machine.cpu,
            &mut machine.memory,
            address,
            translate::Access::Execute,
        )
        .ok()?;
        let mut bytes = [0; 4];
        machine.memory.read(physical, &mut bytes).ok()?;
        let word = u32::from_le_bytes(bytes);
        let decoded = decode::decode(word);
        last = Some(InstructionDiagnostic {
            virtual_address: address,
            physical_address: physical,
            word,
            decoded: decoded.is_ok(),
        });
        if !stalled || decoded.is_err() || decoded.is_ok_and(|instruction| instruction.terminates())
        {
            break;
        }
    }
    last
}
fn walk_diagnostic(cpu: &Cpu, memory: &impl GuestMemory, address: u64) -> Vec<WalkEntry> {
    let mut entries = Vec::new();
    if address == 0 || cpu.system.sctlr_el1 & 1 == 0 {
        return entries;
    }
    let upper = address >> 63 != 0;
    let size = if upper {
        (cpu.system.tcr_el1 >> 16) & 63
    } else {
        cpu.system.tcr_el1 & 63
    };
    let granule = if upper {
        (cpu.system.tcr_el1 >> 30) & 3
    } else {
        (cpu.system.tcr_el1 >> 14) & 3
    };
    if (!upper && granule != 0) || (upper && granule != 2) {
        return entries;
    }
    let bits = 64u64.saturating_sub(size);
    let levels = bits.saturating_sub(12).div_ceil(9).clamp(1, 4) as u8;
    let mut table = if upper {
        cpu.system.ttbr1_el1
    } else {
        cpu.system.ttbr0_el1
    } & 0x0000_ffff_ffff_f000;
    for level in 4 - levels..4 {
        let index = (address >> (39 - 9 * level)) & 0x1ff;
        let mut bytes = [0; 8];
        let descriptor = memory
            .read(table + index * 8, &mut bytes)
            .ok()
            .map(|_| u64::from_le_bytes(bytes));
        entries.push(WalkEntry {
            level,
            table,
            index,
            descriptor,
        });
        let Some(descriptor) = descriptor else {
            break;
        };
        if descriptor & 3 != 3 || level == 3 {
            break;
        }
        table = descriptor & 0x0000_ffff_ffff_f000;
    }
    entries
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn raw_image_executes_native_blocks_serial_mmio_and_psci_shutdown() {
        // A real AArch64 Image header branches over its metadata to native guest code.
        let mut image = vec![0; 64];
        image[..4].copy_from_slice(&0x14000010u32.to_le_bytes());
        image[4..8].copy_from_slice(&0xd503201fu32.to_le_bytes());
        image[24..32].copy_from_slice(&2u64.to_le_bytes());
        image[56..60].copy_from_slice(&boot::MAGIC.to_le_bytes());
        for instruction in [
            0xd2a12001u32, // movz x1, #0x900, lsl #16 (PL011)
            0x52800822,    // movz w2, #65
            0x39000022,    // strb w2, [x1]
            0xd2b08000,    // movz x0, #0x8400, lsl #16
            0xf2800100,    // movk x0, #8 (PSCI_SYSTEM_OFF)
            0xd4000002,    // hvc #0
        ] {
            image.extend_from_slice(&instruction.to_le_bytes());
        }
        let length = image.len() as u64;
        image[16..24].copy_from_slice(&length.to_le_bytes());
        let report =
            boot_with_serial(&image, Duration::from_secs(10), DEFAULT_CMDLINE, |_| {}).unwrap();
        assert_eq!(report.reason, StopReason::Shutdown, "{report}");
        assert_eq!(report.console, b"A");
        assert_eq!(report.unmapped_accesses, 0);
        assert!(report.compiled_blocks >= 3);
    }

    fn create_test_image(instructions: &[u32]) -> Vec<u8> {
        let mut image = vec![0; 64];
        image[..4].copy_from_slice(&0x14000010u32.to_le_bytes());
        image[4..8].copy_from_slice(&0xd503201fu32.to_le_bytes());
        image[24..32].copy_from_slice(&2u64.to_le_bytes());
        image[56..60].copy_from_slice(&boot::MAGIC.to_le_bytes());
        for instruction in instructions {
            image.extend_from_slice(&instruction.to_le_bytes());
        }
        let length = image.len() as u64;
        image[16..24].copy_from_slice(&length.to_le_bytes());
        image
    }

    #[test]
    fn wfi_idle_mode_fast_forward_advances_virtual_counter_and_fires_timer() {
        let image = create_test_image(&[
            0xd28a0000, // movz x0, #0x5000 (20480 ticks in future)
            0xd51be340, // msr cntv_cval_el0, x0
            0xd2800020, // movz x0, #1
            0xd51be320, // msr cntv_ctl_el0, x0
            0xd503207f, // wfi
            0xd2b08000, // movz x0, #0x8400, lsl #16
            0xf2800100, // movk x0, #8 (PSCI_SYSTEM_OFF)
            0xd4000002, // hvc #0
        ]);
        let report = boot_with_options(
            &image,
            Duration::from_secs(5),
            DEFAULT_CMDLINE,
            IdleMode::FastForward,
            |_| {},
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Shutdown, "{report}");
        assert!(report.cpu.system.cntvct_el0 >= 0x5000);
        assert!(report.timer_interrupts >= 1);
    }

    #[test]
    fn wfi_idle_mode_fast_forward_halts_when_no_timer_armed() {
        let image = create_test_image(&[
            0xd503207f, // wfi with no timer enabled
            0xd2b08000, // movz x0, #0x8400, lsl #16
            0xf2800100, // movk x0, #8 (PSCI_SYSTEM_OFF)
            0xd4000002, // hvc #0
        ]);
        let report = boot_with_options(
            &image,
            Duration::from_secs(5),
            DEFAULT_CMDLINE,
            IdleMode::FastForward,
            |_| {},
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Halt, "{report}");
    }

    #[test]
    fn wfi_idle_mode_paced_sleeps_and_fires_timer() {
        let image = create_test_image(&[
            0xd2977000, // movz x0, #48000 (2 ms at 24 MHz)
            0xd51be340, // msr cntv_cval_el0, x0
            0xd2800020, // movz x0, #1
            0xd51be320, // msr cntv_ctl_el0, x0
            0xd503207f, // wfi
            0xd2b08000, // movz x0, #0x8400, lsl #16
            0xf2800100, // movk x0, #8 (PSCI_SYSTEM_OFF)
            0xd4000002, // hvc #0
        ]);
        let report = boot_with_options(
            &image,
            Duration::from_secs(5),
            DEFAULT_CMDLINE,
            IdleMode::Paced,
            |_| {},
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Shutdown, "{report}");
        assert!(report.cpu.system.cntvct_el0 >= 48_000);
        assert!(report.timer_interrupts >= 1);
        assert!(report.elapsed >= Duration::from_millis(2));
    }
    #[test]
    fn boot_system_with_custom_ram_and_initrd() {
        let image = create_test_image(&[
            0xd2a12001u32, // movz x1, #0x900, lsl #16 (PL011)
            0x52800842,    // movz w2, #66 ('B')
            0x39000022,    // strb w2, [x1]
            0xd2b08000,    // movz x0, #0x8400, lsl #16
            0xf2800100,    // movk x0, #8 (PSCI_SYSTEM_OFF)
            0xd4000002,    // hvc #0
        ]);
        let report = boot_system(
            &image,
            SystemConfig::<Vec<u8>> {
                timeout: Duration::from_secs(5),
                cmdline: DEFAULT_CMDLINE.to_string(),
                idle_mode: IdleMode::Paced,
                ram_size: 256 << 20,
                initrd: Some(b"test initrd".to_vec()),
                disk: None,
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Shutdown);
        assert_eq!(report.console, b"B");
        assert!(report.layout.initrd.is_some());
    }
    #[test]
    fn boot_system_with_disk_image() {
        let image = create_test_image(&[
            0xd2a12001u32, // movz x1, #0x900, lsl #16 (PL011)
            0x52800862,    // movz w2, #67 ('C')
            0x39000022,    // strb w2, [x1]
            0xd2b08000,    // movz x0, #0x8400, lsl #16
            0xf2800100,    // movk x0, #8 (PSCI_SYSTEM_OFF)
            0xd4000002,    // hvc #0
        ]);
        let disk = vec![0u8; 1024 * 1024]; // 1 MiB disk
        let report = boot_system(
            &image,
            SystemConfig {
                timeout: Duration::from_secs(5),
                cmdline: DEFAULT_CMDLINE.to_string(),
                idle_mode: IdleMode::Paced,
                ram_size: 128 << 20,
                initrd: None,
                disk: Some(disk),
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(report.reason, StopReason::Shutdown);
        assert_eq!(report.console, b"C");
    }
}
