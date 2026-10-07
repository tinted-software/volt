//! Multi-threaded full-system execution (QEMU MTTCG style).
//!
//! Spawns one host OS thread per virtual CPU. Primary CPU 0 starts at kernel entry;
//! secondary CPUs (1..N-1) start halted and are awakened via PSCI `CPU_ON`.
//! WFI halts vCPU threads using condition variables until timer expiration or IPI/SGI.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use parking_lot::{Condvar, Mutex};

use super::{
    Error, IdleMode, RAM_BASE, RAM_SIZE, StopReason, boot, fdt,
    machine::{DeviceIo, Exit, Machine},
    psci,
};
use crate::aarch64::Cpu;
use crate::devices::{
    bus::Device,
    gicv2::{self, Gicv2},
    pl011::Pl011,
};
use crate::memory::{PhysicalMemory, Region, SharedMemory};

const INVALID_RESERVATION: u64 = u64::MAX;

/// Global exclusive monitor tracking reservations across all virtual CPUs for LDXR/STXR.
pub struct GlobalExclusiveMonitor {
    reservations: [AtomicU64; gicv2::MAX_CPUS],
}

impl Default for GlobalExclusiveMonitor {
    fn default() -> Self {
        Self {
            reservations: core::array::from_fn(|_| AtomicU64::new(INVALID_RESERVATION)),
        }
    }
}

impl GlobalExclusiveMonitor {
    pub fn reserve(&self, cpu: u32, address: u64) {
        if let Some(slot) = self.reservations.get(cpu as usize) {
            slot.store(address & !63, Ordering::Release);
        }
    }

    pub fn check_and_clear(&self, cpu: u32, address: u64) -> bool {
        if let Some(slot) = self.reservations.get(cpu as usize) {
            slot.swap(INVALID_RESERVATION, Ordering::AcqRel) == (address & !63)
        } else {
            false
        }
    }

    pub fn invalidate_others(&self, cpu: u32, address: u64) {
        let line = address & !63;
        for (idx, slot) in self.reservations.iter().enumerate() {
            if idx as u32 != cpu {
                let _ = slot.compare_exchange(
                    line,
                    INVALID_RESERVATION,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcpuState {
    Halted,
    Running,
    WaitingWfi,
    Stopped,
}

pub struct VcpuControl {
    pub state: Mutex<VcpuState>,
    pub cond: Condvar,
    pub pc: AtomicU64,
    pub x0: AtomicU64,
    pub timer_interrupts: AtomicU64,
    pub blocks_run: AtomicU64,
    pub exits: AtomicU64,
}

impl Default for VcpuControl {
    fn default() -> Self {
        Self {
            state: Mutex::new(VcpuState::Halted),
            cond: Condvar::new(),
            pc: AtomicU64::new(0),
            x0: AtomicU64::new(0),
            timer_interrupts: AtomicU64::new(0),
            blocks_run: AtomicU64::new(0),
            exits: AtomicU64::new(0),
        }
    }
}

pub struct SharedDevices<V = ()> {
    pub serial: Arc<Mutex<Pl011>>,
    pub gic: Arc<Mutex<Gicv2>>,
    pub virtio: Option<Arc<Mutex<V>>>,
    pub exclusive: Arc<GlobalExclusiveMonitor>,
    /// Translated blocks shared by every vCPU.
    pub blocks: Arc<crate::aarch64::cache::SharedBlocks>,
    pub controls: Vec<Arc<VcpuControl>>,
    pub uart_line: Arc<AtomicBool>,
    pub virtio_line: Arc<AtomicBool>,
    pub shutdown_reason: Arc<Mutex<Option<StopReason>>>,
    pub shutdown_cond: Arc<Condvar>,
}

impl<V> Clone for SharedDevices<V> {
    fn clone(&self) -> Self {
        Self {
            serial: self.serial.clone(),
            gic: self.gic.clone(),
            virtio: self.virtio.clone(),
            exclusive: self.exclusive.clone(),
            blocks: self.blocks.clone(),
            controls: self.controls.clone(),
            uart_line: self.uart_line.clone(),
            virtio_line: self.virtio_line.clone(),
            shutdown_reason: self.shutdown_reason.clone(),
            shutdown_cond: self.shutdown_cond.clone(),
        }
    }
}

impl<V> SharedDevices<V> {
    pub fn is_shutdown(&self) -> bool {
        self.shutdown_reason.lock().is_some()
    }

    pub fn request_shutdown(&self, reason: StopReason) {
        let mut lock = self.shutdown_reason.lock();
        if lock.is_none() {
            *lock = Some(reason);
            self.shutdown_cond.notify_all();
            for ctrl in &self.controls {
                let mut state = ctrl.state.lock();
                *state = VcpuState::Stopped;
                ctrl.cond.notify_all();
            }
        }
    }

    pub fn kick_targets(&self, targets: u8) {
        for (cpu_id, ctrl) in self.controls.iter().enumerate() {
            if targets & (1 << cpu_id) != 0 {
                ctrl.cond.notify_one();
            }
        }
    }
}

impl<V: crate::devices::VirtioIo> DeviceIo for SharedDevices<V> {
    fn read_uart(&mut self, offset: u64, size: u8) -> u64 {
        self.serial.lock().read(offset, size)
    }
    fn write_uart(&mut self, offset: u64, size: u8, value: u64) {
        self.serial.lock().write(offset, size, value);
    }
    fn read_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        self.gic.lock().read_distributor_for(cpu_id, offset, size)
    }
    fn write_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        let targets = self
            .gic
            .lock()
            .write_distributor_for(cpu_id, offset, size, value);
        if targets != 0 {
            self.kick_targets(targets);
        }
    }
    fn read_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        self.gic.lock().read_cpu_for(cpu_id, offset, size)
    }
    fn write_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        self.gic.lock().write_cpu_for(cpu_id, offset, size, value);
    }
    fn read_virtio(&mut self, offset: u64, size: u8) -> u64 {
        if let Some(v) = &self.virtio {
            v.lock().read(offset, size)
        } else {
            0
        }
    }
    fn write_virtio(&mut self, offset: u64, size: u8, value: u64) {
        if let Some(v) = &self.virtio {
            v.lock().write(offset, size, value);
        }
    }
}

impl DeviceIo for SharedDevices<()> {
    fn read_uart(&mut self, offset: u64, size: u8) -> u64 {
        self.serial.lock().read(offset, size)
    }
    fn write_uart(&mut self, offset: u64, size: u8, value: u64) {
        self.serial.lock().write(offset, size, value);
    }
    fn read_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        self.gic.lock().read_distributor_for(cpu_id, offset, size)
    }
    fn write_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        let targets = self
            .gic
            .lock()
            .write_distributor_for(cpu_id, offset, size, value);
        if targets != 0 {
            self.kick_targets(targets);
        }
    }
    fn read_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        self.gic.lock().read_cpu_for(cpu_id, offset, size)
    }
    fn write_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        self.gic.lock().write_cpu_for(cpu_id, offset, size, value);
    }
}

#[derive(Debug, Clone)]
pub struct SmpConfig<B = Vec<u8>> {
    pub cpus: u32,
    pub timeout: Duration,
    pub cmdline: String,
    pub idle_mode: IdleMode,
    pub ram_size: usize,
    pub initrd: Option<Vec<u8>>,
    pub disk: Option<B>,
}

impl<B> Default for SmpConfig<B> {
    fn default() -> Self {
        Self {
            cpus: 2,
            timeout: Duration::from_secs(20),
            cmdline: super::DEFAULT_CMDLINE.into(),
            idle_mode: IdleMode::default(),
            ram_size: super::RAM_SIZE,
            initrd: None,
            disk: None,
        }
    }
}

#[derive(Debug)]
pub struct SmpReport {
    pub reason: StopReason,
    pub layout: boot::Layout,
    pub cpus: Vec<Cpu>,
    pub console: Vec<u8>,
    pub total_exits: u64,
    pub total_blocks_run: u64,
    pub total_timer_interrupts: u64,
    pub elapsed: Duration,
}

impl fmt::Display for SmpReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- smp stopped: {:?}", self.reason)?;
        writeln!(
            f,
            "entry {:x}, device tree {:x}, cpus: {}",
            self.layout.entry,
            self.layout.device_tree,
            self.cpus.len()
        )?;
        for (idx, c) in self.cpus.iter().enumerate() {
            writeln!(
                f,
                "cpu {idx}: pc {:x} sp {:x} el {} trap {:?}",
                c.pc, c.sp, c.system.el, c.trap
            )?;
        }
        writeln!(
            f,
            "exits {}, blocks executed {}, timer interrupts {}, elapsed {:.3}s",
            self.total_exits,
            self.total_blocks_run,
            self.total_timer_interrupts,
            self.elapsed.as_secs_f64()
        )
    }
}

/// Boot a Linux Image in multi-threaded SMP mode (MTTCG).
pub fn boot_smp<B: crate::devices::BlockBackend + 'static>(
    image: &[u8],
    config: SmpConfig<B>,
    mut sink: impl FnMut(u8) + Send + 'static,
) -> Result<SmpReport, Error> {
    let cpu_count = config.cpus.clamp(1, gicv2::MAX_CPUS as u32);
    let ram_size = if config.ram_size > 0 {
        config.ram_size
    } else {
        RAM_SIZE
    };
    let region = Region::ram(RAM_BASE, vec![0; ram_size]).map_err(Error::Memory)?;
    let memory = PhysicalMemory::new(vec![region]).map_err(Error::Memory)?;
    let mut shared_mem = SharedMemory::new(memory);

    let has_virtio = config.disk.is_some();
    let layout = boot::prepare_boot(
        &mut shared_mem,
        image,
        config.initrd.as_deref(),
        &config.cmdline,
        RAM_BASE,
        ram_size as u64,
        cpu_count,
        has_virtio,
    )
    .map_err(Error::Boot)?;
    let console = Arc::new(Mutex::new(Vec::new()));
    let console_sink = console.clone();
    let uart_line = Arc::new(AtomicBool::new(false));
    let line_notify = uart_line.clone();

    let serial = Pl011::new(move |byte| {
        console_sink.lock().push(byte);
        sink(byte);
    })
    .with_line(move |level| {
        line_notify.store(level, Ordering::Release);
    });

    let mut gic = Gicv2::default();
    gic.cpus = cpu_count;

    let controls: Vec<Arc<VcpuControl>> = (0..cpu_count)
        .map(|_| Arc::new(VcpuControl::default()))
        .collect();

    // CPU 0 starts in Running state
    *controls[0].state.lock() = VcpuState::Running;

    let virtio_line = Arc::new(AtomicBool::new(false));
    let virtio_notify = virtio_line.clone();
    let virtio = config.disk.map(|backend| {
        Arc::new(Mutex::new(
            crate::devices::VirtioBlock::new(backend, shared_mem.clone())
                .with_line(move |level| virtio_notify.store(level, Ordering::Release)),
        ))
    });

    let blocks = Arc::new(crate::aarch64::cache::SharedBlocks::new(true));
    if let Some(reader) = crate::memory::GuestMemory::code_reader(&shared_mem) {
        blocks.enable_ahead(reader, crate::aarch64::ahead::default_threads());
    }
    let devices = SharedDevices {
        serial: Arc::new(Mutex::new(serial)),
        gic: Arc::new(Mutex::new(gic)),
        virtio,
        exclusive: Arc::new(GlobalExclusiveMonitor::default()),
        blocks,
        controls: controls.clone(),
        uart_line,
        virtio_line,
        shutdown_reason: Arc::new(Mutex::new(None)),
        shutdown_cond: Arc::new(Condvar::new()),
    };

    let started = Instant::now();
    let mut workers: Vec<JoinHandle<Cpu>> = Vec::with_capacity(cpu_count as usize);

    for cpu_id in 0..cpu_count {
        let mem = shared_mem.clone();
        let ctrl = controls[cpu_id as usize].clone();
        let dev = devices.clone();
        let idle_mode = config.idle_mode;
        let timeout = config.timeout;
        let entry = layout.entry;
        let dtb = layout.device_tree;

        let handle = thread::Builder::new()
            .name(alloc::format!("vcpu-{cpu_id}"))
            .spawn(move || {
                vcpu_worker_thread(
                    cpu_id, mem, ctrl, dev, timeout, started, idle_mode, entry, dtb,
                )
            })
            .expect("spawn vcpu worker");

        workers.push(handle);
    }

    // Wait until shutdown is requested or timeout expires
    {
        let mut reason_lock = devices.shutdown_reason.lock();
        while reason_lock.is_none() {
            let elapsed = started.elapsed();
            if elapsed >= config.timeout {
                *reason_lock = Some(StopReason::Timeout);
                break;
            }
            let remaining = config.timeout - elapsed;
            let result = devices.shutdown_cond.wait_for(&mut reason_lock, remaining);
            if result.timed_out() {
                *reason_lock = Some(StopReason::Timeout);
                break;
            }
        }
    }

    // Ensure all vCPUs are unparked and exiting
    devices.request_shutdown(StopReason::Timeout);

    let mut cpus = Vec::with_capacity(workers.len());
    for handle in workers {
        if let Ok(cpu) = handle.join() {
            cpus.push(cpu);
        }
    }

    let elapsed = started.elapsed();
    let reason = devices
        .shutdown_reason
        .lock()
        .clone()
        .unwrap_or(StopReason::Timeout);

    let mut total_exits = 0;
    let mut total_blocks_run = 0;
    let mut total_timer_interrupts = 0;
    for ctrl in &controls {
        total_exits += ctrl.exits.load(Ordering::Relaxed);
        total_blocks_run += ctrl.blocks_run.load(Ordering::Relaxed);
        total_timer_interrupts += ctrl.timer_interrupts.load(Ordering::Relaxed);
    }

    let console_output = core::mem::take(&mut *console.lock());

    Ok(SmpReport {
        reason,
        layout,
        cpus,
        console: console_output,
        total_exits,
        total_blocks_run,
        total_timer_interrupts,
        elapsed,
    })
}

fn vcpu_worker_thread<V: crate::devices::VirtioIo + Send + 'static>(
    cpu_id: u32,
    memory: SharedMemory,
    control: Arc<VcpuControl>,
    devices: SharedDevices<V>,
    timeout: Duration,
    started: Instant,
    idle_mode: IdleMode,
    boot_entry: u64,
    device_tree: u64,
) -> Cpu {
    let mut machine = Machine::with_shared_blocks(memory, cpu_id, devices.blocks.clone());

    if cpu_id == 0 {
        machine.cpu.pc = boot_entry;
        machine.cpu.x[0] = device_tree;
        machine.cpu.x[1..4].fill(0);
    } else {
        // Secondary CPU waits for PSCI CPU_ON
        let mut state = control.state.lock();
        while *state == VcpuState::Halted {
            if devices.is_shutdown() {
                return machine.cpu;
            }
            control.cond.wait(&mut state);
        }
        if *state == VcpuState::Stopped {
            return machine.cpu;
        }
        machine.cpu.pc = control.pc.load(Ordering::Acquire);
        machine.cpu.x[0] = control.x0.load(Ordering::Acquire);
        machine.cpu.x[1..4].fill(0);
    }

    loop {
        if devices.is_shutdown() || started.elapsed() >= timeout {
            devices.request_shutdown(StopReason::Timeout);
            break;
        }

        // Check UART line and route to GIC (only CPU 0 handles SPI routing)
        if cpu_id == 0 {
            if devices.uart_line.load(Ordering::Acquire) {
                devices.gic.lock().raise(fdt::UART_INTID);
            } else {
                devices.gic.lock().lower(fdt::UART_INTID);
            }
            if devices.virtio_line.load(Ordering::Acquire) {
                devices.gic.lock().raise(fdt::VIRTIO_INTID);
            } else {
                devices.gic.lock().lower(fdt::VIRTIO_INTID);
            }
        }

        let irq = devices.gic.lock().signalled(cpu_id);
        machine.irq_line = irq;

        let mut dev = devices.clone();
        let exit = machine.run_with_devices(&mut dev);

        control
            .blocks_run
            .store(machine.blocks_run, Ordering::Relaxed);
        control.exits.fetch_add(1, Ordering::Relaxed);

        match exit {
            Ok(Exit::Timer) => {
                let mut gic = devices.gic.lock();
                gic.raise_on(cpu_id, fdt::VIRTUAL_TIMER_INTID);
                control.timer_interrupts.fetch_add(1, Ordering::Relaxed);
            }
            Ok(Exit::Psci { function, args }) => match psci::handle(function, args) {
                psci::Outcome::PowerOff => {
                    devices.request_shutdown(StopReason::Shutdown);
                    break;
                }
                psci::Outcome::Reset => {
                    devices.request_shutdown(StopReason::Reset);
                    break;
                }
                psci::Outcome::CpuOff => {
                    let mut state = control.state.lock();
                    *state = VcpuState::Halted;
                    while *state == VcpuState::Halted {
                        if devices.is_shutdown() {
                            break;
                        }
                        control.cond.wait(&mut state);
                    }
                    if *state == VcpuState::Stopped || devices.is_shutdown() {
                        break;
                    }
                    machine.cpu.pc = control.pc.load(Ordering::Acquire);
                    machine.cpu.x[0] = control.x0.load(Ordering::Acquire);
                    machine.cpu.x[1..4].fill(0);
                }
                psci::Outcome::StartCpu {
                    target,
                    entry,
                    context,
                } => {
                    if let Some(target_ctrl) = devices.controls.get(target as usize) {
                        target_ctrl.pc.store(entry, Ordering::Release);
                        target_ctrl.x0.store(context, Ordering::Release);
                        let mut target_state = target_ctrl.state.lock();
                        if *target_state == VcpuState::Halted {
                            *target_state = VcpuState::Running;
                            target_ctrl.cond.notify_one();
                            machine.cpu.x[0] = psci::SUCCESS;
                        } else {
                            machine.cpu.x[0] = psci::ALREADY_ON;
                        }
                    } else {
                        machine.cpu.x[0] = psci::INVALID_PARAMETERS;
                    }
                }
                psci::Outcome::Value(val) => {
                    machine.cpu.x[0] = val;
                }
            },
            Ok(Exit::Wait) => match idle_mode {
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
                                let mut state = control.state.lock();
                                *state = VcpuState::WaitingWfi;
                                control.cond.wait_for(&mut state, sleep_dur);
                                *state = VcpuState::Running;
                            }
                            machine.cpu.system.cntvct_el0 = cval;
                        }
                        let mut gic = devices.gic.lock();
                        gic.raise_on(cpu_id, fdt::VIRTUAL_TIMER_INTID);
                        control.timer_interrupts.fetch_add(1, Ordering::Relaxed);
                    } else if (!enabled || masked) && !machine.irq_line {
                        let mut state = control.state.lock();
                        *state = VcpuState::WaitingWfi;
                        control.cond.wait_for(&mut state, Duration::from_millis(1));
                        *state = VcpuState::Running;
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
                        let mut gic = devices.gic.lock();
                        gic.raise_on(cpu_id, fdt::VIRTUAL_TIMER_INTID);
                        control.timer_interrupts.fetch_add(1, Ordering::Relaxed);
                    } else if (!enabled || masked) && !machine.irq_line {
                        // Secondary core or idle core waits for IPI from other core
                        let mut state = control.state.lock();
                        *state = VcpuState::WaitingWfi;
                        control.cond.wait_for(&mut state, Duration::from_millis(5));
                        *state = VcpuState::Running;
                    }
                }
                IdleMode::Spin => {}
            },
            Ok(Exit::Interrupted) => {}
            Err(err) => {
                let pc = machine
                    .stalled
                    .map(|_| machine.stalled_at)
                    .unwrap_or(machine.cpu.pc);
                devices.request_shutdown(StopReason::Fault(alloc::format!(
                    "cpu {cpu_id} at {pc:x}: {err}"
                )));
                break;
            }
        }
    }

    machine.cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn exclusive_monitor_tracks_reservations_and_invalidation() {
        let monitor = GlobalExclusiveMonitor::default();
        monitor.reserve(0, 0x1000);
        monitor.reserve(1, 0x2000);

        assert!(monitor.check_and_clear(0, 0x1000));
        assert!(!monitor.check_and_clear(0, 0x1000));

        monitor.reserve(1, 0x2000);
        monitor.invalidate_others(0, 0x2000);
        assert!(!monitor.check_and_clear(1, 0x2000));
    }

    #[test]
    fn smp_two_cpus_boot_secondary_via_psci_and_execute_concurrently() {
        let mut image = vec![0; 512];
        image[..4].copy_from_slice(&0x14000010u32.to_le_bytes());
        image[4..8].copy_from_slice(&0xd503201fu32.to_le_bytes());
        image[24..32].copy_from_slice(&2u64.to_le_bytes());
        image[56..60].copy_from_slice(&boot::MAGIC.to_le_bytes());

        let primary_instructions: [u32; 11] = [
            0xd2a12001, // movz x1, #0x900, lsl #16 (UART base 0x09000000)
            0x52800822, // movz w2, #65 ('A')
            0x39000022, // strb w2, [x1]
            0xd2b88000, // movz x0, #0xc400, lsl #16
            0xf2800060, // movk x0, #3 (PSCI CPU_ON)
            0xd2800021, // movz x1, #1 (target CPU 1)
            0xd2a80002, // movz x2, #0x4000, lsl #16
            0xf2802002, // movk x2, #0x0100 (secondary entry: 0x4000_0100)
            0xd2800003, // movz x3, #0 (context)
            0xd4000002, // hvc #0 (issue CPU_ON)
            0x14000000, // b . (spin waiting for system shutdown)
        ];
        for (i, inst) in primary_instructions.iter().enumerate() {
            let offset = 64 + i * 4;
            image[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
        }

        let secondary_instructions: [u32; 6] = [
            0xd2a12001, // movz x1, #0x900, lsl #16 (UART base 0x09000000)
            0x52800842, // movz w2, #66 ('B')
            0x39000022, // strb w2, [x1]
            0xd2b08000, // movz x0, #0x8400, lsl #16
            0xf2800100, // movk x0, #8 (PSCI SYSTEM_OFF)
            0xd4000002, // hvc #0 (shut down entire machine)
        ];
        for (i, inst) in secondary_instructions.iter().enumerate() {
            let offset = 256 + i * 4;
            image[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
        }

        let length = image.len() as u64;
        image[16..24].copy_from_slice(&length.to_le_bytes());

        let config = SmpConfig::<Vec<u8>> {
            cpus: 2,
            timeout: Duration::from_secs(5),
            cmdline: super::super::DEFAULT_CMDLINE.into(),
            idle_mode: IdleMode::FastForward,
            ..Default::default()
        };

        let report = boot_smp(&image, config, |_| {}).unwrap();
        assert_eq!(report.reason, StopReason::Shutdown, "{report}");
        assert_eq!(report.cpus.len(), 2);
        assert_eq!(report.console, b"AB");
        assert!(report.total_blocks_run >= 2);
    }

    #[test]
    fn smp_secondary_cpu_wfi_condvar_wait_and_timer_wakeup() {
        let mut image = vec![0; 512];
        image[..4].copy_from_slice(&0x14000010u32.to_le_bytes());
        image[4..8].copy_from_slice(&0xd503201fu32.to_le_bytes());
        image[24..32].copy_from_slice(&2u64.to_le_bytes());
        image[56..60].copy_from_slice(&boot::MAGIC.to_le_bytes());

        // Primary CPU 0 wakes CPU 1 via PSCI CPU_ON
        let primary_instructions: [u32; 8] = [
            0xd2b88000, // movz x0, #0xc400, lsl #16
            0xf2800060, // movk x0, #3 (PSCI CPU_ON)
            0xd2800021, // movz x1, #1 (target CPU 1)
            0xd2a80002, // movz x2, #0x4000, lsl #16
            0xf2802002, // movk x2, #0x0100 (secondary entry: 0x4000_0100)
            0xd2800003, // movz x3, #0 (context)
            0xd4000002, // hvc #0
            0x14000000, // b .
        ];
        for (i, inst) in primary_instructions.iter().enumerate() {
            let offset = 64 + i * 4;
            image[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
        }

        // Secondary CPU 1 arms a 2ms timer, waits in WFI on Condvar, prints 'B', and powers off
        let secondary_instructions: [u32; 11] = [
            0xd2977000, // movz x0, #48000 (2 ms at 24 MHz)
            0xd51be340, // msr cntv_cval_el0, x0
            0xd2800020, // movz x0, #1
            0xd51be320, // msr cntv_ctl_el0, x0
            0xd503207f, // wfi (condvar wait!)
            0xd2a12001, // movz x1, #0x900, lsl #16
            0x52800842, // movz w2, #66 ('B')
            0x39000022, // strb w2, [x1]
            0xd2b08000, // movz x0, #0x8400, lsl #16
            0xf2800100, // movk x0, #8 (PSCI SYSTEM_OFF)
            0xd4000002, // hvc #0
        ];
        for (i, inst) in secondary_instructions.iter().enumerate() {
            let offset = 256 + i * 4;
            image[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
        }

        let length = image.len() as u64;
        image[16..24].copy_from_slice(&length.to_le_bytes());

        let config = SmpConfig::<Vec<u8>> {
            cpus: 2,
            timeout: Duration::from_secs(5),
            cmdline: super::super::DEFAULT_CMDLINE.into(),
            idle_mode: IdleMode::Paced,
            ..Default::default()
        };
        let report = boot_smp(&image, config, |_| {}).unwrap();
        assert_eq!(report.reason, StopReason::Shutdown, "{report}");
        assert_eq!(report.console, b"B");
        assert!(report.elapsed >= Duration::from_millis(2));
        assert!(report.total_timer_interrupts >= 1);
    }
}
