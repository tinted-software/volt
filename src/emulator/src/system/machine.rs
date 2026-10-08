//! One-vCPU machine over native translated blocks (Mirage aarch64/Machine.zig).
use super::afdt;
use super::fdt;
use crate::aarch64::cpu::{Exclusive, Trap};
use crate::aarch64::decode::SignExtend;
use crate::aarch64::simd_struct;
use crate::aarch64::{Cache, Cpu};
use crate::aarch64::{exception, translate};
use crate::devices::{bus::Device, gicv2::Gicv2, pl011::Pl011};
use crate::memory::{GuestMemory, MemoryError};

#[derive(Debug)]
pub enum Error {
    Compile(crate::aarch64::compile::Error),
    Memory(MemoryError),
    InvalidAccessWidth(u8),
    /// An LSE atomic on memory that is not host RAM, or not aligned to its size.
    UnsupportedAtomic(u64),
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl core::error::Error for Error {}
pub trait DeviceIo {
    fn read_uart(&mut self, offset: u64, size: u8) -> u64;
    fn write_uart(&mut self, offset: u64, size: u8, value: u64);
    fn read_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64;
    fn write_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64);
    fn read_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64;
    fn write_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64);
    fn read_gic_redistributor(&mut self, offset: u64, size: u8) -> u64 {
        let _ = (offset, size);
        0
    }
    fn write_gic_redistributor(&mut self, offset: u64, size: u8, value: u64) {
        let _ = (offset, size, value);
    }
    fn read_virtio(&mut self, offset: u64, size: u8) -> u64 {
        let _ = (offset, size);
        0
    }
    fn write_virtio(&mut self, offset: u64, size: u8, value: u64) {
        let _ = (offset, size, value);
    }
    /// A device the board places anywhere in the address space (the M1 SoC's
    /// interrupt controller and UART). `None` declines the access, which then goes
    /// to the fixed windows below.
    fn read_soc(&mut self, cpu_id: u32, address: u64, size: u8) -> Option<u64> {
        let _ = (cpu_id, address, size);
        None
    }
    /// Returns `false` when the board's devices do not claim the address.
    fn write_soc(&mut self, cpu_id: u32, address: u64, size: u8, value: u64) -> bool {
        let _ = (cpu_id, address, size, value);
        false
    }
    /// Whether the fixed windows below (the QEMU `virt` UART, GIC and virtio) exist on
    /// this board. A board that maps none of them makes accesses there fault.
    fn has_fixed_windows(&self) -> bool {
        true
    }
}

impl DeviceIo for (&mut Pl011, &mut Gicv2) {
    fn read_uart(&mut self, offset: u64, size: u8) -> u64 {
        self.0.read(offset, size)
    }
    fn write_uart(&mut self, offset: u64, size: u8, value: u64) {
        self.0.write(offset, size, value);
    }
    fn read_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        self.1.read_distributor_for(cpu_id, offset, size)
    }
    fn write_gic_distributor(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        self.1.write_distributor_for(cpu_id, offset, size, value);
    }
    fn read_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8) -> u64 {
        self.1.read_cpu_for(cpu_id, offset, size)
    }
    fn write_gic_cpu(&mut self, cpu_id: u32, offset: u64, size: u8, value: u64) {
        self.1.write_cpu_for(cpu_id, offset, size, value);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Exit {
    Interrupted,
    Wait,
    Timer,
    Psci { function: u32, args: [u64; 3] },
}
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceEntry {
    pub pc: u64,
    pub sp: u64,
}

/// Instruction fetch through the guest MMU, with code-page write tracking.
struct SystemFetch<'a, M> {
    tlb: &'a mut translate::Tlb,
    memory: &'a mut M,
    /// Address and permission-ness of the last failed fetch, for abort reporting.
    fault: Option<(u64, bool)>,
}
impl<M: GuestMemory> crate::aarch64::cache::Fetch for SystemFetch<'_, M> {
    fn fetch(
        &mut self,
        cpu: &Cpu,
        address: u64,
        bytes: &mut [u8],
    ) -> Result<usize, crate::aarch64::compile::Error> {
        let physical = match translate::translate(
            self.tlb,
            cpu,
            self.memory,
            address,
            translate::Access::Execute,
        ) {
            Ok(physical) => physical,
            Err(error) => {
                self.fault = Some((
                    address,
                    matches!(error, translate::TranslateError::PermissionFault),
                ));
                return Err(crate::aarch64::compile::Error::Memory(
                    MemoryError::AccessFault {
                        address,
                        length: bytes.len(),
                    },
                ));
            }
        };
        if let Err(error) = self.memory.read(physical, bytes) {
            self.fault = Some((address, false));
            return Err(crate::aarch64::compile::Error::Memory(error));
        }
        Ok(bytes.len())
    }
    fn code_key(&mut self, cpu: &mut Cpu, pc: u64, watch: bool) -> Option<(u64, u32)> {
        self.memory.code_tracker()?;
        // Faults here are not reported: the caller falls back to `fetch`, which
        // repeats the translation and records the fault.
        let physical =
            translate::translate(self.tlb, &*cpu, self.memory, pc, translate::Access::Execute)
                .ok()?;
        let page = physical >> 12;
        let tracker = self.memory.code_tracker()?;
        let version = if watch {
            let (version, newly_watched) = tracker.watch_tracked(page)?;
            if newly_watched {
                // This vCPU may hold inline write permission for the page. Drop
                // it now: the block compiled next could store to its own page.
                cpu.dtlb.purge_writes();
            }
            version
        } else {
            tracker.version(page)?
        };
        Some((page, version))
    }
    fn translation_epoch(&self) -> Option<u64> {
        // `sync_dtlb` has already folded any context change into the
        // generation, and every TLB flush advances it.
        Some(self.tlb.generation)
    }
    fn page_version(&self, page: u64) -> Option<u32> {
        self.memory.code_tracker()?.version(page)
    }
}

pub struct Machine<M> {
    pub cpu_id: u32,
    pub memory: M,
    pub cache: Cache,
    pub cpu: Cpu,
    pub tlb: translate::Tlb,
    /// `tlb.generation` the inline data TLB (`cpu.dtlb`) was last synced to.
    dtlb_generation: u64,
    /// `CodeTracker::epoch` the inline data TLB's write permissions reflect.
    watch_epoch: u64,
    pub irq_line: bool,
    /// Set when a GICv3 forwards PPI 27 (the virtual timer) as a group 0 FIQ: the
    /// priority it was given. `None` leaves the timer to the GICv2 path.
    pub vtimer_fiq: Option<u8>,
    /// A FIQ source the board drives directly. The M1 interrupt controller presents
    /// the virtual timer this way: it is level-high while the timer's own condition
    /// holds, with no GIC-style priority gating.
    pub fiq_line: bool,
    timer_fired: bool,
    pub stalled_at: u64,
    pub stalled: Option<String>,
    pub unmapped: u64,
    trace: [TraceEntry; 512],
    pub blocks_run: u64,
    faulted: bool,
    device_access: bool,
    /// Per-instruction observer installed by [`Machine::set_step_trace`].
    step_trace: Option<Box<dyn FnMut(&Step) + Send>>,
    /// Per-access observer installed by [`Machine::set_memory_trace`].
    memory_trace: Option<Box<dyn FnMut(&MemoryAccess) + Send>>,
    /// The step in progress: its start `pc` and the system registers before it ran.
    step_pending: Option<(u64, crate::aarch64::cpu::System)>,
}

/// One step seen by a [`Machine::set_step_trace`] observer. Single-stepping makes each
/// step one instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// Address the step started at.
    pub pc: u64,
    /// Address after the step.
    pub next_pc: u64,
    /// Each watched system register that changed, as `(name, before, after)`.
    pub changes: Vec<(&'static str, u64, u64)>,
}

/// One load or store seen by a [`Machine::set_memory_trace`] observer, after translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccess {
    /// Address of the instruction that made the access.
    pub pc: u64,
    pub virtual_address: u64,
    pub physical: u64,
    /// Bytes accessed: 1, 2, 4, 8 or 16.
    pub width: u8,
    pub write: bool,
    /// The low 8 bytes stored; 0 for a load, whose value is not known before it runs.
    pub value: u64,
}

/// The watched system registers whose values differ between `before` and `after`.
fn system_changes(
    before: &crate::aarch64::cpu::System,
    after: &crate::aarch64::cpu::System,
) -> Vec<(&'static str, u64, u64)> {
    macro_rules! watch {
        ($($field:ident),* $(,)?) => {{
            let mut changes = Vec::new();
            $(
                if before.$field != after.$field {
                    changes.push((stringify!($field), before.$field as u64, after.$field as u64));
                }
            )*
            changes
        }};
    }
    let mut changes = watch!(
        el,
        spsel,
        daif,
        pan,
        spsr_el1,
        elr_el1,
        esr_el1,
        far_el1,
        vbar_el1,
        sctlr_el1,
        ttbr0_el1,
        ttbr1_el1,
        tcr_el1,
        mair_el1,
        spsr_el2,
        elr_el2,
        esr_el2,
        far_el2,
        vbar_el2,
        sctlr_el2,
        ttbr0_el2,
        ttbr1_el2,
        tcr_el2,
        mair_el2,
        hcr_el2,
        tpidr_el2,
        cntvoff_el2,
    );
    const SP_NAMES: [&str; 4] = ["sp_el0", "sp_el1", "sp_el2", "sp_el3"];
    for (index, name) in SP_NAMES.iter().enumerate() {
        if before.sp_el[index] != after.sp_el[index] {
            changes.push((name, before.sp_el[index], after.sp_el[index]));
        }
    }
    changes
}
impl<M: GuestMemory> Machine<M> {
    pub fn new(memory: M) -> Self {
        // `VOLT_INLINE_MEMORY=0` turns the inline data-TLB fast path off, for
        // A/B comparisons and for bisecting suspected emulation bugs.
        let cache = if std::env::var_os("VOLT_INLINE_MEMORY").is_some_and(|v| v == "0") {
            Cache::new()
        } else {
            Cache::with_inline_memory()
        };
        if let Some(reader) = memory.code_reader() {
            cache.start_ahead(reader, crate::aarch64::ahead::default_threads());
        }
        Self::with_cache(memory, 0, cache)
    }
    fn with_cache(memory: M, cpu_id: u32, cache: Cache) -> Self {
        Self {
            cpu_id,
            memory,
            cache,
            cpu: Cpu::default(),
            tlb: translate::Tlb::default(),
            dtlb_generation: 0,
            watch_epoch: 0,
            irq_line: false,
            vtimer_fiq: None,
            fiq_line: false,
            timer_fired: false,
            stalled_at: 0,
            stalled: None,
            unmapped: 0,
            trace: [TraceEntry::default(); 512],
            blocks_run: 0,
            faulted: false,
            device_access: false,
            step_trace: None,
            memory_trace: None,
            step_pending: None,
        }
    }
    /// A vCPU whose translated blocks are shared with other vCPUs.
    pub fn with_shared_blocks(
        memory: M,
        cpu_id: u32,
        blocks: std::sync::Arc<crate::aarch64::cache::SharedBlocks>,
    ) -> Self {
        Self::with_cache(memory, cpu_id, Cache::with_shared(blocks))
    }
    pub fn with_cpu_id(memory: M, cpu_id: u32) -> Self {
        Self {
            cpu_id,
            ..Self::new(memory)
        }
    }
    pub fn recent_trace(&self, count: usize) -> Vec<TraceEntry> {
        let shown = self.blocks_run.min(count.min(self.trace.len()) as u64);
        (self.blocks_run - shown..self.blocks_run)
            .map(|at| self.trace[at as usize % self.trace.len()])
            .collect()
    }
    pub fn timer_due(&self) -> bool {
        self.cpu.system.cntvct_el0 >= self.cpu.system.cntv_cval_el0
            && self.cpu.system.cntv_ctl_el0 & 1 != 0
            && self.cpu.system.cntv_ctl_el0 & 2 == 0
    }
    fn timer_edge(&mut self) -> bool {
        if !self.timer_due() {
            self.timer_fired = false;
            return false;
        }
        if self.timer_fired {
            return false;
        }
        self.timer_fired = true;
        true
    }
    /// Whether the interrupt controller presents the virtual timer to this CPU as
    /// a FIQ right now: the GIC forwards it (`vtimer_fiq`), the CPU interface has
    /// group 0 enabled and its priority mask lets the interrupt through, and the
    /// timer's condition holds. Evaluated from live state, so the line drops as soon
    /// as the guest masks the timer or moves its compare value.
    fn fiq_asserted(&self) -> bool {
        let system = &self.cpu.system;
        self.vtimer_fiq.is_some_and(|priority| {
            system.icc_igrpen0_el1 & 1 != 0
                && u64::from(priority) < system.icc_pmr_el1 & 0xff
                && self.timer_due()
        })
    }
    /// Installs a per-instruction trace. While set, blocks are compiled one instruction at
    /// a time, and `sink` sees each step with the watched system registers it changed.
    /// `None` removes the trace.
    pub fn set_step_trace(&mut self, sink: Option<Box<dyn FnMut(&Step) + Send>>) {
        self.step_trace = sink;
        self.rebuild_cache();
    }
    /// Installs a per-access trace: `sink` sees every load and store the guest makes,
    /// translated, except atomics and SIMD structure accesses. While set, the inline data
    /// TLB is off so every access reaches the host. `None` removes the trace.
    pub fn set_memory_trace(&mut self, sink: Option<Box<dyn FnMut(&MemoryAccess) + Send>>) {
        self.memory_trace = sink;
        self.rebuild_cache();
    }
    /// A fresh block cache for the trace state: single-step blocks while stepping, no inline
    /// data TLB while either trace is set, and the default cache otherwise. Compiled blocks
    /// and background compilation of the previous cache are dropped.
    fn rebuild_cache(&mut self) {
        let traced = self.step_trace.is_some() || self.memory_trace.is_some();
        let inline = !traced && !std::env::var_os("VOLT_INLINE_MEMORY").is_some_and(|v| v == "0");
        self.cache = if self.step_trace.is_some() {
            Cache::single_step(false)
        } else if inline {
            Cache::with_inline_memory()
        } else {
            Cache::new()
        };
        self.step_pending = None;
    }
    pub fn run(&mut self, serial: &mut Pl011, gic: &mut Gicv2) -> Result<Exit, Error> {
        self.run_with_devices(&mut (serial, gic))
    }
    /// Reports the step in progress, if any, with the registers it changed. A step ends
    /// when the next one starts or the run returns, so an exception entry the step
    /// caused is part of it.
    fn flush_step(&mut self) {
        if let Some((pc, before)) = self.step_pending.take()
            && let Some(sink) = self.step_trace.as_mut()
        {
            sink(&Step {
                pc,
                next_pc: self.cpu.pc,
                changes: system_changes(&before, &self.cpu.system),
            });
        }
    }
    pub fn run_with_devices(&mut self, devices: &mut impl DeviceIo) -> Result<Exit, Error> {
        let result = self.run_inner(devices);
        self.flush_step();
        result
    }
    fn run_inner(&mut self, devices: &mut impl DeviceIo) -> Result<Exit, Error> {
        self.stalled = None;
        for _ in 0..64 {
            if self.timer_edge() {
                return Ok(Exit::Timer);
            }
            self.flush_step();
            self.step_pending = self
                .step_trace
                .is_some()
                .then(|| (self.cpu.pc, self.cpu.system.clone()));
            let fiq = self.fiq_line || self.fiq_asserted();
            self.cpu.system.isr_el1 = u64::from(self.irq_line) << 7 | u64::from(fiq) << 6;
            let kind = if fiq && self.cpu.system.daif & 1 == 0 {
                Some(exception::Kind::Fiq)
            } else if self.irq_line && self.cpu.system.daif & 2 == 0 {
                Some(exception::Kind::Irq)
            } else {
                None
            };
            if let Some(kind) = kind {
                let pc = self.cpu.pc;
                exception::take(
                    &mut self.cpu,
                    exception::Reason {
                        kind,
                        ..Default::default()
                    },
                    pc,
                );
            }
            self.sync_dtlb();
            self.trace[self.blocks_run as usize % self.trace.len()] = TraceEntry {
                pc: self.cpu.pc,
                sp: self.cpu.sp,
            };
            let mut source = SystemFetch {
                tlb: &mut self.tlb,
                memory: &mut self.memory,
                fault: None,
            };
            let result = self.cache.run_block_with(&mut self.cpu, &mut source);
            let fetch_fault = source.fault;
            if let Err(error) = result {
                if let (Some((address, permission)), crate::aarch64::compile::Error::Memory(_)) =
                    (fetch_fault, &error)
                {
                    // Cache executes a valid lookahead prefix first. An error here means
                    // the failing instruction itself is now the architectural PC.
                    self.cpu.pc = address;
                    self.abort(address, permission, true, false);
                    continue;
                }
                if let crate::aarch64::compile::Error::Decode { word, pc } = &error
                    && crate::aarch64::decode::is_undefined(*word)
                {
                    // An Undefined Instruction exception (EC 0, 32-bit instruction).
                    self.cpu.pc = *pc;
                    exception::take(
                        &mut self.cpu,
                        exception::Reason {
                            ec: 0,
                            instruction: true,
                            status: 0,
                            ..Default::default()
                        },
                        0,
                    );
                    continue;
                }
                self.stalled_at = self.cpu.pc;
                self.stalled = Some(format!("{error:?}"));
                return Err(Error::Compile(error));
            }
            self.blocks_run += 1;
            match self.cpu.trap {
                Trap::None => {}
                Trap::Wfi => {
                    self.cpu.trap = Trap::None;
                    return Ok(if self.timer_edge() {
                        Exit::Timer
                    } else {
                        Exit::Wait
                    });
                }
                Trap::Wfe => {
                    self.cpu.trap = Trap::None;
                    if self.cpu.event_set {
                        self.cpu.event_set = false;
                        continue;
                    }
                    return Ok(if self.timer_edge() {
                        Exit::Timer
                    } else {
                        Exit::Wait
                    });
                }
                Trap::DcZva => {
                    self.cpu.trap = Trap::None;
                    let virtual_address = self.cpu.address & !63;
                    let physical = match translate::translate(
                        &mut self.tlb,
                        &self.cpu,
                        &mut self.memory,
                        virtual_address,
                        translate::Access::Write,
                    ) {
                        Ok(physical) => physical,
                        Err(error) => {
                            self.abort(
                                virtual_address,
                                matches!(error, translate::TranslateError::PermissionFault),
                                false,
                                true,
                            );
                            continue;
                        }
                    };
                    self.memory
                        .write(physical, &[0; 64])
                        .map_err(Error::Memory)?;
                }
                Trap::Eret => {
                    self.cpu.trap = Trap::None;
                    exception::eret(&mut self.cpu);
                }
                Trap::Svc => {
                    self.cpu.trap = Trap::None;
                    let pc = self.cpu.pc;
                    exception::take(
                        &mut self.cpu,
                        exception::Reason {
                            ec: 0x15,
                            status: 0,
                            instruction: true,
                            ..Default::default()
                        },
                        pc,
                    );
                }
                Trap::Brk => {
                    self.cpu.trap = Trap::None;
                    let pc = self.cpu.pc;
                    let immediate = self.cpu.address as u32;
                    // A breakpoint's syndrome is EC 0x3c with its immediate in ISS.
                    exception::take(
                        &mut self.cpu,
                        exception::Reason {
                            ec: 0x3c,
                            status: 0,
                            instruction: true,
                            iss: immediate,
                            ..Default::default()
                        },
                        pc,
                    );
                }
                Trap::Undefined => {
                    let pc = self.cpu.pc;
                    self.cpu.trap = Trap::None;
                    exception::take(
                        &mut self.cpu,
                        exception::Reason {
                            kind: exception::Kind::Sync,
                            ec: 0,
                            instruction: true,
                            status: 0,
                            iss: 0,
                        },
                        pc,
                    );
                }
                Trap::Isb => {
                    self.cpu.trap = Trap::None;
                }
                Trap::Tlbi => {
                    self.tlb.flush();
                    self.cpu.trap = Trap::None;
                }
                Trap::Host => {
                    let word = self.cpu.value as u32;
                    crate::aarch64::host::run(&mut self.cpu, word);
                    self.cpu.trap = Trap::None;
                }
                Trap::AddressTranslate => {
                    let (address, flags) = (self.cpu.address, self.cpu.value);
                    self.cpu.system.par_el1 =
                        self.address_translate(address, flags & 1 != 0, flags & 2 != 0);
                    self.cpu.trap = Trap::None;
                }
                Trap::Translation => {
                    let pc = self.cpu.pc;
                    self.abort(pc, false, true, false);
                }
                Trap::Psci => {
                    return Ok(Exit::Psci {
                        function: self.cpu.x[0] as u32,
                        args: [self.cpu.x[1], self.cpu.x[2], self.cpu.x[3]],
                    });
                }
                Trap::Load | Trap::Store => {
                    self.faulted = false;
                    self.device_access = false;
                    let (width, dest, _, exclusive) = self.cpu.memory_request();
                    self.cpu.exclusive = Exclusive::None;
                    let held = self.cpu.monitor_valid
                        && self.cpu.monitor_address == self.cpu.address
                        && self.cpu.monitor_width == width;
                    if exclusive == Exclusive::Load {
                        self.cpu.monitor_valid = true;
                        self.cpu.monitor_address = self.cpu.address;
                        self.cpu.monitor_width = width;
                    }
                    if exclusive == Exclusive::Store {
                        self.cpu.monitor_valid = false;
                    }
                    let mut success = held;
                    if exclusive == Exclusive::Store {
                        if held {
                            success = self.store_exclusive(
                                self.cpu.address,
                                width,
                                self.cpu.value,
                                devices,
                            )?;
                        }
                    } else {
                        self.access(self.cpu.address, width, dest, self.cpu.value, devices)?;
                    }
                    if exclusive == Exclusive::Load && !self.faulted {
                        let mask = if width >= 8 {
                            u64::MAX
                        } else {
                            (1u64 << (width * 8)) - 1
                        };
                        self.cpu.monitor_value = if dest == 31 {
                            0
                        } else {
                            self.cpu.x[dest as usize] & mask
                        };
                        // `ldxr xzr` discards the value, so it cannot be compared later.
                        self.cpu.monitor_valid = dest != 31;
                    }
                    if exclusive == Exclusive::Store && !self.faulted && self.cpu.status_dest != 31
                    {
                        self.cpu.x[self.cpu.status_dest as usize] = u64::from(!success);
                    }
                    if self.faulted {
                        self.cpu.second_pending = false;
                        self.cpu.writeback = false;
                        continue;
                    }
                    if self.cpu.second_pending {
                        self.cpu.second_pending = false;
                        self.cpu.vector_dest = self.cpu.second_vector;
                        self.cpu.second_vector = false;
                        let (second_width, second_dest, second_signed) =
                            self.cpu.second_memory_request();
                        self.cpu.load_signed = second_signed;
                        self.cpu.second_signed = SignExtend::None;
                        self.access(
                            self.cpu.second_address,
                            second_width,
                            second_dest,
                            self.cpu.second_value,
                            devices,
                        )?;
                    }
                    self.cpu.vector_dest = false;
                    if self.cpu.writeback && !self.faulted {
                        if self.cpu.writeback_dest == 31 {
                            self.cpu.sp = self.cpu.writeback_value;
                        } else {
                            self.cpu.x[self.cpu.writeback_dest as usize] = self.cpu.writeback_value;
                        }
                    }
                    self.cpu.writeback = false;
                    if self.device_access {
                        return Ok(Exit::Interrupted);
                    }
                }
                Trap::Atomic => {
                    self.faulted = false;
                    let Some(old) = self.atomic()? else {
                        continue;
                    };
                    // The words read go to the registers the instruction named;
                    // register 31 is XZR and discards them.
                    for (slot, value) in old
                        .into_iter()
                        .enumerate()
                        .take(self.cpu.atomic_kind.words())
                    {
                        let dest = usize::from(self.cpu.atomic_dests[slot]);
                        if dest < 31 {
                            self.cpu.x[dest] = value;
                        }
                    }
                }
                Trap::SimdStruct => {
                    self.faulted = false;
                    self.device_access = false;
                    self.simd_struct(devices)?;
                    if self.faulted {
                        self.cpu.writeback = false;
                        continue;
                    }
                    if self.cpu.writeback {
                        if self.cpu.writeback_dest == 31 {
                            self.cpu.sp = self.cpu.writeback_value;
                        } else {
                            self.cpu.x[self.cpu.writeback_dest as usize] = self.cpu.writeback_value;
                        }
                    }
                    self.cpu.writeback = false;
                    if self.device_access {
                        return Ok(Exit::Interrupted);
                    }
                }
            }
        }
        Ok(Exit::Interrupted)
    }
    /// Bring the inline data TLB in line with the translation state. Generated
    /// code hits it without calling `translate`, so every reason an entry could
    /// have become wrong must be handled here, before the next block runs.
    fn sync_dtlb(&mut self) {
        // Context changes (TTBR/TCR/SCTLR writes, exception entry and return)
        // are noticed lazily by `translate`; do it eagerly instead.
        self.tlb.update_context(&self.cpu);
        if self.tlb.generation != self.dtlb_generation {
            self.dtlb_generation = self.tlb.generation;
            self.cpu.dtlb.flush();
        }
        // Pages that just became code must stop accepting inline writes, which
        // would bypass the code-page write tracking.
        if let Some(tracker) = self.memory.code_tracker() {
            let epoch = tracker.epoch();
            if epoch != self.watch_epoch {
                self.watch_epoch = epoch;
                self.cpu.dtlb.purge_writes();
            }
        }
    }
    /// Remember a completed RAM access in the inline data TLB so later accesses
    /// to the same page skip the machine entirely.
    fn fill_dtlb(&mut self, address: u64, physical: u64, write: bool) {
        let page = address & !0xfff;
        let physical_page = physical & !0xfff;
        let Some(host) = self.memory.host_page(physical_page) else {
            return;
        };
        let addend = (host as u64).wrapping_sub(page);
        let writable = write
            && self
                .memory
                .code_tracker()
                .and_then(|tracker| tracker.version(physical_page >> 12))
                .is_none_or(|version| version & 1 == 0);
        let entry = &mut self.cpu.dtlb.entries[crate::aarch64::cpu::Dtlb::index(address)];
        if entry.addend != addend || (entry.read != page && entry.write != page) {
            *entry = Default::default();
        }
        entry.addend = addend;
        if write {
            if writable {
                entry.write = page;
            }
        } else {
            entry.read = page;
        }
    }
    fn abort(&mut self, address: u64, permission: bool, instruction: bool, write: bool) {
        let status = match (permission, instruction) {
            (false, _) => exception::status::TRANSLATION,
            (true, false) => exception::status::PERMISSION,
            (true, true) => exception::status::PERMISSION_FETCH,
        };
        self.abort_with_status(address, status, instruction, write);
    }
    /// A synchronous abort with fault status `status`.
    fn abort_with_status(&mut self, address: u64, status: u8, instruction: bool, write: bool) {
        // EC 0x20/0x24 for a lower-EL abort, 0x21/0x25 for one taken from EL1 or EL2.
        let same_el = self.cpu.system.el != 0;
        let ec = match (instruction, same_el) {
            (true, false) => 0x20,
            (true, true) => 0x21,
            (false, false) => 0x24,
            (false, true) => 0x25,
        };
        self.cpu.trap = Trap::None;
        if !instruction {
            self.cpu.pc = self.cpu.pc.wrapping_sub(4);
        }
        exception::take(
            &mut self.cpu,
            exception::Reason {
                kind: exception::Kind::Sync,
                ec,
                instruction,
                status,
                iss: 0,
            },
            address,
        );
        if write {
            self.cpu.system.esr_el1 |= 1 << 6;
        }
        self.faulted = true;
    }
    /// Store-exclusive: succeeds only if the location still holds the value the
    /// paired load-exclusive saw. RAM uses a host compare-exchange, so it is
    /// atomic against every other vCPU (and against their inline stores).
    /// Returns whether the store happened; a fault aborts and returns `false`.
    fn store_exclusive(
        &mut self,
        address: u64,
        width: u8,
        value: u64,
        devices: &mut impl DeviceIo,
    ) -> Result<bool, Error> {
        use core::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering::SeqCst};
        if !matches!(width, 1 | 2 | 4 | 8) {
            return Err(Error::InvalidAccessWidth(width));
        }
        let physical = match translate::translate(
            &mut self.tlb,
            &self.cpu,
            &mut self.memory,
            address,
            translate::Access::Write,
        ) {
            Ok(physical) => physical,
            Err(error) => {
                self.abort(
                    address,
                    matches!(error, translate::TranslateError::PermissionFault),
                    false,
                    true,
                );
                return Ok(false);
            }
        };
        if physical % u64::from(width) == 0
            && let Some(page) = self.memory.host_page(physical & !0xfff)
        {
            let mask = if width == 8 {
                u64::MAX
            } else {
                (1u64 << (width * 8)) - 1
            };
            let (expected, new) = (self.cpu.monitor_value & mask, value & mask);
            // SAFETY: `page` is a live RAM page; the offset stays inside it and
            // is aligned to `width`.
            let stored = unsafe {
                let at = page.add((physical & 0xfff) as usize);
                match width {
                    1 => AtomicU8::from_ptr(at)
                        .compare_exchange(expected as u8, new as u8, SeqCst, SeqCst)
                        .is_ok(),
                    2 => AtomicU16::from_ptr(at.cast())
                        .compare_exchange(expected as u16, new as u16, SeqCst, SeqCst)
                        .is_ok(),
                    4 => AtomicU32::from_ptr(at.cast())
                        .compare_exchange(expected as u32, new as u32, SeqCst, SeqCst)
                        .is_ok(),
                    _ => AtomicU64::from_ptr(at.cast())
                        .compare_exchange(expected, new, SeqCst, SeqCst)
                        .is_ok(),
                }
            };
            if stored && let Some(tracker) = self.memory.code_tracker() {
                tracker.note_write(physical, width as usize);
            }
            return Ok(stored);
        }
        // Device memory or a misaligned address: an ordinary store.
        self.access(address, width, 31, value, devices)?;
        Ok(!self.faulted)
    }
    /// `AT S1E{1,0}{R,W}`: the `PAR_EL1` value of a stage-1 walk of `address`. Success
    /// holds the page's physical address; failure sets `F` with a fault status code
    /// (level 3 translation or permission fault).
    fn address_translate(&mut self, address: u64, write: bool, user: bool) -> u64 {
        let access = if write {
            translate::Access::Write
        } else {
            translate::Access::Read
        };
        let result = if user {
            // Walk with EL0 permissions on a scratch TLB: switching the context of
            // the real one would flush it.
            let mut cpu = self.cpu.clone();
            cpu.system.el = 0;
            translate::translate(
                &mut translate::Tlb::new(),
                &cpu,
                &mut self.memory,
                address,
                access,
            )
        } else {
            translate::translate(&mut self.tlb, &self.cpu, &mut self.memory, address, access)
        };
        match result {
            Ok(physical) => (physical & 0x0000_ffff_ffff_f000) | 1 << 11,
            Err(translate::TranslateError::PermissionFault) => 1 | 0b001111 << 1,
            Err(_) => 1 | 0b000111 << 1,
        }
    }
    /// SIMD structure load/store: `simd_struct::execute` over the vector registers,
    /// one element per `access`. The registers change only if every element succeeded.
    fn simd_struct(&mut self, devices: &mut impl DeviceIo) -> Result<(), Error> {
        let desc = self.cpu.structure;
        let saved = self.cpu.v;
        let mut registers = saved;
        // `access` takes its direction from the trap kind.
        self.cpu.trap = if desc.store { Trap::Store } else { Trap::Load };
        // `Err(None)` is a guest fault `access` already recorded; `Some` is fatal.
        let result = simd_struct::execute(
            &desc,
            &mut registers,
            self.cpu.address,
            |address, width, store| {
                if let Some(value) = store {
                    self.cpu.vector_dest = false;
                    self.access(address, width, 31, value, devices)
                        .map_err(Some)?;
                } else {
                    // `access` loads whole registers: use v0 as scratch and merge the
                    // element into `registers` (v0 is restored below).
                    self.cpu.vector_dest = true;
                    self.access(address, width, 0, 0, devices).map_err(Some)?;
                }
                if self.faulted {
                    return Err(None);
                }
                Ok(self.cpu.v[0][0])
            },
        );
        self.cpu.vector_dest = false;
        self.cpu.trap = Trap::None;
        self.cpu.v = saved;
        match result {
            Ok(()) => {
                self.cpu.v = registers;
                Ok(())
            }
            Err(None) => Ok(()),
            Err(Some(error)) => Err(error),
        }
    }
    /// LSE atomic (`ldadd`, `swp`, `cas`, `casp`, ...) on RAM. Every LSE operation
    /// holds one global lock while it reads, combines and writes back, so a CASP
    /// pair stays consistent against every other LSE operation on any vCPU; each
    /// word is still accessed through a host atomic. Plain stores that bypass this
    /// path (inline JIT stores, `stxr`) are not ordered against the lock.
    /// Returns the words read, or `None` if translation faulted.
    fn atomic(&mut self) -> Result<Option<[u64; 2]>, Error> {
        static LSE: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
        let (kind, address, width) = (self.cpu.atomic_kind, self.cpu.address, self.cpu.width);
        if !matches!(width, 1 | 2 | 4 | 8) {
            return Err(Error::InvalidAccessWidth(width));
        }
        let words = kind.words();
        let span = u64::from(width) * words as u64;
        let physical = match translate::translate(
            &mut self.tlb,
            &self.cpu,
            &mut self.memory,
            address,
            translate::Access::Write,
        ) {
            Ok(physical) => physical,
            Err(error) => {
                self.abort(
                    address,
                    matches!(error, translate::TranslateError::PermissionFault),
                    false,
                    true,
                );
                return Ok(None);
            }
        };
        // Aligned to the whole span, so every word lies in this one page.
        let page = (physical % span == 0)
            .then(|| self.memory.host_page(physical & !0xfff))
            .flatten()
            .ok_or(Error::UnsupportedAtomic(address))?;
        let width_bytes = usize::from(width);
        // SAFETY: `page` is a live RAM page and `physical` is aligned to `span`, so
        // each word's address is inside the page and aligned to `width`.
        let word =
            |slot: usize| unsafe { page.add((physical & 0xfff) as usize + slot * width_bytes) };
        let guard = LSE.lock();
        let mut old = [0u64; 2];
        for (slot, value) in old.iter_mut().enumerate().take(words) {
            *value = Self::load_word(word(slot), width);
        }
        let new = if kind.is_exclusive_pair() {
            self.cpu.exclusive_pair(kind, old)
        } else {
            kind.apply(old, self.cpu.atomic_operands, width)
        };
        let mut changed = false;
        for slot in 0..words {
            if new[slot] != old[slot] {
                Self::store_word(word(slot), width, new[slot]);
                changed = true;
            }
        }
        drop(guard);
        if changed && let Some(tracker) = self.memory.code_tracker() {
            tracker.note_write(physical, span as usize);
        }
        Ok(Some(old))
    }
    fn load_word(at: *mut u8, width: u8) -> u64 {
        use core::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering::SeqCst};
        // SAFETY: callers pass an address inside a live RAM page, aligned to `width`.
        unsafe {
            match width {
                1 => u64::from(AtomicU8::from_ptr(at).load(SeqCst)),
                2 => u64::from(AtomicU16::from_ptr(at.cast()).load(SeqCst)),
                4 => u64::from(AtomicU32::from_ptr(at.cast()).load(SeqCst)),
                _ => AtomicU64::from_ptr(at.cast()).load(SeqCst),
            }
        }
    }
    fn store_word(at: *mut u8, width: u8, value: u64) {
        use core::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering::SeqCst};
        // SAFETY: callers pass an address inside a live RAM page, aligned to `width`.
        unsafe {
            match width {
                1 => AtomicU8::from_ptr(at).store(value as u8, SeqCst),
                2 => AtomicU16::from_ptr(at.cast()).store(value as u16, SeqCst),
                4 => AtomicU32::from_ptr(at.cast()).store(value as u32, SeqCst),
                _ => AtomicU64::from_ptr(at.cast()).store(value, SeqCst),
            }
        }
    }
    fn access(
        &mut self,
        address: u64,
        width: u8,
        dest: u8,
        value: u64,
        devices: &mut impl DeviceIo,
    ) -> Result<(), Error> {
        if !matches!(width, 1 | 2 | 4 | 8 | 16) {
            return Err(Error::InvalidAccessWidth(width));
        }
        let load = self.cpu.trap == Trap::Load;
        let signed = self.cpu.memory_request().2;
        self.cpu.load_signed = SignExtend::None;
        let access = if load {
            translate::Access::Read
        } else {
            translate::Access::Write
        };
        let physical =
            match translate::translate(&mut self.tlb, &self.cpu, &mut self.memory, address, access)
            {
                Ok(physical) => physical,
                Err(error) => {
                    self.abort(
                        address,
                        matches!(error, translate::TranslateError::PermissionFault),
                        false,
                        !load,
                    );
                    return Ok(());
                }
            };
        if let Some(sink) = self.memory_trace.as_mut() {
            sink(&MemoryAccess {
                pc: self.cpu.pc.wrapping_sub(4),
                virtual_address: address,
                physical,
                width,
                write: !load,
                value: if load { 0 } else { value },
            });
        }
        let mut bytes = [0u8; 16];
        if !load {
            if self.cpu.vector_dest {
                let v = self.cpu.v[dest as usize];
                bytes[0..8].copy_from_slice(&v[0].to_le_bytes());
                bytes[8..16].copy_from_slice(&v[1].to_le_bytes());
                if width != 16 {
                    bytes[8..16].fill(0);
                }
            } else {
                bytes[0..8].copy_from_slice(&value.to_le_bytes());
            }
        }
        let first = (4096 - (address & 4095) as usize).min(width as usize);
        let result = if first == width as usize {
            if load {
                self.memory.read(physical, &mut bytes[..first])
            } else {
                self.memory.write(physical, &bytes[..first])
            }
        } else {
            let second_address = address.wrapping_add(first as u64);
            let second_physical = match translate::translate(
                &mut self.tlb,
                &self.cpu,
                &mut self.memory,
                second_address,
                access,
            ) {
                Ok(physical) => physical,
                Err(error) => {
                    self.abort(
                        second_address,
                        matches!(error, translate::TranslateError::PermissionFault),
                        false,
                        !load,
                    );
                    return Ok(());
                }
            };
            let result = if load {
                self.memory
                    .read(physical, &mut bytes[..first])
                    .and_then(|()| {
                        self.memory
                            .read(second_physical, &mut bytes[first..width as usize])
                    })
            } else {
                self.memory.write(physical, &bytes[..first]).and_then(|()| {
                    self.memory
                        .write(second_physical, &bytes[first..width as usize])
                })
            };
            if matches!(result, Err(MemoryError::AccessFault { .. })) {
                self.unmapped += 1;
                self.abort_with_status(address, exception::status::SYNC_EXTERNAL, false, !load);
                return Ok(());
            }
            result
        };
        match result {
            Ok(()) => {}
            Err(MemoryError::AccessFault { .. }) => {
                self.device_access = true;
                let end = physical.checked_add(width as u64);
                let fixed = devices.has_fixed_windows();
                let window = |base, length| {
                    fixed && physical >= base && end.is_some_and(|end| end <= base + length)
                };
                let soc = if load {
                    devices.read_soc(self.cpu_id, physical, width).map(Some)
                } else {
                    devices
                        .write_soc(self.cpu_id, physical, width, value)
                        .then_some(None)
                };
                let read = if let Some(read) = soc {
                    read
                } else if window(fdt::UART_BASE, crate::devices::pl011::LEN) {
                    if load {
                        Some(devices.read_uart(physical - fdt::UART_BASE, width))
                    } else {
                        devices.write_uart(physical - fdt::UART_BASE, width, value);
                        None
                    }
                } else if window(fdt::GICD_BASE, crate::devices::gicv3::DISTRIBUTOR_SIZE) {
                    if load {
                        Some(devices.read_gic_distributor(
                            self.cpu_id,
                            physical - fdt::GICD_BASE,
                            width,
                        ))
                    } else {
                        devices.write_gic_distributor(
                            self.cpu_id,
                            physical - fdt::GICD_BASE,
                            width,
                            value,
                        );
                        None
                    }
                } else if window(fdt::GIC_CPU_BASE, crate::devices::gicv2::CPU_SIZE) {
                    if load {
                        Some(devices.read_gic_cpu(self.cpu_id, physical - fdt::GIC_CPU_BASE, width))
                    } else {
                        devices.write_gic_cpu(
                            self.cpu_id,
                            physical - fdt::GIC_CPU_BASE,
                            width,
                            value,
                        );
                        None
                    }
                } else if window(afdt::GICR_BASE, crate::devices::gicv3::REDISTRIBUTOR_SIZE) {
                    if load {
                        Some(devices.read_gic_redistributor(physical - afdt::GICR_BASE, width))
                    } else {
                        devices.write_gic_redistributor(physical - afdt::GICR_BASE, width, value);
                        None
                    }
                } else if window(fdt::VIRTIO_MMIO_BASE, fdt::VIRTIO_MMIO_SIZE) {
                    if load {
                        Some(devices.read_virtio(physical - fdt::VIRTIO_MMIO_BASE, width))
                    } else {
                        devices.write_virtio(physical - fdt::VIRTIO_MMIO_BASE, width, value);
                        None
                    }
                } else {
                    self.unmapped += 1;
                    self.abort_with_status(address, exception::status::SYNC_EXTERNAL, false, !load);
                    return Ok(());
                };
                if let Some(value) = read {
                    bytes[0..8].copy_from_slice(&value.to_le_bytes());
                }
            }
            Err(error) => return Err(Error::Memory(error)),
        }
        if first == width as usize && !self.device_access && self.cache.inline_memory() {
            self.fill_dtlb(address, physical, !load);
        }
        if load {
            if self.cpu.vector_dest {
                let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                let hi = if width == 16 {
                    u64::from_le_bytes(bytes[8..16].try_into().unwrap())
                } else {
                    0
                };
                let bits = (width as u32).min(8) * 8;
                let mask = if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                self.cpu.v[dest as usize] = [lo & mask, hi];
            } else if dest != 31 {
                let value = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                let bits = width as u32 * 8;
                let kept = if bits == 64 {
                    value
                } else {
                    value & ((1u64 << bits) - 1)
                };
                self.cpu.x[dest as usize] = match signed {
                    SignExtend::None => kept,
                    SignExtend::To32 => {
                        (((kept << (64 - bits)) as i64 >> (64 - bits)) as u64) & 0xffff_ffff
                    }
                    SignExtend::To64 => ((kept << (64 - bits)) as i64 >> (64 - bits)) as u64,
                };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::memory::{PhysicalMemory, Region};

    fn machine() -> Machine<PhysicalMemory> {
        Machine::new(
            PhysicalMemory::new(vec![Region::ram(0x40000000, vec![0; 4096]).unwrap()]).unwrap(),
        )
    }

    #[test]
    fn timer_reports_one_rising_edge_per_due_condition() {
        let mut machine = machine();
        machine.cpu.system.cntv_ctl_el0 = 1;
        machine.cpu.system.cntv_cval_el0 = 10;
        machine.cpu.system.cntvct_el0 = 9;
        assert!(!machine.timer_edge());
        machine.cpu.system.cntvct_el0 = 10;
        assert!(machine.timer_edge());
        assert!(!machine.timer_edge());
        machine.cpu.system.cntv_ctl_el0 = 3;
        assert!(!machine.timer_edge());
        machine.cpu.system.cntv_ctl_el0 = 1;
        assert!(machine.timer_edge());
    }

    #[test]
    fn narrow_reads_extend_to_the_selected_register_width() {
        let mut machine = machine();
        machine.memory.write(0x40000000, &[0x80]).unwrap();
        let mut serial = Pl011::new(|_| {});
        let mut gic = Gicv2::default();
        machine.cpu.trap = Trap::Load;
        machine.cpu.load_signed = SignExtend::To64;
        machine
            .access(0x40000000, 1, 0, 0, &mut (&mut serial, &mut gic))
            .unwrap();
        assert_eq!(machine.cpu.x[0], 0xffff_ffff_ffff_ff80);
        machine.cpu.load_signed = SignExtend::To32;
        machine
            .access(0x40000000, 1, 1, 0, &mut (&mut serial, &mut gic))
            .unwrap();
        assert_eq!(machine.cpu.x[1], 0xffff_ff80);
        assert_eq!(machine.cpu.load_signed, SignExtend::None);
    }

    #[test]
    fn unmapped_device_access_enters_guest_data_abort_and_retries_instruction() {
        let mut machine = machine();
        machine.cpu.pc = 0x40000004;
        machine.cpu.system.vbar_el1 = 0x40000800;
        machine.cpu.trap = Trap::Store;
        let mut serial = Pl011::new(|_| {});
        let mut gic = Gicv2::default();
        machine
            .access(0x0b000000, 4, 31, 123, &mut (&mut serial, &mut gic))
            .unwrap();
        assert_eq!(machine.cpu.system.elr_el1, 0x40000000);
        assert_eq!(machine.cpu.system.far_el1, 0x0b000000);
        assert_eq!(machine.cpu.system.esr_el1 >> 26, 0x25);
        assert_ne!(machine.cpu.system.esr_el1 & (1 << 6), 0);
        assert_eq!(machine.cpu.pc, 0x40000a00);
        assert!(machine.faulted);
        assert_eq!(machine.unmapped, 1);
    }

    #[test]
    fn instruction_abort_follows_retired_fetch_prefix_at_the_actual_fault_pc() {
        let mut machine = machine();
        machine.cpu.pc = 0x40000ffc;
        machine.cpu.system.vbar_el1 = 0x40000000;
        machine
            .memory
            .write(0x40000ffc, &0xd2800020u32.to_le_bytes())
            .unwrap(); // movz x0, #1
        machine
            .memory
            .write(0x40000200, &0xd4000002u32.to_le_bytes())
            .unwrap(); // hvc #0 at synchronous vector
        let mut serial = Pl011::new(|_| {});
        let mut gic = Gicv2::default();
        let exit = machine.run(&mut serial, &mut gic).unwrap();
        assert!(matches!(exit, Exit::Psci { function: 1, .. }));
        assert_eq!(machine.cpu.system.elr_el1, 0x40001000);
        assert_eq!(machine.cpu.system.far_el1, 0x40001000);
        assert_eq!(machine.cpu.system.esr_el1 >> 26, 0x21);
        assert_eq!(machine.cpu.system.cntvct_el0, 2);
    }

    #[test]
    fn an_architecturally_undefined_word_raises_undef_after_the_decoded_prefix_ran() {
        let mut machine = machine();
        machine.cpu.pc = 0x40000000;
        machine.cpu.system.vbar_el1 = 0x40000000;
        // movz x0, #1; XNU's TRAP_DEBUGGER word; the guest's synchronous vector.
        for (at, word) in [
            (0x40000000, 0xd2800020u32),
            (0x40000004, 0xe7ffdeff),
            (0x40000200, 0xd4000002),
        ] {
            machine.memory.write(at, &word.to_le_bytes()).unwrap();
        }
        let mut serial = Pl011::new(|_| {});
        let mut gic = Gicv2::default();
        let exit = machine.run(&mut serial, &mut gic).unwrap();
        assert!(matches!(exit, Exit::Psci { function: 1, .. }));
        assert_eq!(machine.cpu.x[0], 1, "the instruction before it ran");
        assert_eq!(
            machine.cpu.system.elr_el1, 0x40000004,
            "the exception is taken at the word"
        );
        assert_eq!(machine.cpu.system.esr_el1 >> 26, 0, "unknown reason");
        assert_ne!(
            machine.cpu.system.esr_el1 & (1 << 25),
            0,
            "32-bit instruction"
        );
    }

    #[test]
    fn the_virtual_timer_is_a_fiq_only_while_every_gate_is_open() {
        // (F masked, forwarded by the GIC, ICC_IGRPEN0, ICC_PMR, timer CTL, expect FIQ)
        let cases = [
            (false, true, 1, 0xff, 1, true),
            (true, true, 1, 0xff, 1, false),
            (false, false, 1, 0xff, 1, false),
            (false, true, 0, 0xff, 1, false),
            (false, true, 1, 0x00, 1, false),
            (false, true, 1, 0xff, 3, false),
        ];
        for (masked, forwarded, group0, pmr, ctl, expect) in cases {
            let mut machine = machine();
            machine.cpu.pc = 0x40000000;
            machine.cpu.system.vbar_el1 = 0x40000000;
            machine.cpu.system.daif = u64::from(masked);
            machine.vtimer_fiq = forwarded.then_some(0);
            machine.cpu.system.icc_igrpen0_el1 = group0;
            machine.cpu.system.icc_pmr_el1 = pmr;
            machine.cpu.system.cntv_ctl_el0 = ctl;
            // The mainline code and the FIQ vector (current EL, SPx: +0x200 + 2 * 0x80)
            // both end in `hvc`; only a taken FIQ sets ELR.
            for at in [0x40000000, 0x40000300] {
                machine
                    .memory
                    .write(at, &0xd4000002u32.to_le_bytes())
                    .unwrap();
            }
            let mut serial = Pl011::new(|_| {});
            let mut gic = Gicv2::default();
            let exit = loop {
                match machine.run(&mut serial, &mut gic).unwrap() {
                    Exit::Timer => continue,
                    other => break other,
                }
            };
            assert!(matches!(exit, Exit::Psci { .. }));
            assert_eq!(
                machine.cpu.system.elr_el1 == 0x40000000,
                expect,
                "masked={masked} forwarded={forwarded} group0={group0} pmr={pmr:#x} ctl={ctl}"
            );
            // `ISR_EL1.F` reports the asserted line whether or not the CPU masks it.
            let asserted = forwarded && group0 == 1 && pmr > 0 && ctl == 1;
            assert_eq!(machine.cpu.system.isr_el1 & (1 << 6) != 0, asserted);
        }
    }

    #[test]
    fn address_translation_reports_the_physical_page_in_par() {
        // `at s1e1r, x0` with the MMU off translates to the same address.
        let mut machine = machine();
        machine.cpu.x[0] = 0x40000123;
        run_to_hvc(&mut machine, &[0xd5087800, HVC]);
        assert_eq!(machine.cpu.system.par_el1 & 1, 0, "no fault");
        assert_eq!(
            machine.cpu.system.par_el1 & 0x0000_ffff_ffff_f000,
            0x40000000
        );
    }

    /// Run guest code at `0x40000000` until it hits the `hvc` that ends it.
    fn run_to_hvc(machine: &mut Machine<PhysicalMemory>, words: &[u32]) {
        for (i, word) in words.iter().enumerate() {
            machine
                .memory
                .write(0x40000000 + 4 * i as u64, &word.to_le_bytes())
                .unwrap();
        }
        machine.cpu.pc = 0x40000000;
        let mut serial = Pl011::new(|_| {});
        let mut gic = Gicv2::default();
        assert!(matches!(
            machine.run(&mut serial, &mut gic).unwrap(),
            Exit::Psci { .. }
        ));
    }

    #[test]
    fn brk_takes_an_el1_breakpoint_exception_with_its_immediate_in_the_syndrome() {
        let mut machine = machine();
        machine.cpu.system.el = 1;
        machine.cpu.system.spsel = true;
        machine.cpu.system.vbar_el1 = 0x40000400;
        // `brk #0x800`, the immediate the kernel's `BUG()` uses.
        machine
            .memory
            .write(0x40000000, &0xd4210000u32.to_le_bytes())
            .unwrap();
        // The vector for a synchronous exception from EL1 on SP_EL1: `VBAR + 0x200`.
        machine
            .memory
            .write(0x40000600, &HVC.to_le_bytes())
            .unwrap();
        machine.cpu.pc = 0x40000000;
        let mut serial = Pl011::new(|_| {});
        let mut gic = Gicv2::default();
        assert!(matches!(
            machine.run(&mut serial, &mut gic).unwrap(),
            Exit::Psci { .. }
        ));
        let esr = machine.cpu.system.esr_el1;
        assert_eq!(esr >> 26, 0x3c, "EC is a breakpoint");
        assert_eq!((esr >> 25) & 1, 1, "IL: a 32-bit instruction");
        assert_eq!(esr & 0xffff, 0x800, "ISS is the immediate");
        assert_eq!(
            machine.cpu.system.elr_el1, 0x40000000,
            "ELR is the BRK itself"
        );
    }

    #[test]
    fn el2_with_e2h_sends_el1_register_names_to_the_el2_banks() {
        let mut machine = machine();
        machine.cpu.system.el = 2;
        machine.cpu.system.spsel = true;
        machine.cpu.system.hcr_el2 = crate::aarch64::cpu::HCR_E2H;
        machine.cpu.system.vbar_el1 = 0x1111;
        machine.cpu.system.vbar_el2 = 0x2222;
        // movz x0, #0x1234; msr vbar_el1, x0; mrs x1, vbar_el1
        run_to_hvc(&mut machine, &[0xd2824680, 0xd518c000, 0xd538c001, HVC]);
        assert_eq!(machine.cpu.x[1], 0x1234);
        assert_eq!(machine.cpu.system.vbar_el2, 0x1234);
        assert_eq!(machine.cpu.system.vbar_el1, 0x1111);
    }

    #[test]
    fn el2_without_e2h_writes_the_el1_bank_and_leaves_el2_alone() {
        let mut machine = machine();
        machine.cpu.system.el = 2;
        machine.cpu.system.spsel = true;
        machine.cpu.system.hcr_el2 = 0;
        machine.cpu.system.vbar_el1 = 0x1111;
        machine.cpu.system.vbar_el2 = 0x2222;
        run_to_hvc(&mut machine, &[0xd2824680, 0xd518c000, 0xd538c001, HVC]);
        assert_eq!(machine.cpu.x[1], 0x1234);
        assert_eq!(machine.cpu.system.vbar_el1, 0x1234);
        assert_eq!(machine.cpu.system.vbar_el2, 0x2222);
    }

    #[test]
    fn step_trace_reports_each_instruction_and_the_system_register_it_changed() {
        use parking_lot::Mutex;
        use std::sync::Arc;
        let mut machine = machine();
        machine.cpu.system.el = 1;
        machine.cpu.system.spsel = true;
        let steps = Arc::new(Mutex::new(Vec::new()));
        let sink = steps.clone();
        machine.set_step_trace(Some(Box::new(move |step| {
            sink.lock().push(step.clone());
        })));
        // movz x0, #1; msr vbar_el1, x0; hvc
        run_to_hvc(&mut machine, &[0xd2800020, 0xd518c000, HVC]);
        let steps = steps.lock();
        let pcs: Vec<u64> = steps.iter().map(|s| s.pc).collect();
        assert_eq!(pcs, [0x40000000, 0x40000004, 0x40000008]);
        assert!(steps[0].changes.is_empty());
        assert_eq!(steps[1].changes, [("vbar_el1", 0, 1)]);
        assert_eq!(steps[1].next_pc, 0x40000008);
    }
    #[test]
    fn memory_trace_reports_each_translated_access_with_its_pc_and_stored_value() {
        use parking_lot::Mutex;
        use std::sync::Arc;
        let mut machine = machine();
        machine.cpu.system.el = 1;
        machine.cpu.system.spsel = true;
        let accesses = Arc::new(Mutex::new(Vec::new()));
        let sink = accesses.clone();
        machine.set_memory_trace(Some(Box::new(move |access| sink.lock().push(*access))));
        // movz x0,#0x1234; movz x1,#0x4000,lsl #16; add x1,x1,#0x800; str x0,[x1];
        // ldr w2,[x1,#4]
        run_to_hvc(
            &mut machine,
            &[
                0xd2824680, 0xd2a80001, 0x91200021, 0xf9000020, 0xb9400422, HVC,
            ],
        );
        let accesses = accesses.lock();
        assert_eq!(
            *accesses,
            [
                MemoryAccess {
                    pc: 0x4000000c,
                    virtual_address: 0x40000800,
                    physical: 0x40000800,
                    width: 8,
                    write: true,
                    value: 0x1234,
                },
                MemoryAccess {
                    pc: 0x40000010,
                    virtual_address: 0x40000804,
                    physical: 0x40000804,
                    width: 4,
                    write: false,
                    value: 0,
                },
            ]
        );
    }
    const LDXR_X1_X0: u32 = 0xc85f7c01;
    const STXR_W2_X3_X0: u32 = 0xc8027c03;
    const HVC: u32 = 0xd4000002;
    #[test]
    fn ld1_of_several_registers_loads_every_register_and_post_indexes() {
        // `ld1.2d {v16, v17, v18, v19}, [x10]`, then the same with `, #64`.
        for (word, advance) in [(0x4c402d50, 0), (0x4cdf2d50, 64)] {
            let mut machine = machine();
            machine.cpu.x[10] = 0x40000800;
            for i in 0..8u64 {
                machine
                    .memory
                    .write(0x40000800 + 8 * i, &(11 + i).to_le_bytes())
                    .unwrap();
            }
            run_to_hvc(&mut machine, &[word, HVC]);
            assert_eq!(machine.cpu.v[16], [11, 12]);
            assert_eq!(machine.cpu.v[17], [13, 14]);
            assert_eq!(machine.cpu.v[18], [15, 16]);
            assert_eq!(machine.cpu.v[19], [17, 18]);
            assert_eq!(machine.cpu.x[10], 0x40000800 + advance);
        }
    }

    const LDXP_X9_X21_X8: u32 = 0xc87f5509;
    /// `stxp w11, x19, x10, [x8]`.
    const STXP_W11_X19_X10_X8: u32 = 0xc82b2913;

    #[test]
    fn exclusive_pair_stores_only_while_both_words_are_unchanged() {
        let read_pair = |machine: &mut Machine<PhysicalMemory>| {
            let mut bytes = [0u8; 16];
            machine.memory.read(0x40000800, &mut bytes).unwrap();
            [
                u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            ]
        };
        for (interfere, status, stored) in [(false, 0, [30, 40]), (true, 1, [1, 99])] {
            let mut machine = machine();
            machine.cpu.x[8] = 0x40000800;
            machine.cpu.x[19] = 30;
            machine.cpu.x[10] = 40;
            let mut init = [0u8; 16];
            init[..8].copy_from_slice(&1u64.to_le_bytes());
            init[8..].copy_from_slice(&2u64.to_le_bytes());
            machine.memory.write(0x40000800, &init).unwrap();
            run_to_hvc(&mut machine, &[LDXP_X9_X21_X8, HVC]);
            assert_eq!(
                (machine.cpu.x[9], machine.cpu.x[21]),
                (1, 2),
                "ldxp loads both words"
            );
            if interfere {
                // Another agent changes the second word before the store.
                machine
                    .memory
                    .write(0x40000808, &99u64.to_le_bytes())
                    .unwrap();
            }
            run_to_hvc(&mut machine, &[STXP_W11_X19_X10_X8, HVC]);
            assert_eq!(machine.cpu.x[11], status, "interfere = {interfere}");
            assert_eq!(read_pair(&mut machine), stored);
            // The monitor is spent: a second stxp without a new ldxp fails.
            run_to_hvc(&mut machine, &[STXP_W11_X19_X10_X8, HVC]);
            assert_eq!(machine.cpu.x[11], 1);
        }
    }

    #[test]
    fn store_exclusive_succeeds_when_memory_is_unchanged() {
        let mut machine = machine();
        machine.cpu.x[0] = 0x40000800;
        machine.cpu.x[3] = 7;
        machine
            .memory
            .write(0x40000800, &5u64.to_le_bytes())
            .unwrap();
        run_to_hvc(&mut machine, &[LDXR_X1_X0, STXR_W2_X3_X0, HVC]);
        assert_eq!(machine.cpu.x[1], 5, "ldxr loaded the old value");
        assert_eq!(machine.cpu.x[2], 0, "stxr reports success");
        let mut now = [0u8; 8];
        machine.memory.read(0x40000800, &mut now).unwrap();
        assert_eq!(u64::from_le_bytes(now), 7);
    }

    #[test]
    fn store_exclusive_fails_after_another_cpu_writes_the_location() {
        let mut machine = machine();
        machine.cpu.x[0] = 0x40000800;
        machine.cpu.x[3] = 7;
        machine
            .memory
            .write(0x40000800, &5u64.to_le_bytes())
            .unwrap();
        run_to_hvc(&mut machine, &[LDXR_X1_X0, HVC]);
        // Another vCPU (or an inline store) changes the word before our stxr.
        machine
            .memory
            .write(0x40000800, &9u64.to_le_bytes())
            .unwrap();
        run_to_hvc(&mut machine, &[STXR_W2_X3_X0, HVC]);
        assert_eq!(machine.cpu.x[2], 1, "stxr reports failure");
        let mut now = [0u8; 8];
        machine.memory.read(0x40000800, &mut now).unwrap();
        assert_eq!(u64::from_le_bytes(now), 9, "a failed stxr must not store");
        // The reservation is consumed: a second stxr cannot succeed either.
        run_to_hvc(&mut machine, &[STXR_W2_X3_X0, HVC]);
        assert_eq!(machine.cpu.x[2], 1);
    }
}
