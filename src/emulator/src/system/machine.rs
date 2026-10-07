//! One-vCPU machine over native translated blocks (Mirage aarch64/Machine.zig).
use super::fdt;
use crate::aarch64::cpu::{Exclusive, Trap};
use crate::aarch64::decode::SignExtend;
use crate::aarch64::{Cache, Cpu};
use crate::aarch64::{exception, translate};
use crate::devices::{bus::Device, gicv2::Gicv2, pl011::Pl011};
use crate::memory::{GuestMemory, MemoryError};

#[derive(Debug)]
pub enum Error {
    Compile(crate::aarch64::compile::Error),
    Memory(MemoryError),
    InvalidAccessWidth(u8),
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
    fn read_virtio(&mut self, offset: u64, size: u8) -> u64 {
        let _ = (offset, size);
        0
    }
    fn write_virtio(&mut self, offset: u64, size: u8, value: u64) {
        let _ = (offset, size, value);
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
    timer_fired: bool,
    pub stalled_at: u64,
    pub stalled: Option<String>,
    pub unmapped: u64,
    trace: [TraceEntry; 512],
    pub blocks_run: u64,
    faulted: bool,
    device_access: bool,
}
impl<M: GuestMemory> Machine<M> {
    pub fn new(memory: M) -> Self {
        Self {
            cpu_id: 0,
            memory,
            // `VOLT_INLINE_MEMORY=0` turns the inline data-TLB fast path off, for
            // A/B comparisons and for bisecting suspected emulation bugs.
            cache: if std::env::var_os("VOLT_INLINE_MEMORY").is_some_and(|v| v == "0") {
                Cache::new()
            } else {
                Cache::with_inline_memory()
            },
            cpu: Cpu::default(),
            tlb: translate::Tlb::default(),
            dtlb_generation: 0,
            watch_epoch: 0,
            irq_line: false,
            timer_fired: false,
            stalled_at: 0,
            stalled: None,
            unmapped: 0,
            trace: [TraceEntry::default(); 512],
            blocks_run: 0,
            faulted: false,
            device_access: false,
        }
    }
    /// A vCPU whose translated blocks are shared with other vCPUs.
    pub fn with_shared_blocks(
        memory: M,
        cpu_id: u32,
        blocks: std::sync::Arc<crate::aarch64::cache::SharedBlocks>,
    ) -> Self {
        Self {
            cache: Cache::with_shared(blocks),
            ..Self::with_cpu_id(memory, cpu_id)
        }
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
    pub fn run(&mut self, serial: &mut Pl011, gic: &mut Gicv2) -> Result<Exit, Error> {
        self.run_with_devices(&mut (serial, gic))
    }
    pub fn run_with_devices(&mut self, devices: &mut impl DeviceIo) -> Result<Exit, Error> {
        self.stalled = None;
        for _ in 0..64 {
            if self.timer_edge() {
                return Ok(Exit::Timer);
            }
            if self.irq_line && self.cpu.system.daif & 2 == 0 {
                let pc = self.cpu.pc;
                exception::take(
                    &mut self.cpu,
                    exception::Reason {
                        kind: exception::Kind::Irq,
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
                Trap::Svc | Trap::Brk => {
                    let ec = if self.cpu.trap == Trap::Svc { 0x15 } else { 0 };
                    self.cpu.trap = Trap::None;
                    let pc = self.cpu.pc;
                    exception::take(
                        &mut self.cpu,
                        exception::Reason {
                            ec,
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
                        },
                        pc,
                    );
                }
                Trap::Sync => {
                    self.tlb.flush();
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
        let ec = if instruction {
            if self.cpu.system.el == 1 { 0x21 } else { 0x20 }
        } else if self.cpu.system.el == 1 {
            0x25
        } else {
            0x24
        };
        let status = if permission {
            if instruction { 13 } else { 12 }
        } else {
            4
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
    fn access(
        &mut self,
        address: u64,
        width: u8,
        dest: u8,
        value: u64,
        devices: &mut impl DeviceIo,
    ) -> Result<(), Error> {
        if !matches!(width, 1 | 2 | 4 | 8) {
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
        let mut bytes = value.to_le_bytes();
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
                self.abort(address, false, false, !load);
                return Ok(());
            }
            result
        };
        match result {
            Ok(()) => {}
            Err(MemoryError::AccessFault { .. }) => {
                self.device_access = true;
                let end = physical.checked_add(width as u64);
                let window =
                    |base, length| physical >= base && end.is_some_and(|end| end <= base + length);
                let read = if window(fdt::UART_BASE, crate::devices::pl011::LEN) {
                    if load {
                        Some(devices.read_uart(physical - fdt::UART_BASE, width))
                    } else {
                        devices.write_uart(physical - fdt::UART_BASE, width, value);
                        None
                    }
                } else if window(fdt::GICD_BASE, crate::devices::gicv2::DISTRIBUTOR_SIZE) {
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
                } else if window(fdt::VIRTIO_MMIO_BASE, fdt::VIRTIO_MMIO_SIZE) {
                    if load {
                        Some(devices.read_virtio(physical - fdt::VIRTIO_MMIO_BASE, width))
                    } else {
                        devices.write_virtio(physical - fdt::VIRTIO_MMIO_BASE, width, value);
                        None
                    }
                } else {
                    self.unmapped += 1;
                    self.abort(address, false, false, !load);
                    return Ok(());
                };
                if let Some(value) = read {
                    bytes = value.to_le_bytes();
                }
            }
            Err(error) => return Err(Error::Memory(error)),
        }
        if first == width as usize && !self.device_access && self.cache.inline_memory() {
            self.fill_dtlb(address, physical, !load);
        }
        if load && dest != 31 {
            let value = u64::from_le_bytes(bytes);
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
    const LDXR_X1_X0: u32 = 0xc85f7c01;
    const STXR_W2_X3_X0: u32 = 0xc8027c03;
    const HVC: u32 = 0xd4000002;

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
