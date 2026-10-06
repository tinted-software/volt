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

pub struct Machine<M> {
    pub memory: M,
    pub cache: Cache,
    pub cpu: Cpu,
    pub tlb: translate::Tlb,
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
            memory,
            cache: Cache::new(),
            cpu: Cpu::default(),
            tlb: translate::Tlb::default(),
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
            self.trace[self.blocks_run as usize % self.trace.len()] = TraceEntry {
                pc: self.cpu.pc,
                sp: self.cpu.sp,
            };
            let tlb = &mut self.tlb;
            let memory = &mut self.memory;
            let mut fetch_fault = None;
            let result = self.cache.run_block(&mut self.cpu, |cpu, address, bytes| {
                let physical = match translate::translate(
                    tlb,
                    cpu,
                    memory,
                    address,
                    translate::Access::Execute,
                ) {
                    Ok(physical) => physical,
                    Err(error) => {
                        fetch_fault = Some((
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
                if let Err(error) = memory.read(physical, bytes) {
                    fetch_fault = Some((address, false));
                    return Err(crate::aarch64::compile::Error::Memory(error));
                }
                Ok(bytes.len())
            });
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
                    let exclusive = self.cpu.exclusive;
                    self.cpu.exclusive = Exclusive::None;
                    let held = self.cpu.monitor_valid
                        && self.cpu.monitor_address == self.cpu.address
                        && self.cpu.monitor_width == self.cpu.width;
                    if exclusive == Exclusive::Load {
                        self.cpu.monitor_valid = true;
                        self.cpu.monitor_address = self.cpu.address;
                        self.cpu.monitor_width = self.cpu.width;
                    }
                    if exclusive == Exclusive::Store {
                        self.cpu.monitor_valid = false;
                    }
                    if exclusive != Exclusive::Store || held {
                        self.access(
                            self.cpu.address,
                            self.cpu.width,
                            self.cpu.dest,
                            self.cpu.value,
                            serial,
                            gic,
                        )?;
                    }
                    if exclusive == Exclusive::Store && !self.faulted && self.cpu.status_dest != 31
                    {
                        self.cpu.x[self.cpu.status_dest as usize] = u64::from(!held);
                    }
                    if self.faulted {
                        self.cpu.second_pending = false;
                        self.cpu.writeback = false;
                        continue;
                    }
                    if self.cpu.second_pending {
                        self.cpu.second_pending = false;
                        self.cpu.load_signed = self.cpu.second_signed;
                        self.cpu.second_signed = SignExtend::None;
                        self.access(
                            self.cpu.second_address,
                            self.cpu.second_width,
                            self.cpu.second_dest,
                            self.cpu.second_value,
                            serial,
                            gic,
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
    fn access(
        &mut self,
        address: u64,
        width: u8,
        dest: u8,
        value: u64,
        serial: &mut Pl011,
        gic: &mut Gicv2,
    ) -> Result<(), Error> {
        if !matches!(width, 1 | 2 | 4 | 8) {
            return Err(Error::InvalidAccessWidth(width));
        }
        let load = self.cpu.trap == Trap::Load;
        let signed = self.cpu.load_signed;
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
                        Some(serial.read(physical - fdt::UART_BASE, width))
                    } else {
                        serial.write(physical - fdt::UART_BASE, width, value);
                        None
                    }
                } else if window(fdt::GICD_BASE, crate::devices::gicv2::DISTRIBUTOR_SIZE) {
                    if load {
                        Some(gic.read_distributor(physical - fdt::GICD_BASE, width))
                    } else {
                        gic.write_distributor(physical - fdt::GICD_BASE, width, value);
                        None
                    }
                } else if window(fdt::GIC_CPU_BASE, crate::devices::gicv2::CPU_SIZE) {
                    if load {
                        Some(gic.read_cpu(physical - fdt::GIC_CPU_BASE, width))
                    } else {
                        gic.write_cpu(physical - fdt::GIC_CPU_BASE, width, value);
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
            .access(0x40000000, 1, 0, 0, &mut serial, &mut gic)
            .unwrap();
        assert_eq!(machine.cpu.x[0], 0xffff_ffff_ffff_ff80);
        machine.cpu.load_signed = SignExtend::To32;
        machine
            .access(0x40000000, 1, 1, 0, &mut serial, &mut gic)
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
            .access(0x0b000000, 4, 31, 123, &mut serial, &mut gic)
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
}
