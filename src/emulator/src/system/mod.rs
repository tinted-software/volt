//! AArch64 Linux full-system execution on the Volt native JIT.
//! Ports Mirage's JIT Machine, raw Image launch and one-vCPU serial/GIC runner.

pub mod boot;
pub mod esr;
pub mod fdt;
pub mod machine;
pub mod psci;

use crate::aarch64::{Cpu, decode, translate};
use crate::devices::{gicv2::Gicv2, pl011::Pl011};
use crate::memory::{GuestMemory, PhysicalMemory, Region};
use core::{
    cell::{Cell, RefCell},
    time::Duration,
};
use std::{io::Write, rc::Rc, time::Instant};

pub const RAM_BASE: u64 = 0x4000_0000;
pub const RAM_SIZE: usize = 128 << 20;
pub const DEFAULT_CMDLINE: &str = "console=ttyAMA0 earlycon=pl011,0x9000000 nokaslr loglevel=8";

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
    Fault(String),
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
            "entry {:x}, device tree {:x}",
            self.layout.entry, self.layout.device_tree
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

/// Boot a raw Linux Image with live serial output on stdout and a bounded wall-clock run.
/// Execution faults are returned in the report with the full guest state, not discarded.
pub fn boot(image: &[u8], timeout: Duration, cmdline: &str) -> Result<BootReport, Error> {
    let mut output = std::io::BufWriter::new(std::io::stdout());
    boot_with_serial(image, timeout, cmdline, move |byte| {
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
    mut sink: impl FnMut(u8) + 'static,
) -> Result<BootReport, Error> {
    let region = Region::ram(RAM_BASE, vec![0; RAM_SIZE]).map_err(Error::Memory)?;
    let mut memory = PhysicalMemory::new(vec![region]).map_err(Error::Memory)?;
    let layout = boot::prepare(&mut memory, image, cmdline, RAM_BASE, RAM_SIZE as u64)
        .map_err(Error::Boot)?;
    let console = Rc::new(RefCell::new(Vec::new()));
    let serial_console = console.clone();
    let uart_line = Rc::new(Cell::new(false));
    let serial_line = uart_line.clone();
    let mut serial = Pl011::new(move |byte| {
        serial_console.borrow_mut().push(byte);
        sink(byte);
    })
    .with_line(move |level| serial_line.set(level));
    let mut gic = Gicv2::default();
    let mut machine = machine::Machine::new(memory);
    machine.cpu.pc = layout.entry;
    machine.cpu.x[0] = layout.device_tree;
    machine.cpu.x[1..4].fill(0);
    let started = Instant::now();
    let mut exits = 0;
    let mut timer_interrupts = 0;
    let reason = loop {
        if started.elapsed() >= timeout {
            break StopReason::Timeout;
        }
        if uart_line.get() {
            gic.raise(fdt::UART_INTID);
        } else {
            gic.lower(fdt::UART_INTID);
        }
        machine.irq_line = gic.signalled(0);
        match machine.run(&mut serial, &mut gic) {
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
            Ok(machine::Exit::Wait | machine::Exit::Interrupted) => {}
            Err(error) => break StopReason::Fault(error.to_string()),
        }
        exits += 1;
        let sp = machine.cpu.sp;
        if sp != 0 && !(RAM_BASE..=RAM_BASE + RAM_SIZE as u64).contains(&sp) && sp >> 48 != 0xffff {
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
        console: core::mem::take(&mut *console.borrow_mut()),
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
}
