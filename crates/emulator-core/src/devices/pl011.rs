//! A PL011 serial port, enough of one for a guest to print.
//!
//! This is the device an aarch64 Linux guest reaches with `earlycon=pl011`, so
//! it is the first thing a guest says anything through. Only two registers
//! matter for output: the data register the guest writes a byte to, and the
//! flag register it polls first.

use alloc::boxed::Box;

use super::bus::Device;

/// Mapping length of the PL011 register window.
pub const LEN: u64 = 0x1000;

const DR: u64 = 0x000;
const FR: u64 = 0x018;
const IBRD: u64 = 0x024;
const FBRD: u64 = 0x028;
const LCR_H: u64 = 0x02c;
const CR: u64 = 0x030;
const IFLS: u64 = 0x034;
const IMSC: u64 = 0x038;
const RIS: u64 = 0x03c;
const MIS: u64 = 0x040;
/// Writing here says the driver has dealt with an interrupt.
const ICR: u64 = 0x044;
const PERIPH_ID0: u64 = 0xfe0;
const CELL_ID3: u64 = 0xffc;

/// `UARTPeriphID0` through `UARTPCellID3`, from the PL011 manual. The bus reads
/// the four peripheral bytes as one word and matches it against the driver, and
/// the four cell bytes are the constant every PrimeCell carries.
const IDENTIFICATION: [u32; 8] = [0x11, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

/// `TXFE`, the transmitter holds nothing, and `RXFE`, nothing has arrived. A
/// guest polls the flag register before it writes, so a transmitter that never
/// reports itself empty leaves the guest spinning in its own loop forever.
const FR_TXFE: u64 = 1 << 7;
const FR_RXFE: u64 = 1 << 4;

/// The transmit interrupt. This port never fills, so there is always room and
/// the raw status always has this bit set. What decides whether the guest sees
/// it is whether the driver asked.
const INTERRUPT_TX: u32 = 1 << 5;

/// Registers this port keeps but does not act on: the baud divisors, the line
/// control and the control register itself. A driver reads one, changes a bit
/// and writes it back, so a port that answers zero hands back a value the
/// driver never chose and undoes its own configuration.
const KEPT: [u64; 5] = [IBRD, FBRD, LCR_H, CR, IFLS];

/// A PL011 serial port.
///
/// The sink takes each transmitted byte; the optional interrupt line is raised
/// (`true`) or released (`false`) as the masked transmit status changes. A
/// port with nowhere to report is still usable by a driver that polls, which
/// is what an early console does.
pub struct Pl011 {
    sink: Box<dyn FnMut(u8) + Send>,
    line: Option<Box<dyn FnMut(bool) + Send>>,
    imsc: u32,
    /// What was written to each of [`KEPT`], in the same order.
    held: [u32; KEPT.len()],
}

impl Pl011 {
    /// A port writing bytes to `sink`, with no interrupt line (polling only).
    pub fn new(sink: impl FnMut(u8) + Send + 'static) -> Self {
        Self {
            sink: Box::new(sink),
            line: None,
            imsc: 0,
            held: [0; KEPT.len()],
        }
    }

    /// Attach an interrupt line reporting raise/release of the masked status.
    pub fn with_line(mut self, line: impl FnMut(bool) + Send + 'static) -> Self {
        self.line = Some(Box::new(line));
        self
    }
    /// The raw status. Room to transmit is the only thing ever reported.
    fn status(&self) -> u32 {
        INTERRUPT_TX
    }

    /// Tell the line whether this port is asking for attention. The interrupt
    /// is a level, so it has to be released as well as raised.
    fn refresh(&mut self) {
        let level = self.status() & self.imsc != 0;
        if let Some(line) = self.line.as_mut() {
            line(level);
        }
    }
}

impl Device for Pl011 {
    fn read(&mut self, offset: u64, _size: u8) -> u64 {
        if (PERIPH_ID0..=CELL_ID3).contains(&offset) {
            let index = ((offset - PERIPH_ID0) / 4) as usize;
            return IDENTIFICATION[index] as u64;
        }
        for (each, held) in KEPT.iter().zip(self.held.iter()) {
            if offset == *each {
                return *held as u64;
            }
        }
        match offset {
            FR => FR_TXFE | FR_RXFE,
            IMSC => self.imsc as u64,
            RIS => self.status() as u64,
            MIS => (self.status() & self.imsc) as u64,
            _ => 0,
        }
    }

    fn write(&mut self, offset: u64, _size: u8, value: u64) {
        for (each, held) in KEPT.iter().zip(self.held.iter_mut()) {
            if offset == *each {
                *held = value as u32;
                return;
            }
        }
        match offset {
            // The sink is total by contract: device code runs inside the
            // guest's run loop, where there is nothing sensible to do with a
            // host output error, so a fallible destination must handle its own
            // errors (count, buffer, or ignore) behind the closure.
            DR => (self.sink)(value as u8),
            IMSC => {
                self.imsc = value as u32;
                self.refresh();
            }
            // There is nothing to clear: the only thing reported is room to
            // transmit, and writing a byte does not use it up. The line is
            // refreshed anyway, so a driver that masks through this register
            // is still obeyed.
            ICR => self.refresh(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::bus::{Bus, BusDevice};
    use super::{LEN, Pl011};
    use alloc::vec::Vec;

    const BASE: u64 = 0x0900_0000;

    #[derive(Clone, Copy)]
    struct SendPtr(*mut Vec<u8>);
    unsafe impl Send for SendPtr {}
    impl SendPtr {
        unsafe fn push(&self, byte: u8) {
            unsafe { (*self.0).push(byte) }
        }
    }

    fn bus_with_sink(buffer: &mut Vec<u8>) -> (Bus, *mut Vec<u8>) {
        // The port owns the sink closure; hand the test a raw pointer to the
        // buffer it pushes into. Single-threaded tests only.
        let ptr = SendPtr(buffer);
        let raw = ptr.0;
        let uart = Pl011::new(move |byte| unsafe {
            ptr.push(byte);
        });
        let mut bus = Bus::new();
        bus.attach(BusDevice::new(BASE, LEN, uart));
        (bus, raw)
    }

    #[test]
    fn the_data_register_writes_its_byte_to_the_sink() {
        let mut buffer = Vec::new();
        let (mut bus, _) = bus_with_sink(&mut buffer);
        bus.write(BASE, 1, b'h' as u64);
        bus.write(BASE, 1, b'i' as u64);
        assert_eq!(buffer, b"hi");
    }

    #[test]
    fn the_flag_register_reports_transmitter_empty_and_nothing_received() {
        let mut buffer = Vec::new();
        let (mut bus, _) = bus_with_sink(&mut buffer);
        // A guest polls this before it writes. Reporting the transmitter busy
        // forever hangs the guest in its own loop.
        let flags = bus.read(BASE + 0x018, 4).unwrap();
        assert_ne!(flags & (1 << 7), 0);
        assert_ne!(flags & (1 << 4), 0);
        assert_eq!(flags & (1 << 5), 0);
    }

    #[test]
    fn the_primecell_identification_registers_name_this_device() {
        let mut buffer = Vec::new();
        let (mut bus, _) = bus_with_sink(&mut buffer);
        // The bus reads four bytes and joins them into the peripheral number.
        // Masked the way the driver masks it, this has to be the PL011.
        let mut periph: u32 = 0;
        for i in 0..4 {
            periph |= (bus.read(BASE + 0xfe0 + i * 4, 4).unwrap() as u32) << (i * 8);
        }
        assert_eq!(periph & 0x000f_ffff, 0x0004_1011);
        // Every PrimeCell carries the same cell number; zero binds nothing.
        let mut cell: u32 = 0;
        for i in 0..4 {
            cell |= (bus.read(BASE + 0xff0 + i * 4, 4).unwrap() as u32) << (i * 8);
        }
        assert_eq!(cell, 0xb105_f00d);
    }

    #[test]
    fn the_masked_status_is_the_raw_status_with_the_mask_applied() {
        let mut buffer = Vec::new();
        let (mut bus, _) = bus_with_sink(&mut buffer);
        // Room to transmit is always there, so the raw status always says so.
        assert_eq!(bus.read(BASE + 0x03c, 4), Some(1 << 5));
        // Nothing is masked in yet, so the driver is told nothing.
        assert_eq!(bus.read(BASE + 0x040, 4), Some(0));
        bus.write(BASE + 0x038, 4, 1 << 5);
        assert_eq!(bus.read(BASE + 0x040, 4), Some(1 << 5));
    }

    #[test]
    fn a_port_with_nowhere_to_report_is_still_usable_by_polling() {
        let mut buffer = Vec::new();
        let (mut bus, _) = bus_with_sink(&mut buffer);
        // An early console never asks for an interrupt, and must not need one.
        bus.write(BASE + 0x038, 4, 1 << 5);
        bus.write(BASE, 1, b'p' as u64);
        assert_eq!(buffer, b"p");
    }

    #[test]
    fn a_register_the_port_keeps_reads_back_as_written() {
        let mut buffer = Vec::new();
        let (mut bus, _) = bus_with_sink(&mut buffer);
        // The driver reads the control register, sets the bits it wants and
        // writes it back. A port that answers zero undoes its configuration.
        bus.write(BASE + 0x030, 4, 0x0301);
        assert_eq!(bus.read(BASE + 0x030, 4), Some(0x0301));
        bus.write(BASE + 0x02c, 4, 0x70);
        assert_eq!(bus.read(BASE + 0x02c, 4), Some(0x70));
    }

    #[test]
    fn asking_for_the_transmit_interrupt_raises_and_releases_the_line() {
        let mut buffer = Vec::new();
        let raised = alloc::sync::Arc::new(core::sync::atomic::AtomicBool::new(false));
        let flag = raised.clone();
        let ptr = SendPtr(&mut buffer);
        let uart = Pl011::new(move |byte| unsafe {
            ptr.push(byte);
        })
        .with_line(move |level| flag.store(level, core::sync::atomic::Ordering::Release));
        let mut bus = Bus::new();
        bus.attach(BusDevice::new(BASE, LEN, uart));
        assert!(!raised.load(core::sync::atomic::Ordering::Acquire));
        bus.write(BASE + 0x038, 4, 1 << 5);
        assert!(raised.load(core::sync::atomic::Ordering::Acquire));
        bus.write(BASE + 0x038, 4, 0);
        assert!(!raised.load(core::sync::atomic::Ordering::Acquire));
    }
}
