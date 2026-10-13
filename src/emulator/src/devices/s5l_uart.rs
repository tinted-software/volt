//! Apple S5L UART (`apple,s5l-uart`), the console port of M1-class SoCs.
//!
//! The Apple variant differs from the classic Samsung register map in one way
//! that matters here: it has no `UINTP`/`UINTM` interrupt registers. Its
//! interrupt causes are the flag bits of `UTRSTAT` itself. A cause is cleared
//! by writing a one to its bit, and a `UCON` enable bit decides whether a
//! pending cause reaches the line. `apple_serial_handle_irq` in the driver
//! works this way and never touches `UINTP`.
//!
//! Transmit completes at once. Every byte written to `UTXH` reaches the sink,
//! and the transmit FIFO never reports full. The one transmit cause is
//! `TXTHRESH` (`UTRSTAT` bit 5). Real hardware latches it when the transmit
//! FIFO drops below its threshold, and in this model every transmit does that,
//! so the cause is raised after each byte. The line rises only while
//! `UCON.TXTHRESH_ENA` is set, which is when the driver wants a TX interrupt.
//!
//! The driver's TX path never waits on that interrupt. `enable_tx_pio` sets the
//! enable bit and calls the TX routine itself, because the Apple UART only
//! raises edge-triggered TX interrupts (see the comment there). That routine
//! drains the queue synchronously, since the FIFO never fills, and then clears
//! the enable bit through `stop_tx`. If an interrupt does arrive while work is
//! left, the handler clears `TXTHRESH` first and then writes more bytes, which
//! raises the cause again, so each entry drains some of the queue.
//!
//! Receive has no input in this model: the receive FIFO is empty, `RXDR` stays
//! clear, and no receive cause is ever raised.

use alloc::boxed::Box;
use alloc::collections::VecDeque;

/// Mapping length of the UART register window.
pub const LEN: u64 = 0x1000;

const ULCON: u64 = 0x00;
const UCON: u64 = 0x04;
const UFCON: u64 = 0x08;
const UMCON: u64 = 0x0c;
const UTRSTAT: u64 = 0x10;
const UFSTAT: u64 = 0x18;
const UTXH: u64 = 0x20;
const URXH: u64 = 0x24;
const UBRDIV: u64 = 0x28;
const UFRACVAL: u64 = 0x2c;

/// `UCON` bit that enables the transmit interrupt line, `APPLE_S5L_UCON_TXTHRESH_ENA`.
const UCON_TXTHRESH_ENA: u32 = 1 << 13;

/// `UTRSTAT` bit for the transmit-threshold cause, `APPLE_S5L_UTRSTAT_TXTHRESH`.
const UTRSTAT_TXTHRESH: u32 = 1 << 5;
/// Transmitter empty and transmit FIFO empty. Always set: transmit is instant.
const UTRSTAT_TXE: u32 = 1 << 2;
const UTRSTAT_TXFE: u32 = 1 << 1;
/// Receive data ready, set only while the receive FIFO holds data.
const UTRSTAT_RXDR: u32 = 1 << 0;

/// `UFCON` bits that reset a FIFO. The hardware clears them itself, so they
/// never read back as set.
const UFCON_RESETRX: u32 = 1 << 1;
const UFCON_RESETTX: u32 = 1 << 2;

/// `UFSTAT` receive count field and its full flag.
const UFSTAT_RXMASK: u32 = 0xf;
const UFSTAT_RXFULL: u32 = 1 << 8;
/// Receive FIFO depth reported in `UFSTAT`, `fifosize` for this port.
const FIFO_DEPTH: usize = 16;

/// An Apple S5L UART.
///
/// The sink takes each transmitted byte in order. The interrupt line is read
/// through [`S5lUart::irq_asserted`].
pub struct S5lUart {
    sink: Box<dyn FnMut(u8) + Send>,
    ulcon: u32,
    ucon: u32,
    ufcon: u32,
    umcon: u32,
    ubrdiv: u32,
    ufracval: u32,
    /// Pending interrupt causes, in `UTRSTAT` bit positions.
    pending: u32,
    rx: VecDeque<u8>,
}

impl S5lUart {
    /// A UART writing transmitted bytes to `sink`.
    pub fn new(sink: impl FnMut(u8) + Send + 'static) -> Self {
        Self {
            sink: Box::new(sink),
            ulcon: 0,
            ucon: 0,
            ufcon: 0,
            umcon: 0,
            ubrdiv: 0,
            ufracval: 0,
            pending: 0,
            rx: VecDeque::new(),
        }
    }

    /// Read a register. `offset` is relative to the device base; `size` is in
    /// bytes. Unknown offsets read as zero.
    pub fn read(&mut self, offset: u64, size: u8) -> u64 {
        let value = match offset {
            ULCON => self.ulcon,
            UCON => self.ucon,
            UFCON => self.ufcon,
            UMCON => self.umcon,
            UTRSTAT => self.utrstat(),
            UFSTAT => self.ufstat(),
            URXH => self.rx.pop_front().map_or(0, u32::from),
            UBRDIV => self.ubrdiv,
            UFRACVAL => self.ufracval,
            _ => 0,
        };
        u64::from(value) & lane_mask(size)
    }

    /// Write a register. Only the low `size` bytes of `value` are used.
    /// Unknown offsets and read-only registers ignore the write.
    pub fn write(&mut self, offset: u64, size: u8, value: u64) {
        let value = (value & lane_mask(size)) as u32;
        match offset {
            ULCON => self.ulcon = value,
            UCON => self.ucon = value,
            UFCON => {
                if value & UFCON_RESETRX != 0 {
                    self.rx.clear();
                }
                self.ufcon = value & !(UFCON_RESETRX | UFCON_RESETTX);
            }
            UMCON => self.umcon = value,
            // Write-one-to-clear. Bits that are not causes have nothing pending
            // to clear, so they fall through harmlessly.
            UTRSTAT => self.pending &= !value,
            UTXH => self.transmit(value as u8),
            UBRDIV => self.ubrdiv = value,
            UFRACVAL => self.ufracval = value,
            _ => {}
        }
    }

    /// Level of the interrupt line to the AIC: asserted while the transmit
    /// cause is pending and its `UCON` enable bit is set.
    pub fn irq_asserted(&self) -> bool {
        self.pending & UTRSTAT_TXTHRESH != 0 && self.ucon & UCON_TXTHRESH_ENA != 0
    }

    fn transmit(&mut self, byte: u8) {
        (self.sink)(byte);
        self.pending |= UTRSTAT_TXTHRESH;
    }

    fn utrstat(&self) -> u32 {
        let rxdr = if self.rx.is_empty() { 0 } else { UTRSTAT_RXDR };
        self.pending | UTRSTAT_TXE | UTRSTAT_TXFE | rxdr
    }

    /// The transmit FIFO never holds anything, so only the receive count shows.
    fn ufstat(&self) -> u32 {
        let len = self.rx.len();
        let full = if len >= FIFO_DEPTH { UFSTAT_RXFULL } else { 0 };
        full | (len as u32 & UFSTAT_RXMASK)
    }

    /// Feed one byte into the receive FIFO, as input from a host would.
    #[cfg(test)]
    fn feed_rx(&mut self, byte: u8) {
        self.rx.push_back(byte);
    }
}

/// Mask for the low `size` bytes of a register access.
fn lane_mask(size: u8) -> u64 {
    match size {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        S5lUart, UBRDIV, UCON, UCON_TXTHRESH_ENA, UFCON, UFRACVAL, UFSTAT, ULCON, UMCON, URXH,
        UTRSTAT, UTRSTAT_TXTHRESH, UTXH,
    };
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    const CAPACITY: usize = 512;
    const ZERO: AtomicU8 = AtomicU8::new(0);

    /// Bytes the sink received, shared with the test through atomics so the
    /// sink stays `Send` without `std`.
    struct Capture {
        len: AtomicUsize,
        bytes: [AtomicU8; CAPACITY],
    }

    impl Capture {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                len: AtomicUsize::new(0),
                bytes: [ZERO; CAPACITY],
            })
        }

        fn push(&self, byte: u8) {
            let index = self.len.fetch_add(1, Ordering::SeqCst);
            self.bytes[index].store(byte, Ordering::SeqCst);
        }

        fn collected(&self) -> alloc::vec::Vec<u8> {
            (0..self.len.load(Ordering::SeqCst))
                .map(|i| self.bytes[i].load(Ordering::SeqCst))
                .collect()
        }
    }

    fn uart_with_capture() -> (S5lUart, Arc<Capture>) {
        let capture = Capture::new();
        let sink = Arc::clone(&capture);
        (S5lUart::new(move |byte| sink.push(byte)), capture)
    }

    /// The `UFCON` value the driver's startup writes: FIFO mode, RX trigger 8,
    /// and the RX reset bit it ORs in (plus TX reset when not a console).
    const UFCON_STARTUP: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4);

    #[test]
    fn transmitted_bytes_reach_sink_in_order() {
        let (mut uart, capture) = uart_with_capture();
        for &byte in b"volt\n" {
            uart.write(UTXH, 4, u64::from(byte));
        }
        // Only the low byte of the register is the character.
        uart.write(UTXH, 4, 0x1_0000_0041);
        uart.write(UTXH, 1, 0x142);
        assert_eq!(capture.collected(), b"volt\nAB");
    }

    #[test]
    fn transmit_fifo_never_reports_full_or_nonempty() {
        let (mut uart, _capture) = uart_with_capture();
        for i in 0..200u64 {
            uart.write(UTXH, 4, i);
            let ufstat = uart.read(UFSTAT, 4);
            assert_eq!(ufstat & (1 << 9), 0, "TXFULL set after byte {i}");
            assert_eq!(ufstat & (0xf << 4), 0, "TX count nonzero after byte {i}");
        }
    }

    #[test]
    fn early_console_sees_room_and_empty_transmitter() {
        // `samsung_early_putc` polls TXFULL when UFCON has FIFO mode set, and
        // TXFE in UTRSTAT otherwise. The driver's startup sets FIFO mode.
        let (mut uart, _capture) = uart_with_capture();
        uart.write(UFCON, 4, u64::from(UFCON_STARTUP));
        uart.write(UTXH, 4, u64::from(b'x'));
        assert_eq!(uart.read(UFSTAT, 4) & (1 << 9), 0);
        assert_ne!(uart.read(UTRSTAT, 4) & (1 << 1), 0);
        // Without FIFO mode the early console polls TXFE only; it must also pass.
        uart.write(UFCON, 4, 0);
        assert_ne!(uart.read(UTRSTAT, 4) & (1 << 1), 0);
    }

    #[test]
    fn write_one_to_clear_drops_the_line() {
        let (mut uart, _capture) = uart_with_capture();
        uart.write(UCON, 4, u64::from(UCON_TXTHRESH_ENA));
        uart.write(UTXH, 4, u64::from(b'a'));
        assert!(uart.irq_asserted());
        assert_ne!(uart.read(UTRSTAT, 4) & u64::from(UTRSTAT_TXTHRESH), 0);

        uart.write(UTRSTAT, 4, u64::from(UTRSTAT_TXTHRESH));
        assert!(!uart.irq_asserted());
        assert_eq!(uart.read(UTRSTAT, 4) & u64::from(UTRSTAT_TXTHRESH), 0);
    }

    #[test]
    fn enable_bit_masks_the_line_but_not_the_cause() {
        let (mut uart, _capture) = uart_with_capture();
        uart.write(UTXH, 4, u64::from(b'a'));
        assert!(!uart.irq_asserted(), "no enable bit, no line");
        assert_ne!(uart.read(UTRSTAT, 4) & u64::from(UTRSTAT_TXTHRESH), 0);

        uart.write(UCON, 4, u64::from(UCON_TXTHRESH_ENA));
        assert!(uart.irq_asserted(), "enabling reveals the pending cause");
    }

    #[test]
    fn driver_tx_kick_completes_without_waiting_for_an_interrupt() {
        // `enable_tx_pio`: set the enable bit, write the queued bytes inline,
        // then `stop_tx` clears the enable bit. The line must end released.
        let (mut uart, capture) = uart_with_capture();
        uart.write(UCON, 4, u64::from(UCON_TXTHRESH_ENA));
        for &byte in b"boot" {
            uart.write(UTXH, 4, u64::from(byte));
        }
        uart.write(UCON, 4, 0);
        assert!(!uart.irq_asserted());
        assert_eq!(capture.collected(), b"boot");

        // A later kick with a cause left over from before still makes progress:
        // the handler clears the cause, then drains and disables the line.
        uart.write(UCON, 4, u64::from(UCON_TXTHRESH_ENA));
        assert!(uart.irq_asserted());
        uart.write(UTRSTAT, 4, u64::from(UTRSTAT_TXTHRESH));
        uart.write(UTXH, 4, u64::from(b'!'));
        uart.write(UCON, 4, 0);
        assert!(!uart.irq_asserted());
        assert_eq!(capture.collected(), b"boot!");
    }

    #[test]
    fn startup_registers_read_back() {
        let (mut uart, _capture) = uart_with_capture();
        let ucon = 0x3a85_u32; // APPLE_S5L_UCON_DEFAULT with all four enable bits set
        uart.write(ULCON, 4, 0x3);
        uart.write(UCON, 4, u64::from(ucon));
        uart.write(UFCON, 4, u64::from(UFCON_STARTUP));
        uart.write(UMCON, 4, 0x10);
        uart.write(UBRDIV, 4, 0x1a0);
        uart.write(UFRACVAL, 4, 0x7);

        assert_eq!(uart.read(ULCON, 4), 0x3);
        assert_eq!(uart.read(UCON, 4), u64::from(ucon));
        // The FIFO reset bits are self-clearing and do not read back.
        assert_eq!(uart.read(UFCON, 4), u64::from(UFCON_STARTUP & !0b110));
        assert_eq!(uart.read(UMCON, 4), 0x10);
        assert_eq!(uart.read(UBRDIV, 4), 0x1a0);
        assert_eq!(uart.read(UFRACVAL, 4), 0x7);
    }

    #[test]
    fn unknown_offsets_read_zero_and_ignore_writes() {
        let (mut uart, _capture) = uart_with_capture();
        uart.write(0x30, 4, 0xffff_ffff);
        uart.write(0x38, 4, 0xffff_ffff);
        assert_eq!(uart.read(0x30, 4), 0);
        assert_eq!(uart.read(0x38, 4), 0);
        assert!(!uart.irq_asserted());
    }

    #[test]
    fn receive_fifo_drives_data_ready_and_urxh() {
        let (mut uart, _capture) = uart_with_capture();
        assert_eq!(uart.read(URXH, 4), 0);
        assert_eq!(uart.read(UTRSTAT, 4) & 1, 0);

        uart.feed_rx(b'k');
        assert_ne!(uart.read(UTRSTAT, 4) & 1, 0);
        assert_eq!(uart.read(UFSTAT, 4) & 0xf, 1);
        assert_eq!(uart.read(URXH, 4), u64::from(b'k'));
        assert_eq!(uart.read(UTRSTAT, 4) & 1, 0);
    }
}
