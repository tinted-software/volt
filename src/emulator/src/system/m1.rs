//! Apple M1 (t8103) SoC: the devices the Asahi Linux kernel's
//! `arch/arm64/boot/dts/apple/t8103.dtsi` describes, far enough to boot it.
//!
//! The addresses are the ones the M1 Mac mini's own device tree (`DeviceTree.j274ap`
//! in the macOS restore image) gives: arm-io's `ranges` places the SoC at
//! `0x2_0000_0000`, and the Linux nodes use the same absolute addresses. DRAM starts
//! at `0x8_0000_0000`. The kernel is booted with the device tree the caller supplies,
//! patched to describe the RAM this run provides.
use super::boot::{self, Error, Layout};
use super::dtb;
use crate::devices::aic::{self, Aic};
use crate::devices::pmgr::Pmgr;
use crate::devices::s5l_uart::{self, S5lUart};
use crate::memory::GuestMemory;

/// DRAM base (`memory@800000000` in the Asahi device tree).
pub const RAM_BASE: u64 = 0x8_0000_0000;
/// Default guest RAM: enough for the Linux kernel's own memory, well below the host's.
pub const DEFAULT_RAM_SIZE: usize = 1 << 30;
/// Interrupt controller (`aic: interrupt-controller@23b100000`).
pub const AIC_BASE: u64 = 0x2_3b10_0000;
/// Power-state register blocks (`power-management@…` with the `apple,pmgr` syscon):
/// the always-on block and the NUB block, as the device tree gives them.
pub const PMGR_BLOCKS: [(u64, u64); 2] = [(0x2_3b70_0000, 0x1_4000), (0x2_3d28_0000, 0x4000)];
/// Console (`serial0: serial@235200000`), the `stdout-path` of the M1 device trees.
pub const UART_BASE: u64 = 0x2_3520_0000;
/// Hardware IRQ of the console (`AIC_IRQ 605` on `serial0`).
pub const UART_IRQ: u32 = 605;
/// Physical space below DRAM, where the device tree (an identity-mapped `simple-bus`)
/// places the SoC's devices. Devices this model does not implement live here.
pub const UNMODELED_BASE: u64 = 0;
pub const UNMODELED_SIZE: u64 = RAM_BASE;
/// `earlycon` without an address takes the device from `stdout-path`.
pub const DEFAULT_CMDLINE: &str = "earlycon nokaslr loglevel=8";

/// `MIDR_EL1` of an Icestorm core: implementer Apple (0x61), architecture from ID
/// registers (0xf), part 0x022.
pub const MIDR_EL1: u64 = 0x610f_0220;
/// `ID_AA64PFR0_EL1`: EL0 and EL1 in AArch64, no EL2 or EL3, and floating point and
/// Advanced SIMD implemented, as on the M1.
pub const ID_AA64PFR0_EL1: u64 = 0x11;
/// `ID_AA64MMFR0_EL1`: 4 KiB and 16 KiB translation granules (the Asahi kernel is built
/// for 16 KiB pages and refuses a CPU without one), no 64 KiB granule, 40-bit PA.
pub const ID_AA64MMFR0_EL1: u64 = (0xf << 24) | (1 << 20) | 2;
/// `ID_AA64ISAR1_EL1`: LRCPC, and the architected QARMA5 for address (APA) and generic
/// (GPA) pointer authentication. No implementation-defined algorithm (API, GPI).
pub const ID_AA64ISAR1_EL1: u64 = (1 << 20) | (1 << 24) | (1 << 4);

/// The machine runs one vCPU. The device tree lists eight cores, but their
/// `cpu-release-addr` is zero, so Linux's spin-table code does not start them.
const CPUS: u32 = 1;

/// Give `system` the identity of an M1 core.
pub fn identify(system: &mut crate::aarch64::cpu::System) {
    system.midr_el1 = MIDR_EL1;
    system.id_aa64pfr0_el1 = ID_AA64PFR0_EL1;
    system.id_aa64mmfr0_el1 = ID_AA64MMFR0_EL1;
    system.id_aa64isar1_el1 = ID_AA64ISAR1_EL1;
}

/// The SoC devices on the bus: the AIC, the power-state blocks, the console, and the
/// line between the console and the AIC.
pub struct Soc {
    aic: Aic,
    pmgr: [Pmgr; PMGR_BLOCKS.len()],
    uart: S5lUart,
    unmodeled: u64,
}

impl Soc {
    /// `sink` receives every byte the kernel writes to the console.
    pub fn new(sink: impl FnMut(u8) + Send + 'static) -> Self {
        Self {
            aic: Aic::new(CPUS),
            pmgr: PMGR_BLOCKS.map(|(_, len)| Pmgr::new(len)),
            uart: S5lUart::new(sink),
            unmodeled: 0,
        }
    }

    /// Present the current device lines to the interrupt controller. Call before each
    /// run of the CPU.
    pub fn sync_lines(&mut self) {
        self.aic.set_irq(UART_IRQ, self.uart.irq_asserted());
    }

    /// Whether the IRQ pin of CPU `cpu` is asserted.
    pub fn irq_pending(&self, cpu: u32) -> bool {
        self.aic.irq_pending(cpu)
    }

    /// Accesses to arm-io devices this model does not implement: reads return zero and
    /// writes are dropped, so a probe of such a device cannot fault the kernel.
    pub fn unmodeled_accesses(&self) -> u64 {
        self.unmodeled
    }

    /// The power-state block and offset an access of `size` bytes at `address` falls in.
    fn pmgr_window(address: u64, size: u8) -> Option<(usize, u64)> {
        PMGR_BLOCKS
            .iter()
            .enumerate()
            .find_map(|(block, &(base, len))| {
                window(address, size, base, len).map(|offset| (block, offset))
            })
    }

    /// Read `size` bytes at guest-physical `address`, or `None` when no SoC device maps it.
    pub fn read(&mut self, cpu: u32, address: u64, size: u8) -> Option<u64> {
        if let Some(offset) = window(address, size, AIC_BASE, aic::LEN) {
            return Some(self.aic.read(cpu, offset, size));
        }
        if let Some((block, offset)) = Self::pmgr_window(address, size) {
            return Some(self.pmgr[block].read(offset, size));
        }
        if let Some(offset) = window(address, size, UART_BASE, s5l_uart::LEN) {
            return Some(self.uart.read(offset, size));
        }
        if window(address, size, UNMODELED_BASE, UNMODELED_SIZE).is_some() {
            self.unmodeled += 1;
            return Some(0);
        }
        None
    }

    /// Write `size` bytes to guest-physical `address`; `false` when no SoC device maps it.
    pub fn write(&mut self, cpu: u32, address: u64, size: u8, value: u64) -> bool {
        if let Some(offset) = window(address, size, AIC_BASE, aic::LEN) {
            self.aic.write(cpu, offset, size, value);
            return true;
        }
        if let Some((block, offset)) = Self::pmgr_window(address, size) {
            self.pmgr[block].write(offset, size, value);
            return true;
        }
        if let Some(offset) = window(address, size, UART_BASE, s5l_uart::LEN) {
            self.uart.write(offset, size, value);
            return true;
        }
        if window(address, size, UNMODELED_BASE, UNMODELED_SIZE).is_some() {
            self.unmodeled += 1;
            return true;
        }
        false
    }
}

/// Offset of an access of `size` bytes at `address` within `[base, base + len)`.
/// An access that straddles the end of the window matches nothing.
fn window(address: u64, size: u8, base: u64, len: u64) -> Option<u64> {
    let end = address.checked_add(u64::from(size))?;
    (address >= base && end <= base + len).then(|| address - base)
}

/// Place a Linux `Image` in RAM and the device tree after it, with the tree's
/// `/chosen/bootargs` set to `cmdline` (when non-empty) and its memory node describing
/// `ram_size` bytes at [`RAM_BASE`]. The kernel is entered with `x0` = the tree.
pub fn prepare(
    memory: &mut impl GuestMemory,
    image: &[u8],
    device_tree: &[u8],
    cmdline: &str,
    ram_size: u64,
) -> Result<Layout, Error> {
    let bootargs = (!cmdline.is_empty()).then_some(cmdline);
    let tree =
        dtb::patch(device_tree, bootargs, Some((RAM_BASE, ram_size))).map_err(Error::DeviceTree)?;
    boot::place_with_tree(memory, image, &tree, RAM_BASE, ram_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_matches_only_accesses_entirely_inside_the_device() {
        assert_eq!(window(AIC_BASE, 4, AIC_BASE, aic::LEN), Some(0));
        assert_eq!(
            window(AIC_BASE + aic::LEN - 4, 4, AIC_BASE, aic::LEN),
            Some(aic::LEN - 4)
        );
        // Straddling the end, or wrapping, matches nothing.
        assert_eq!(window(AIC_BASE + aic::LEN - 2, 4, AIC_BASE, aic::LEN), None);
        assert_eq!(window(u64::MAX - 1, 4, AIC_BASE, aic::LEN), None);
    }

    #[test]
    fn soc_routes_by_address_and_declines_everything_else() {
        let mut soc = Soc::new(|_| {});
        // The AIC identifies the reading CPU at WHOAMI.
        assert_eq!(soc.read(0, AIC_BASE + 0x2000, 4), Some(0));
        // Above DRAM, where the SoC maps nothing, is not an M1 address.
        assert_eq!(soc.read(0, RAM_BASE + (1 << 40), 4), None);
        assert!(!soc.write(0, RAM_BASE + (1 << 40), 4, 0));
        assert_eq!(soc.unmodeled_accesses(), 0);
    }

    #[test]
    fn power_state_registers_of_each_pmgr_block_acknowledge_their_target() {
        let mut soc = Soc::new(|_| {});
        for (base, _) in PMGR_BLOCKS {
            // A domain's state register, as its driver polls it.
            let state = base + 0x100;
            assert_eq!(soc.read(0, state, 4), Some(0xff));
            assert!(soc.write(0, state, 4, 0x0));
            assert_eq!(soc.read(0, state, 4), Some(0x00));
        }
        assert_eq!(soc.unmodeled_accesses(), 0);
    }

    #[test]
    fn unmodeled_arm_io_device_reads_zero_and_drops_writes_and_is_counted() {
        let mut soc = Soc::new(|_| {});
        // The pin controller (`pinctrl@23e820000`) is probed at boot.
        assert_eq!(soc.read(0, 0x2_3e82_0000, 4), Some(0));
        assert!(soc.write(0, 0x2_3e82_0000, 4, 0xffff_ffff));
        assert_eq!(soc.read(0, 0x2_3e82_0000, 4), Some(0));
        assert_eq!(soc.unmodeled_accesses(), 3);
    }
}
