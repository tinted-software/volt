use super::cpu::Cpu;
use crate::memory::GuestMemory;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    Execute,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslateError {
    TranslationFault,
    PermissionFault,
    MalformedTables,
}
impl core::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl core::error::Error for TranslateError {}
const PAGE_MASK: u64 = 0xfff;
#[derive(Clone, Copy, Debug, Default)]
pub struct Entry {
    pub virtual_address: u64,
    pub physical: u64,
    pub generation: u64,
    pub ap: u8,
    pub executable: bool,
}
/// The translation regime for the current EL. `split` regimes (EL1&0, and EL2 with
/// `HCR_EL2.E2H`, or EL0 under TGE with E2H) walk a TTBR0 and a TTBR1 range under the
/// `TCR_EL1` layout. EL2 without E2H has one TTBR0 range under `TCR_EL2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Regime {
    sctlr: u64,
    tcr: u64,
    ttbr0: u64,
    ttbr1: u64,
    split: bool,
    /// Bit position of the physical-address size field: IPS (34:32) or PS (18:16).
    ips_shift: u32,
}
impl Regime {
    fn of(cpu: &Cpu) -> Self {
        use super::cpu::{HCR_E2H, HCR_TGE};
        let hcr = cpu.system.hcr_el2;
        let el2_regime =
            cpu.system.el == 2 || (cpu.system.el == 0 && hcr & HCR_TGE != 0 && hcr & HCR_E2H != 0);
        if !el2_regime {
            return Self {
                sctlr: cpu.system.sctlr_el1,
                tcr: cpu.system.tcr_el1,
                ttbr0: cpu.system.ttbr0_el1,
                ttbr1: cpu.system.ttbr1_el1,
                split: true,
                ips_shift: 32,
            };
        }
        if hcr & HCR_E2H != 0 {
            Self {
                sctlr: cpu.system.sctlr_el2,
                tcr: cpu.system.tcr_el2,
                ttbr0: cpu.system.ttbr0_el2,
                ttbr1: cpu.system.ttbr1_el2,
                split: true,
                ips_shift: 32,
            }
        } else {
            Self {
                sctlr: cpu.system.sctlr_el2,
                tcr: cpu.system.tcr_el2,
                ttbr0: cpu.system.ttbr0_el2,
                ttbr1: 0,
                split: false,
                ips_shift: 16,
            }
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Context {
    regime: Regime,
    el: u8,
}
impl Context {
    fn of(cpu: &Cpu) -> Self {
        Self {
            regime: Regime::of(cpu),
            el: cpu.system.el,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Tlb {
    pub entries: [Entry; 256],
    pub generation: u64,
    context: Option<Context>,
}
impl Default for Tlb {
    fn default() -> Self {
        Self {
            entries: [Entry::default(); 256],
            generation: 1,
            context: None,
        }
    }
}
impl Tlb {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn flush(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation <= 1 {
            self.entries = [Entry::default(); 256];
        }
    }
    fn slot(virtual_address: u64) -> usize {
        ((virtual_address >> 12) & 255) as usize
    }
    /// Flush if the translation context (regime, tables, EL) changed. Called
    /// eagerly each dispatch because inline data-TLB hits never reach `translate`.
    pub fn update_context(&mut self, cpu: &Cpu) {
        let context = Context::of(cpu);
        if self.context != Some(context) {
            self.flush();
            self.context = Some(context);
        }
    }
}
fn enabled(cpu: &Cpu) -> Result<bool, TranslateError> {
    let regime = Regime::of(cpu);
    if regime.sctlr & 1 == 0 {
        return Ok(false);
    }
    if regime.tcr & 63 == 0 {
        return Err(TranslateError::MalformedTables);
    }
    Ok(true)
}
pub fn physical_bits(cpu: &Cpu) -> Result<u8, TranslateError> {
    if !enabled(cpu)? {
        return Ok(64);
    }
    let regime = Regime::of(cpu);
    // IPS (TCR_EL1 bits 34:32) or PS (TCR_EL2 bits 18:16).
    let field = (regime.tcr >> regime.ips_shift) & 7;
    Ok([32, 36, 40, 42, 48, 52, 56, 56][field as usize])
}
fn permitted(ap: u8, executable: bool, access: Access, el: u8) -> Result<(), TranslateError> {
    if (el == 0 && ap & 1 == 0)
        || (access == Access::Write && ap & 2 != 0)
        || (access == Access::Execute && !executable)
    {
        Err(TranslateError::PermissionFault)
    } else {
        Ok(())
    }
}
pub fn translate<M: GuestMemory + ?Sized>(
    tlb: &mut Tlb,
    cpu: &Cpu,
    memory: &mut M,
    virtual_address: u64,
    access: Access,
) -> Result<u64, TranslateError> {
    if !enabled(cpu)? {
        return Ok(virtual_address);
    }
    let regime = Regime::of(cpu);
    let t1sz = ((regime.tcr >> 16) & 63) as u32;
    // TTBR1's region size only means something while EPD1 leaves its walk enabled.
    let upper_walk = regime.split && regime.tcr & (1 << 23) == 0;
    if upper_walk && t1sz == 0 {
        return Err(TranslateError::MalformedTables);
    }
    if ((virtual_address >> 55) & 1 != 0 && virtual_address >> 48 != 0xffff)
        || ((virtual_address >> 55) & 1 == 0 && virtual_address >> 48 != 0)
    {
        return Err(TranslateError::TranslationFault);
    }
    let lower_bits = 64 - (regime.tcr & 63) as u32;
    let upper_bits = 64 - t1sz;
    let upper = if virtual_address >> lower_bits == 0 {
        false
    } else if upper_walk && virtual_address >> upper_bits == u64::MAX >> upper_bits {
        true
    } else {
        return Err(TranslateError::TranslationFault);
    };
    tlb.update_context(cpu);
    // Entries are per 4 KiB page: every address in a page resolves through the
    // same descriptors, and a 4 KiB piece of any larger granule or block maps
    // contiguously. `Entry::virtual_address` and `physical` hold page bases.
    let page = virtual_address & !PAGE_MASK;
    let slot = Tlb::slot(virtual_address);
    let entry = &tlb.entries[slot];
    if entry.generation == tlb.generation && entry.virtual_address == page {
        // A cached entry that denies the access is re-walked below: the
        // descriptor may have been upgraded since it was cached, which
        // hardware also resolves by retrying the walk.
        if permitted(entry.ap, entry.executable, access, cpu.system.el).is_ok() {
            return Ok(entry.physical | (virtual_address & PAGE_MASK));
        }
    }
    // Faults are not cached. ARMv8 never caches an invalid descriptor, so a mapping
    // installed later, without a TLBI or an ISB (as Linux's `ioremap` does), must be
    // seen by the next access.
    let found = walk(cpu, memory, virtual_address, access, upper)?;
    tlb.entries[slot] = Entry {
        virtual_address: page,
        physical: found.physical & !PAGE_MASK,
        generation: tlb.generation,
        ..found
    };
    Ok(found.physical)
}
fn walk<M: GuestMemory + ?Sized>(
    cpu: &Cpu,
    memory: &mut M,
    virtual_address: u64,
    access: Access,
    upper: bool,
) -> Result<Entry, TranslateError> {
    let regime = Regime::of(cpu);
    let tcr = regime.tcr;
    if (upper && tcr & (1 << 23) != 0) || (!upper && regime.split && tcr & (1 << 7) != 0) {
        return Err(TranslateError::TranslationFault);
    }
    let granule = if upper {
        [0u32, 14, 12, 16][((tcr >> 30) & 3) as usize]
    } else {
        [12u32, 16, 14, 0][((tcr >> 14) & 3) as usize]
    };
    if granule == 0 {
        return Err(TranslateError::MalformedTables);
    }
    // ASID is in TTBR bits 63:48, never part of a physical table address.
    let root = if upper { regime.ttbr1 } else { regime.ttbr0 };
    let va_bits = 64 - if upper { (tcr >> 16) & 63 } else { tcr & 63 };
    if va_bits > 48 || va_bits <= u64::from(granule) {
        return Err(TranslateError::MalformedTables);
    }
    let stride = granule - 3; // eight-byte descriptors per granule
    let levels = ((va_bits as u32 - granule) + stride - 1) / stride;
    if levels > 4 {
        return Err(TranslateError::MalformedTables);
    }
    let address_mask = 0x0000_ffff_ffff_ffff & !((1u64 << granule) - 1);
    let top_bits = va_bits as u32 - (granule + stride * (levels - 1));
    // Retain Mirage's full 4KB root table convention. Other granules have
    // reduced first-level tables, whose size and index follow the VA size.
    let root_alignment = if granule == 12 {
        granule
    } else {
        (top_bits + 3).max(6)
    };
    let mut table = root & 0x0000_ffff_ffff_ffff & !((1u64 << root_alignment) - 1);
    let mut no_user = false;
    let mut read_only = false;
    let mut pxn_table = false;
    let mut uxn_table = false;
    for level in (0..levels).rev() {
        let shift = granule + stride * level;
        let index_bits = if granule != 12 && level == levels - 1 {
            top_bits
        } else {
            stride
        };
        let index = (virtual_address >> shift) & ((1u64 << index_bits) - 1);
        let at = table.wrapping_add(index * 8);
        let mut word = [0; 8];
        memory
            .read(at, &mut word)
            .map_err(|_| TranslateError::TranslationFault)?;
        let descriptor = u64::from_le_bytes(word);
        let kind = descriptor & 3;
        if kind == 3 && level != 0 {
            no_user |= descriptor & (1 << 61) != 0;
            read_only |= descriptor & (1 << 62) != 0;
            pxn_table |= descriptor & (1 << 59) != 0;
            uxn_table |= descriptor & (1 << 60) != 0;
            table = descriptor & address_mask;
            continue;
        }
        // Level 0 blocks and 64KB level 1 blocks are architecturally reserved.
        let architectural_level = 3 - level;
        let block_allowed = architectural_level >= if granule == 16 { 2 } else { 1 };
        if !((kind == 3 && level == 0) || (kind == 1 && level != 0 && block_allowed)) {
            return Err(TranslateError::TranslationFault);
        }
        let mask = (1u64 << shift) - 1;
        let mut ap = ((descriptor >> 6) & 3) as u8;
        if no_user {
            ap &= !1;
        }
        if read_only {
            ap |= 2;
        }
        let executable = if cpu.system.el == 0 {
            !uxn_table && descriptor & (1 << 54) == 0
        } else {
            !pxn_table && descriptor & (1 << 53) == 0
        };
        let found = Entry {
            physical: (descriptor & address_mask & !mask) | (virtual_address & mask),
            ap,
            executable,
            ..Entry::default()
        };
        permitted(found.ap, found.executable, access, cpu.system.el)?;
        // Mirage's hardware-style AF update is best-effort. Read-only reads
        // do not require AF, and a failed AF write must not deny the access.
        if descriptor & (1 << 10) == 0 && !(access == Access::Read && found.ap & 2 != 0) {
            let _ = memory.write(at, &(descriptor | (1 << 10)).to_le_bytes());
        }
        return Ok(found);
    }
    Err(TranslateError::TranslationFault)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::memory::MemoryError;
    struct Memory {
        bytes: Vec<u8>,
        reads: core::cell::Cell<usize>,
    }
    impl Memory {
        fn new() -> Self {
            Self {
                bytes: vec![0; 0x80000],
                reads: core::cell::Cell::new(0),
            }
        }
        fn put(&mut self, at: u64, word: u64) {
            self.bytes[at as usize..at as usize + 8].copy_from_slice(&word.to_le_bytes());
        }
        fn get(&self, at: usize) -> u64 {
            u64::from_le_bytes(self.bytes[at..at + 8].try_into().unwrap())
        }
        fn chain(
            &mut self,
            root: u64,
            virtual_address: u64,
            granule: u32,
            levels: u32,
            leaf_level: u32,
            leaf: u64,
        ) {
            let stride = granule - 3;
            let mut table = root;
            for level in (leaf_level + 1..levels).rev() {
                let index = (virtual_address >> (granule + stride * level)) & ((1 << stride) - 1);
                let next = table + (1 << granule);
                self.put(table + index * 8, next | 3);
                table = next;
            }
            let index = (virtual_address >> (granule + stride * leaf_level)) & ((1 << stride) - 1);
            self.put(table + index * 8, leaf);
        }
    }
    impl GuestMemory for Memory {
        fn read(&self, address: u64, out: &mut [u8]) -> Result<(), MemoryError> {
            let start = usize::try_from(address).ok();
            let bytes = start.and_then(|start| {
                start
                    .checked_add(out.len())
                    .and_then(|end| self.bytes.get(start..end))
            });
            let bytes = bytes.ok_or(MemoryError::AccessFault {
                address,
                length: out.len(),
            })?;
            self.reads.set(self.reads.get() + 1);
            out.copy_from_slice(bytes);
            Ok(())
        }
        fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryError> {
            let start = usize::try_from(address).ok();
            let target = start.and_then(|start| {
                start
                    .checked_add(bytes.len())
                    .and_then(|end| self.bytes.get_mut(start..end))
            });
            let target = target.ok_or(MemoryError::AccessFault {
                address,
                length: bytes.len(),
            })?;
            target.copy_from_slice(bytes);
            Ok(())
        }
    }
    fn cpu() -> Cpu {
        let mut cpu = Cpu::default();
        cpu.system.sctlr_el1 = 1;
        cpu.system.tcr_el1 = 16 | (17 << 16) | (2 << 30) | (4 << 32);
        cpu
    }
    #[test]
    fn disabled_translation_is_identity_and_noncanonical_tags_fault_when_enabled() {
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        assert_eq!(
            translate(
                &mut tlb,
                &Cpu::default(),
                &mut memory,
                0xab00_0000_0000_1234,
                Access::Read
            ),
            Ok(0xab00_0000_0000_1234)
        );
        assert_eq!(
            translate(
                &mut tlb,
                &cpu(),
                &mut memory,
                0xab00_0000_0000_1234,
                Access::Read
            ),
            Err(TranslateError::TranslationFault)
        );
        assert_eq!(memory.reads.get(), 0);
    }
    #[test]
    fn four_level_page_walk_access_flag_and_tlb_flush() {
        let cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0x1234_5678, 12, 4, 0, 0x8000_0003);
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0x1234_5678, Access::Read),
            Ok(0x8000_0678)
        );
        let leaf_at = 0x3000 + (((0x1234_5678u64 >> 12) & 511) * 8);
        assert_ne!(memory.get(leaf_at as usize) & (1 << 10), 0);
        memory.put(leaf_at, 0x9000_0003);
        let count = memory.reads.get();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0x1234_5678, Access::Read),
            Ok(0x8000_0678)
        );
        assert_eq!(memory.reads.get(), count);
        tlb.flush();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0x1234_5678, Access::Read),
            Ok(0x9000_0678)
        );
    }
    #[test]
    fn block_walk_covers_each_offset() {
        let cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0x20_0000, 12, 4, 1, 0x4000_0001);
        for address in [0x20_0000, 0x20_1234, 0x3f_ffff] {
            assert_eq!(
                translate(&mut tlb, &cpu, &mut memory, address, Access::Write),
                Ok(0x4000_0000 | (address & 0x1f_ffff))
            );
        }
    }
    #[test]
    fn permissions_are_rechecked_on_hits_and_permission_faults_are_not_cached() {
        let cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0, 12, 4, 0, 0x8000_0003 | (2 << 6) | (1 << 53));
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Execute),
            Err(TranslateError::PermissionFault)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x8000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Write),
            Err(TranslateError::PermissionFault)
        );
        assert_eq!(memory.get(0x3000) & (1 << 10), 0);
    }
    #[test]
    fn tlb_hits_for_any_offset_within_a_cached_page() {
        let cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0, 12, 4, 0, 0x8000_0003 | (1 << 6));
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0x10, Access::Read),
            Ok(0x8000_0010)
        );
        let reads = memory.reads.get();
        for offset in [0x0, 0x8, 0x123, 0xfff] {
            assert_eq!(
                translate(&mut tlb, &cpu, &mut memory, offset, Access::Read),
                Ok(0x8000_0000 + offset)
            );
        }
        assert_eq!(memory.reads.get(), reads, "same page must not walk again");
    }
    #[test]
    fn invalid_descriptors_fail_closed() {
        let cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.put(0, 0x1002);
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Err(TranslateError::TranslationFault)
        );
        let count = memory.reads.get();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Err(TranslateError::TranslationFault)
        );
        // A fault is not cached: the repeated access walks the tables again.
        assert!(memory.reads.get() > count);
    }
    #[test]
    fn table_descriptor_bits_eleven_to_two_are_ignored() {
        // Table descriptors ignore bits 11:2 (ARM ARM D8.3.1). Set them on every table on the
        // walk for address 0: the page at 0x4000_0000 must still resolve.
        let cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0, 12, 4, 0, 0x4000_0403);
        for table in [0usize, 0x1000, 0x2000] {
            let descriptor = memory.get(table);
            memory.put(table as u64, descriptor | 0x3f0);
        }
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
    }
    #[test]
    fn upper_root_a1_asid_and_regime_changes_do_not_alias() {
        let mut cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        let high = 0xffff_8000_0000_1000;
        memory.chain(0, 0x1000, 12, 4, 0, 0x8000_0003);
        memory.chain(0x5000, high, 12, 4, 0, 0x4000_0003);
        cpu.system.ttbr1_el1 = (0x1234 << 48) | 0x5000;
        cpu.system.tcr_el1 |= 1 << 22;
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, high, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0x1000, Access::Read),
            Ok(0x8000_0000)
        );
        memory.put(0x3008, 0x9000_0003);
        cpu.system.ttbr0_el1 = 0x5678 << 48;
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0x1000, Access::Read),
            Ok(0x9000_0000)
        );
        cpu.system.tcr_el1 |= 1 << 23;
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, high, Access::Read),
            Err(TranslateError::TranslationFault)
        );
    }
    #[test]
    fn non_four_kilobyte_granules_use_their_actual_descriptor_stride() {
        for (granule, tg0, tg1, levels) in [(14, 2, 1, 4), (16, 1, 3, 3)] {
            let mut cpu = cpu();
            let mut memory = Memory::new();
            let mut tlb = Tlb::new();
            cpu.system.tcr_el1 = 16 | (17 << 16) | (tg0 << 14) | (tg1 << 30) | (4 << 32);
            // Force an index beyond 511 at a lower level.
            let address = (0x612u64 << granule) | 0x2345;
            memory.chain(0, address, granule, levels, 0, 0x8000_0003);
            assert_eq!(
                translate(&mut tlb, &cpu, &mut memory, address, Access::Read),
                Ok(0x8000_2345)
            );
        }
    }
    #[test]
    fn generation_wrap_clears_entries() {
        let mut tlb = Tlb::new();
        tlb.generation = u64::MAX;
        tlb.entries[0].generation = 0;
        tlb.entries[0].physical = 123;
        tlb.flush();
        assert_eq!(tlb.entries[0].physical, 0);
    }
    #[test]
    fn upper_64k_root_uses_reduced_index_and_table_alignment() {
        let mut cpu = cpu();
        cpu.system.tcr_el1 = 16 | (17 << 16) | (1 << 14) | (3 << 30) | (4 << 32);
        cpu.system.ttbr1_el1 = (0xabcd << 48) | 0x5000;
        let mut memory = Memory::new();
        memory.put(0x5000, 0x10003);
        memory.put(0x10000, 0x20003);
        memory.put(0x20000, 0x4000_0003);
        assert_eq!(
            translate(
                &mut Tlb::new(),
                &cpu,
                &mut memory,
                0xffff_8000_0000_1234,
                Access::Execute
            ),
            Ok(0x4000_1234)
        );
    }
    #[test]
    fn el0_cannot_access_kernel_pages_and_tlb_permissions_follow_the_level() {
        let mut cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0, 12, 4, 0, 0x4000_0403);
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        cpu.system.el = 0;
        for access in [Access::Read, Access::Write, Access::Execute] {
            assert_eq!(
                translate(&mut tlb, &cpu, &mut memory, 0, access),
                Err(TranslateError::PermissionFault)
            );
        }
        memory.put(0x3000, 0x4000_0443);
        tlb.flush();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Write),
            Ok(0x4000_0000)
        );
    }
    #[test]
    fn execute_never_selects_uxn_for_el0_and_pxn_for_el1_on_walks_and_hits() {
        let mut cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0, 12, 4, 0, 0x4000_0443 | (1 << 54));
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Execute),
            Ok(0x4000_0000)
        );
        cpu.system.el = 0;
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Execute),
            Err(TranslateError::PermissionFault)
        );
        memory.put(0x3000, 0x4000_0443 | (1 << 53));
        tlb.flush();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Execute),
            Ok(0x4000_0000)
        );
        cpu.system.el = 1;
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Execute),
            Err(TranslateError::PermissionFault)
        );
    }
    #[test]
    fn configured_va_region_holes_do_not_alias_lower_root_indices() {
        let mut cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        cpu.system.tcr_el1 = 25 | (25 << 16) | (2 << 30) | (4 << 32);
        memory.chain(0, 0, 12, 3, 0, 0x4000_0403);
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        let reads = memory.reads.get();
        for address in [
            1 << 39,
            1 << 40,
            0x0000_8000_0000_0000,
            0xffff_0000_0000_0000,
        ] {
            assert_eq!(
                translate(&mut tlb, &cpu, &mut memory, address, Access::Read),
                Err(TranslateError::TranslationFault)
            );
        }
        assert_eq!(memory.reads.get(), reads);
        let high = 0xffff_ff80_0000_0000;
        memory.chain(0x5000, high, 12, 3, 0, 0x8000_0403);
        cpu.system.ttbr1_el1 = 0x5000;
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, high, Access::Read),
            Ok(0x8000_0000)
        );
    }
    #[test]
    fn ancestor_ap_and_xn_restrictions_survive_tlb_hits() {
        let mut cpu = cpu();
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        memory.chain(0, 0, 12, 4, 0, 0x4000_0443);
        memory.put(0, 0x1003 | (1 << 62));
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Write),
            Err(TranslateError::PermissionFault)
        );
        memory.put(0, 0x1003 | (1 << 59));
        tlb.flush();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Execute),
            Err(TranslateError::PermissionFault)
        );
        cpu.system.el = 0;
        memory.put(0, 0x1003 | (1 << 60));
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Execute),
            Err(TranslateError::PermissionFault)
        );
        memory.put(0, 0x1003 | (1 << 61));
        tlb.flush();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Err(TranslateError::PermissionFault)
        );
    }
    #[test]
    fn el2_walks_only_ttbr0_el2_under_tcr_el2_and_has_no_upper_range() {
        let mut cpu = Cpu::default();
        cpu.system.el = 2;
        cpu.system.sctlr_el2 = 1;
        // T0SZ = 16 (48-bit VAs), TG0 = 4 KiB, PS = 40 bits.
        cpu.system.tcr_el2 = 16 | (2 << 16);
        cpu.system.ttbr0_el2 = 0;
        // The EL1&0 regime is disabled: EL2 translates from its own registers.
        cpu.system.sctlr_el1 = 0;
        let mut memory = Memory::new();
        memory.chain(0, 0, 12, 4, 0, 0x4000_0443);
        let mut tlb = Tlb::new();
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0, Access::Read),
            Ok(0x4000_0000)
        );
        assert_eq!(physical_bits(&cpu), Ok(40));
        // Bit 55 set with an all-ones upper half: a TTBR1 address, which EL2 lacks.
        assert_eq!(
            translate(
                &mut tlb,
                &cpu,
                &mut memory,
                0xffff_8000_0000_0000,
                Access::Read
            ),
            Err(TranslateError::TranslationFault)
        );
    }
    #[test]
    fn epd1_disables_the_upper_walk_so_its_t1sz_is_never_checked() {
        // The tinted-boot firmware's TCR_EL1: T0SZ 25 (a 39-bit lower range), EPD1 set,
        // and T1SZ 0, which is only meaningful while the TTBR1 walk is enabled.
        let mut cpu = cpu();
        cpu.system.tcr_el1 = 25 | (1 << 23) | (2 << 30) | (4 << 32);
        let mut memory = Memory::new();
        let mut tlb = Tlb::new();
        // A 1 GiB block at 0x4000_0000 in the TTBR0 root table, which sits at 0.
        memory.chain(0, 0x4000_0000, 12, 3, 2, 0x4000_0401);
        assert_eq!(
            translate(&mut tlb, &cpu, &mut memory, 0x4000_0000, Access::Execute),
            Ok(0x4000_0000)
        );
        assert_eq!(
            translate(
                &mut tlb,
                &cpu,
                &mut memory,
                0xffff_8000_0000_0000,
                Access::Read
            ),
            Err(TranslateError::TranslationFault)
        );
    }
}
