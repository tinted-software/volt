//! Apple's GL2 monitor (SPTM), entered at EL2 with the MMU off.
//!
//! The boot firmware copies the monitor image into DRAM and jumps to its entry at the
//! image's *physical* address, with x0 the physical address of a boot structure. The
//! monitor builds its own translation tables (its `__BOOTDATA` and `__LATE_CONST`
//! segments are zero in the file), sets `HCR_EL2.{E2H,TGE}`, and turns the MMU on
//! itself. [`load`] does the same. The structure's fields come from the entry code and
//! from where the monitor reads its copy of it:
//!
//! | offset | value                                                                  |
//! |--------|------------------------------------------------------------------------|
//! | 0      | count (the code tests `>= 3`)                                          |
//! | 8      | virtual address of the first DRAM byte in the monitor's map (`x22`)    |
//! | 16     | physical address of the first DRAM byte (`x23`)                        |
//! | 24     | bytes of DRAM to map (`x24`)                                           |
//! | 0x60   | device tree address, as the monitor sees it (the linear map)           |
//! | 0x68   | device tree size in bytes (32 bits)                                    |
//! | 1152   | physical address of a scratch buffer, zeroed and stamped `"SPTM"`      |
//!
//! The monitor maps all of DRAM linearly (`virtual = physical + [8] - [16]`) in 32 MiB
//! blocks, which drop the low 25 bits of both addresses, so both bases are block-aligned.
//! Its own image sits inside that map at its link address, which fixes `[8]`: the image's
//! physical base equals its virtual base modulo 32 MiB. Everything the monitor reads
//! before it has its own mappings (the boot structure, the device tree, the scratch
//! buffer, the kernel collection it protects) is reached through that map, so it all lies
//! in DRAM. Seen with the per-instruction and memory traces (`Machine::set_step_trace`,
//! `Machine::set_memory_trace`).

use crate::{
    memory::{GuestMemory, MemoryError},
    system::xnu,
};

/// The monitor's block-mapping granularity: 32 MiB (a 16 KiB-granule level-2 block).
pub const BLOCK: u64 = 32 << 20;
/// Offsets of the device-tree address and size in the boot structure, read from the
/// monitor's copy of it (`ldr x9, [x8]; ldr w8, [x8, #8]` at `+0x60`).
const DEVICE_TREE_SLOT: u64 = 0x60;
/// Offset of the scratch-buffer pointer in the boot structure (`ldr x5, [x20, #1152]`).
const SCRATCH_SLOT: u64 = 1152;
/// Bytes the entry code zeroes at the scratch buffer (`mov x7, #0x17c0`).
const SCRATCH_BYTES: u64 = 0x17c0;
/// Bytes of boot structure written: through the scratch slot.
const BOOT_BYTES: usize = SCRATCH_SLOT as usize + 8;

#[derive(Debug)]
pub enum Error {
    /// The image is not a Mach-O executable the loader can place.
    Image(crate::system::boot::Error),
    Memory(MemoryError),
    /// The physical image base is not congruent with the virtual base modulo 32 MiB.
    Misaligned {
        virtual_base: u64,
        physical_base: u64,
    },
    /// DRAM is not 32 MiB-aligned, or is smaller than one block. A size under one block
    /// makes the monitor's mapping loop wrap and run for ever.
    Dram {
        base: u64,
        size: u64,
    },
    /// A part the monitor reads through its linear map is outside DRAM.
    OutsideDram {
        what: &'static str,
        physical: u64,
    },
    /// The device tree is not a well-formed Apple flattened device tree.
    DeviceTree(&'static str),
    /// The entry point lies in no part of the image.
    NoEntryPart,
}

impl From<MemoryError> for Error {
    fn from(error: MemoryError) -> Self {
        Self::Memory(error)
    }
}

/// Where the firmware puts each part of the monitor, all physical addresses.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// First byte of DRAM and its size, both multiples of [`BLOCK`].
    pub dram_base: u64,
    pub dram_size: u64,
    /// Physical address of the image's lowest segment (`__TEXT`). Must be congruent
    /// with that segment's link address modulo [`BLOCK`], and inside DRAM.
    pub image_phys: u64,
    /// Physical address of the boot structure, inside DRAM.
    pub boot_phys: u64,
    /// Physical address of the monitor's scratch buffer, inside DRAM.
    pub scratch_phys: u64,
    /// Physical address the device tree is copied to, inside DRAM.
    pub device_tree_phys: u64,
}

/// A monitor image placed in memory, ready to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Monitor {
    /// Physical address of the entry point (`LC_UNIXTHREAD` pc, relocated).
    pub entry: u64,
    /// Physical address of the boot structure, passed in x0.
    pub boot_phys: u64,
}

/// Bit 31 of an ADT property length marks a placeholder property; the length is the low 31
/// bits. The monitor adds the raw length to the property's address, so the flag would make
/// every walk overflow its bounds check.
const PLACEHOLDER: u32 = 1 << 31;

/// A copy of the Apple flattened device tree `tree` with every placeholder flag cleared.
/// Nodes are `{properties: u32, children: u32}`, each property a 32-byte name, a `u32`
/// length and its value padded to four bytes, then the children.
pub fn without_placeholder_flags(tree: &[u8]) -> Result<Vec<u8>, Error> {
    const TRUNCATED: Error = Error::DeviceTree("device tree is truncated");
    fn word(tree: &[u8], at: usize) -> Result<u32, Error> {
        let bytes = tree.get(at..at + 4).ok_or(TRUNCATED)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }
    fn node(tree: &mut [u8], at: usize) -> Result<usize, Error> {
        let properties = word(tree, at)?;
        let children = word(tree, at + 4)?;
        let mut at = at + 8;
        for _ in 0..properties {
            let length = word(tree, at + 32)? & !PLACEHOLDER;
            tree.get_mut(at + 32..at + 36)
                .ok_or(TRUNCATED)?
                .copy_from_slice(&length.to_le_bytes());
            at += 36 + (length as usize).next_multiple_of(4);
            if at > tree.len() {
                return Err(TRUNCATED);
            }
        }
        for _ in 0..children {
            at = node(tree, at)?;
        }
        Ok(at)
    }
    let mut copy = tree.to_vec();
    node(&mut copy, 0)?;
    Ok(copy)
}

/// Appends a property `name = value` to the node at `path` of the device tree `tree`
/// (`path` names children of the root, e.g. `["chosen", "memory-map"]`).
pub fn add_property(
    tree: &[u8],
    path: &[&[u8]],
    name: &str,
    value: &[u8],
) -> Result<Vec<u8>, Error> {
    const TRUNCATED: Error = Error::DeviceTree("device tree is truncated");
    const NAME: usize = 32;
    fn word(tree: &[u8], at: usize) -> Result<u32, Error> {
        let bytes = tree.get(at..at + 4).ok_or(TRUNCATED)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }
    /// The offset just past the properties of the node at `at`, and its name.
    fn properties(tree: &[u8], at: usize) -> Result<(usize, Option<&[u8]>), Error> {
        let mut found = None;
        let mut cursor = at + 8;
        for _ in 0..word(tree, at)? {
            let length = (word(tree, cursor + NAME)? & !PLACEHOLDER) as usize;
            let value = cursor + NAME + 4;
            if tree.get(cursor..cursor + 5) == Some(b"name\0") {
                found = tree.get(value..value + length);
            }
            cursor = value + length.next_multiple_of(4);
            if cursor > tree.len() {
                return Err(TRUNCATED);
            }
        }
        Ok((cursor, found.map(|n| n.strip_suffix(b"\0").unwrap_or(n))))
    }
    /// The offset just past the node at `at` and everything under it.
    fn skip(tree: &[u8], at: usize) -> Result<usize, Error> {
        let (mut cursor, _) = properties(tree, at)?;
        for _ in 0..word(tree, at + 4)? {
            cursor = skip(tree, cursor)?;
        }
        Ok(cursor)
    }
    /// The offset of the child of the node at `at` named `wanted`.
    fn child(tree: &[u8], at: usize, wanted: &[u8]) -> Result<usize, Error> {
        let (mut cursor, _) = properties(tree, at)?;
        for _ in 0..word(tree, at + 4)? {
            if properties(tree, cursor)?.1 == Some(wanted) {
                return Ok(cursor);
            }
            cursor = skip(tree, cursor)?;
        }
        Err(Error::DeviceTree("device tree has no such node"))
    }
    if name.len() >= NAME || value.len() > u32::MAX as usize {
        return Err(Error::DeviceTree("property name or value is too large"));
    }
    let mut node = 0;
    for part in path {
        node = child(tree, node, part)?;
    }
    let (end, _) = properties(tree, node)?;
    let mut property = vec![0u8; NAME + 4 + value.len()];
    property[..name.len()].copy_from_slice(name.as_bytes());
    property[NAME..NAME + 4].copy_from_slice(&(value.len() as u32).to_le_bytes());
    property[NAME + 4..].copy_from_slice(value);
    property.resize(property.len().next_multiple_of(4), 0);
    let mut out = Vec::with_capacity(tree.len() + property.len());
    out.extend_from_slice(&tree[..end]);
    out.extend_from_slice(&property);
    out.extend_from_slice(&tree[end..]);
    let count = word(&out, node)? + 1;
    out[node..node + 4].copy_from_slice(&count.to_le_bytes());
    Ok(out)
}

/// Appends the property `name = {address: u64, size: u64}` to the device tree's
/// `chosen/memory-map` node, the way the boot firmware records a region it loaded.
pub fn add_memory_map_region(
    tree: &[u8],
    name: &str,
    address: u64,
    size: u64,
) -> Result<Vec<u8>, Error> {
    let mut value = Vec::with_capacity(16);
    value.extend(address.to_le_bytes());
    value.extend(size.to_le_bytes());
    add_property(tree, &[b"chosen", b"memory-map"], name, &value)
}

/// Places the `MH_EXECUTE` monitor `image` and `device_tree`, and writes the boot structure.
pub fn load(
    memory: &mut impl GuestMemory,
    image: &[u8],
    device_tree: &[u8],
    layout: &Layout,
) -> Result<Monitor, Error> {
    let kernel = xnu::parse(image).map_err(Error::Image)?;
    let device_tree = without_placeholder_flags(device_tree)?;
    let base = kernel
        .segments
        .iter()
        .filter(|s| s.vmsize != 0)
        .map(|s| s.vmaddr)
        .min()
        .unwrap_or(0);
    let (dram, size) = (layout.dram_base, layout.dram_size);
    if dram & (BLOCK - 1) != 0 || size & (BLOCK - 1) != 0 || size < BLOCK {
        return Err(Error::Dram { base: dram, size });
    }
    if (base ^ layout.image_phys) & (BLOCK - 1) != 0 {
        return Err(Error::Misaligned {
            virtual_base: base,
            physical_base: layout.image_phys,
        });
    }
    let end = kernel
        .segments
        .iter()
        .map(|s| s.vmaddr + s.vmsize)
        .max()
        .unwrap_or(base);
    for (what, physical, length) in [
        ("image", layout.image_phys, end - base),
        ("boot structure", layout.boot_phys, BOOT_BYTES as u64),
        ("scratch buffer", layout.scratch_phys, SCRATCH_BYTES),
        (
            "device tree",
            layout.device_tree_phys,
            device_tree.len() as u64,
        ),
    ] {
        if physical < dram || physical + length > dram + size {
            return Err(Error::OutsideDram { what, physical });
        }
    }
    for segment in &kernel.segments {
        if segment.vmsize == 0 {
            continue;
        }
        let mut bytes = vec![0u8; segment.vmsize as usize];
        let file = &image[segment.fileoff as usize..][..segment.filesize as usize];
        bytes[..file.len()].copy_from_slice(file);
        memory.write(layout.image_phys + (segment.vmaddr - base), &bytes)?;
    }
    // The image sits at its link address in the linear map, which fixes where DRAM starts.
    let dram_va = base.wrapping_sub(layout.image_phys - dram);
    let linear = dram_va.wrapping_sub(dram);
    let mut boot = [0u8; BOOT_BYTES];
    let mut put = |at: u64, value: u64| {
        boot[at as usize..at as usize + 8].copy_from_slice(&value.to_le_bytes());
    };
    put(0, 3);
    put(8, dram_va);
    put(16, dram);
    put(24, size);
    put(
        DEVICE_TREE_SLOT,
        layout.device_tree_phys.wrapping_add(linear),
    );
    put(DEVICE_TREE_SLOT + 8, device_tree.len() as u64);
    put(SCRATCH_SLOT, layout.scratch_phys);
    memory.write(layout.device_tree_phys, &device_tree)?;
    memory.write(layout.boot_phys, &boot)?;
    memory.write(layout.scratch_phys, &vec![0u8; SCRATCH_BYTES as usize])?;
    Ok(Monitor {
        entry: layout.image_phys + (kernel.entry - base),
        boot_phys: layout.boot_phys,
    })
}

/// The lowest nonempty segment address of `kernel`: the image's link base.
fn image_base(kernel: &xnu::Kernel) -> u64 {
    kernel
        .segments
        .iter()
        .filter(|s| s.vmsize != 0)
        .map(|s| s.vmaddr)
        .min()
        .unwrap_or(0)
}

/// Places a Mach-O image so that its lowest segment is at physical `phys`. `phys` must
/// match the link base modulo [`BLOCK`], so the monitor's linear map reaches the image at
/// its link address. Returns the physical entry point.
pub fn place_image(memory: &mut impl GuestMemory, image: &[u8], phys: u64) -> Result<u64, Error> {
    let kernel = xnu::parse(image).map_err(Error::Image)?;
    let base = image_base(&kernel);
    if (base ^ phys) & (BLOCK - 1) != 0 {
        return Err(Error::Misaligned {
            virtual_base: base,
            physical_base: phys,
        });
    }
    write_segments(memory, image, &kernel, base, phys)?;
    Ok(phys + (kernel.entry - base))
}

/// Writes each segment's file bytes, zero-filled to its size, at `phys + (vmaddr - base)`.
fn write_segments(
    memory: &mut impl GuestMemory,
    image: &[u8],
    kernel: &xnu::Kernel,
    base: u64,
    phys: u64,
) -> Result<(), Error> {
    for segment in &kernel.segments {
        if segment.vmsize == 0 {
            continue;
        }
        let mut bytes = vec![0u8; segment.vmsize as usize];
        let file = &image[segment.fileoff as usize..][..segment.filesize as usize];
        bytes[..file.len()].copy_from_slice(file);
        memory.write(phys + (segment.vmaddr - base), &bytes)?;
    }
    Ok(())
}

/// A named span of a Mach-O image, in the monitor's naming: link addresses `lo..hi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub name: &'static str,
    pub lo: u64,
    pub hi: u64,
}

/// The parts of a Mach-O image, as the monitor names them. `ro` covers `__TEXT`,
/// `__PRELINK_TEXT`, `__DATA_CONST` and `__LATE_CONST`; `rx` covers `__TEXT_EXEC` and
/// `__LAST`; `bx` is `__TEXT_BOOT_EXEC`; `rw` covers `__DATA` and `__BOOTDATA`; `le` is
/// `__LINKEDIT`; `rs` is `__DATA_SPTM` where present; `virt` is the whole image; `entry`
/// is the segment holding the entry point. A part is the smallest span covering its
/// segments and is copied as one piece, so the monitor maps it contiguously.
pub fn image_parts(image: &[u8]) -> Result<Vec<Part>, Error> {
    let kernel = xnu::parse(image).map_err(Error::Image)?;
    let base = image_base(&kernel);
    let end = kernel
        .segments
        .iter()
        .map(|s| s.vmaddr + s.vmsize)
        .max()
        .unwrap_or(base);
    let span = |names: &[&[u8]]| -> Option<(u64, u64)> {
        kernel
            .segments
            .iter()
            .filter(|s| s.vmsize != 0 && names.contains(&s.name()))
            .map(|s| (s.vmaddr, s.vmaddr + s.vmsize))
            .fold(None, |acc, (lo, hi)| match acc {
                None => Some((lo, hi)),
                Some((a, b)) => Some((a.min(lo), b.max(hi))),
            })
    };
    let entry = kernel
        .segments
        .iter()
        .find(|s| s.vmsize != 0 && (s.vmaddr..s.vmaddr + s.vmsize).contains(&kernel.entry))
        .map(|s| (s.vmaddr, s.vmaddr + s.vmsize));
    let spans = [
        (
            "ro",
            span(&[
                b"__TEXT",
                b"__PRELINK_TEXT",
                b"__DATA_CONST",
                b"__LATE_CONST",
            ]),
        ),
        ("rx", span(&[b"__TEXT_EXEC", b"__LAST"])),
        ("bx", span(&[b"__TEXT_BOOT_EXEC"])),
        ("rw", span(&[b"__DATA", b"__BOOTDATA"])),
        ("le", span(&[b"__LINKEDIT"])),
        ("rs", span(&[b"__DATA_SPTM"])),
        ("virt", Some((base, end))),
        ("entry", entry),
    ];
    Ok(spans
        .into_iter()
        .filter_map(|(name, range)| range.map(|(lo, hi)| Part { name, lo, hi }))
        .collect())
}

/// Copies the link span of `part` from `image` to physical `pa`: the file bytes of each
/// segment inside it, zeros elsewhere.
pub fn write_part(
    memory: &mut impl GuestMemory,
    image: &[u8],
    part: &Part,
    pa: u64,
) -> Result<(), Error> {
    let kernel = xnu::parse(image).map_err(Error::Image)?;
    let mut bytes = vec![0u8; (part.hi - part.lo) as usize];
    for segment in &kernel.segments {
        let lo = segment.vmaddr.max(part.lo);
        let hi = (segment.vmaddr + segment.filesize).min(part.hi);
        if lo >= hi {
            continue;
        }
        let from = segment.fileoff + (lo - segment.vmaddr);
        let file = &image[from as usize..][..(hi - lo) as usize];
        bytes[(lo - part.lo) as usize..(hi - part.lo) as usize].copy_from_slice(file);
    }
    memory.write(pa, &bytes)?;
    Ok(())
}

/// Writes every part of `image` at the address `place` gives it, and returns the image's
/// `chosen/memory-map` regions as `(prefix-part, address, size)`. `entry` is not written:
/// it lies at the same offset within the part that holds the entry point.
pub fn place_parts(
    memory: &mut impl GuestMemory,
    image: &[u8],
    prefix: &str,
    place: impl Fn(&Part) -> u64,
) -> Result<Vec<(String, u64, u64)>, Error> {
    let parts = image_parts(image)?;
    let mut regions = Vec::new();
    for part in parts.iter().filter(|p| p.name != "entry") {
        let pa = place(part);
        write_part(memory, image, part, pa)?;
        regions.push((format!("{prefix}-{}", part.name), pa, part.hi - part.lo));
    }
    if let Some(entry) = parts.iter().find(|p| p.name == "entry") {
        let holder = parts
            .iter()
            .find(|p| !matches!(p.name, "entry" | "virt") && p.lo <= entry.lo && entry.hi <= p.hi)
            .ok_or(Error::NoEntryPart)?;
        let pa = place(holder) + (entry.lo - holder.lo);
        regions.push((format!("{prefix}-entry"), pa, entry.hi - entry.lo));
    }
    Ok(regions)
}

/// Appends every region of `regions` to `chosen/memory-map` of `tree`, in order.
pub fn add_memory_map_regions(
    tree: &[u8],
    regions: &[(String, u64, u64)],
) -> Result<Vec<u8>, Error> {
    let mut tree = tree.to_vec();
    for (name, address, size) in regions {
        tree = add_memory_map_region(&tree, name, *address, *size)?;
    }
    Ok(tree)
}

/// Sets the CPU to the state the firmware leaves for the monitor: EL2, MMU off, at the
/// entry's physical address, with the boot structure in x0.
pub fn prepare_cpu(cpu: &mut crate::aarch64::cpu::Cpu, monitor: &Monitor) {
    let system = &mut cpu.system;
    system.el = 2;
    system.spsel = true;
    // EL1 is AArch64. The monitor sets `E2H` and `TGE` itself.
    system.hcr_el2 = 1 << 31;
    system.sctlr_el2 = 0;
    system.daif = 15;
    cpu.pc = monitor.entry;
    cpu.x[0] = monitor.boot_phys;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{PhysicalMemory, Region};

    /// A minimal `MH_EXECUTE` arm64 image: one `__TEXT` segment holding `payload`,
    /// linked at `vmaddr`, with an `LC_UNIXTHREAD` entry `entry_offset` bytes in.
    fn image(vmaddr: u64, entry_offset: u64, payload: &[u8]) -> Vec<u8> {
        let thread = 8 + 8 + 272;
        let segment = 72;
        let header = 32;
        let file_size = 0x4000u64;
        let mut out = Vec::new();
        for word in [
            0xfeed_facf_u32,
            0x0100_000c,
            0,
            2,
            2,
            (segment + thread) as u32,
            0,
            0,
        ] {
            out.extend(word.to_le_bytes());
        }
        out.extend(0x19u32.to_le_bytes());
        out.extend((segment as u32).to_le_bytes());
        let mut name = [0u8; 16];
        name[..6].copy_from_slice(b"__TEXT");
        out.extend(name);
        for word in [vmaddr, file_size, 0, file_size] {
            out.extend(word.to_le_bytes());
        }
        for word in [5u32, 5, 0, 0] {
            out.extend(word.to_le_bytes());
        }
        out.extend(5u32.to_le_bytes());
        out.extend((thread as u32).to_le_bytes());
        out.extend(6u32.to_le_bytes());
        out.extend(68u32.to_le_bytes());
        let mut state = [0u64; 34];
        state[32] = vmaddr + entry_offset;
        for word in state {
            out.extend(word.to_le_bytes());
        }
        out.resize(header + segment + thread, 0);
        out.resize(0x200, 0);
        out.extend_from_slice(payload);
        out.resize(file_size as usize, 0);
        out
    }

    const DRAM: u64 = 0x8_0000_0000;
    const VMADDR: u64 = 0xffff_fff0_2700_4000;

    fn memory() -> PhysicalMemory {
        PhysicalMemory::new(vec![
            Region::ram(DRAM, vec![0; (2 * BLOCK) as usize]).unwrap(),
        ])
        .unwrap()
    }

    /// Two blocks of DRAM, the image at its link offset in the first, the rest below it.
    fn layout() -> Layout {
        Layout {
            dram_base: DRAM,
            dram_size: 2 * BLOCK,
            image_phys: DRAM + (VMADDR & (BLOCK - 1)),
            boot_phys: DRAM + 0x10_0000,
            scratch_phys: DRAM + 0x11_0000,
            device_tree_phys: DRAM + 0x12_0000,
        }
    }

    #[test]
    fn the_image_is_placed_and_the_boot_structure_describes_dram() {
        let image = image(VMADDR, 0x200, &[0xaa, 0xbb, 0xcc, 0xdd]);
        let mut memory = memory();
        let layout = layout();
        let monitor = load(&mut memory, &image, &[0; 8], &layout).unwrap();
        assert_eq!(monitor.entry, layout.image_phys + 0x200);
        let mut word = [0u8; 8];
        let mut field = |at: u64| {
            memory.read(layout.boot_phys + at, &mut word).unwrap();
            u64::from_le_bytes(word)
        };
        // The image is at its link address in the linear map, so DRAM's first byte is
        // mapped as many bytes below the link address as the image is above DRAM's start.
        let dram_va = VMADDR - (layout.image_phys - DRAM);
        assert_eq!(field(0), 3);
        assert_eq!(field(8), dram_va);
        assert_eq!(field(16), DRAM);
        assert_eq!(field(24), 2 * BLOCK);
        assert_eq!(field(SCRATCH_SLOT), layout.scratch_phys);
        assert_eq!(
            field(DEVICE_TREE_SLOT),
            layout.device_tree_phys - DRAM + dram_va
        );
        assert_eq!(field(DEVICE_TREE_SLOT + 8), 8);
        let mut payload = [0u8; 4];
        memory
            .read(layout.image_phys + 0x200, &mut payload)
            .unwrap();
        assert_eq!(payload, [0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(
            dram_va & (BLOCK - 1),
            0,
            "DRAM's virtual base is block-aligned"
        );
    }

    #[test]
    fn a_physical_base_that_is_not_congruent_with_the_virtual_base_is_refused() {
        let layout = Layout {
            image_phys: DRAM + 0x8000,
            ..layout()
        };
        assert!(matches!(
            load(&mut memory(), &image(VMADDR, 0x200, &[]), &[0; 8], &layout),
            Err(Error::Misaligned { .. })
        ));
    }

    #[test]
    fn dram_that_is_unaligned_or_under_one_block_is_refused() {
        for (base, size) in [
            (DRAM + 0x1000, 2 * BLOCK),
            (DRAM, BLOCK - 0x1000),
            (DRAM, 0),
        ] {
            let layout = Layout {
                dram_base: base,
                dram_size: size,
                ..layout()
            };
            assert!(matches!(
                load(&mut memory(), &image(VMADDR, 0x200, &[]), &[0; 8], &layout),
                Err(Error::Dram { .. })
            ));
        }
    }

    #[test]
    fn a_part_the_monitor_reads_through_its_map_must_lie_in_dram() {
        let layout = Layout {
            boot_phys: DRAM + 2 * BLOCK,
            ..layout()
        };
        assert!(matches!(
            load(&mut memory(), &image(VMADDR, 0x200, &[]), &[0; 8], &layout),
            Err(Error::OutsideDram {
                what: "boot structure",
                ..
            })
        ));
    }

    /// A root node with one property `a` of `length` (raw, flags included) holding `data`,
    /// and one child with no properties.
    fn tree(length: u32, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(1u32.to_le_bytes());
        out.extend(1u32.to_le_bytes());
        let mut name = [0u8; 32];
        name[0] = b'a';
        out.extend(name);
        out.extend(length.to_le_bytes());
        out.extend(data);
        out.resize(out.len().next_multiple_of(4), 0);
        out.extend([0u8; 8]);
        out
    }

    #[test]
    fn placeholder_flags_are_cleared_from_property_lengths_and_nothing_else_changes() {
        let flagged = tree(PLACEHOLDER | 5, b"hello");
        let clean = without_placeholder_flags(&flagged).unwrap();
        assert_eq!(clean, tree(5, b"hello"));
        // The child node after the padded value is still found, so the walk stayed in step.
        assert_eq!(clean.len(), flagged.len());
    }

    #[test]
    fn a_tree_that_ends_inside_a_property_is_refused() {
        let mut cut = tree(PLACEHOLDER | 5, b"hello");
        cut.truncate(40);
        assert!(matches!(
            without_placeholder_flags(&cut),
            Err(Error::DeviceTree(_))
        ));
    }

    /// A node: its properties (name, value) and children, in the Apple flattened format.
    fn adt(properties: &[(&str, &[u8])], children: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend((properties.len() as u32).to_le_bytes());
        out.extend((children.len() as u32).to_le_bytes());
        for (name, value) in properties {
            let mut field = [0u8; 32];
            field[..name.len()].copy_from_slice(name.as_bytes());
            out.extend(field);
            out.extend((value.len() as u32).to_le_bytes());
            out.extend(*value);
            out.resize(out.len().next_multiple_of(4), 0);
        }
        for child in children {
            out.extend(child);
        }
        out
    }

    #[test]
    fn a_region_lands_in_chosen_memory_map_and_other_nodes_are_untouched() {
        let map = |props: &[(&str, &[u8])]| {
            let mut all = vec![("name", &b"memory-map\0"[..])];
            all.extend(props);
            adt(&all, &[])
        };
        let tree = |inner: Vec<u8>| {
            adt(
                &[("name", b"device-tree\0")],
                &[
                    adt(&[("name", b"cpus\0")], &[]),
                    adt(&[("name", b"chosen\0")], &[inner]),
                    adt(&[("name", b"after\0")], &[]),
                ],
            )
        };
        let before = tree(map(&[("RAMDisk", &[1; 16])]));
        let after = add_memory_map_region(&before, "BootKC-rs", 0x1234, 0x56).unwrap();
        // The same tree, built with the property already present, is byte-identical.
        let mut value = Vec::new();
        value.extend(0x1234u64.to_le_bytes());
        value.extend(0x56u64.to_le_bytes());
        assert_eq!(
            after,
            tree(map(&[("RAMDisk", &[1; 16]), ("BootKC-rs", &value)]))
        );
    }

    #[test]
    fn a_tree_without_chosen_memory_map_is_refused() {
        let tree = adt(&[("name", b"device-tree\0")], &[]);
        assert!(matches!(
            add_memory_map_region(&tree, "BootKC-rs", 0, 0),
            Err(Error::DeviceTree(_))
        ));
    }
}
