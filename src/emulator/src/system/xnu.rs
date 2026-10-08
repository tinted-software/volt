//! XNU boot: a Mach-O kernel, a `boot_args` block and an Apple device tree,
//! handed over the way iBoot does on a cold boot (`osfmk/arm64/start.s`
//! `start_first_cpu`): MMU off, `x0` = physical address of `boot_args`.
//!
//! XNU derives its slide from the PC-relative address of `_mh_execute_header`,
//! so the Mach-O header must sit at `physBase` and every segment at
//! `physBase + (vmaddr - virtBase)`. The kernel then builds its own bootstrap
//! page tables and enables the MMU.
//!
//! Memory, low to high: kernel segments, device tree, `boot_args`, up to
//! `topOfKernelData`; everything above is the kernel's to manage.
use super::boot::{Error, Layout};
use crate::memory::GuestMemory;

const MH_MAGIC_64: u32 = 0xfeed_facf;
const CPU_TYPE_ARM64: u32 = 0x0100_000c;
const MH_EXECUTE: u32 = 2;
const LC_UNIXTHREAD: u32 = 0x5;
const LC_SEGMENT_64: u32 = 0x19;
const LC_MAIN: u32 = 0x8000_0028;
const ARM_THREAD_STATE64: u32 = 6;
const HEADER_SIZE: usize = 32;
const SEGMENT_COMMAND_SIZE: usize = 72;
/// `cmd, cmdsize, flavor, count`, then x0-x28, fp, lr, sp, pc.
const THREAD_PC_OFFSET: usize = 16 + 32 * 8;

/// The kernel's page size (`__ARM_16K_PG__`); `boot_args` regions are padded to it.
const PAGE: u64 = 0x4000;
/// One L2 block of the 16 KiB granule. `start_first_cpu` maps whole L2 blocks
/// and keeps only the VA offset within a block, so physical and virtual
/// addresses must agree modulo this size.
const L2_BLOCK: u64 = 32 << 20;

const BOOT_ARGS_REVISION: u16 = 2;
const BOOT_ARGS_VERSION: u16 = 2;
pub const BOOT_LINE_LENGTH: usize = 1024;
// `struct boot_args` in `pexpert/pexpert/arm64/boot.h`.
const BA_VIRT_BASE: usize = 8;
const BA_PHYS_BASE: usize = 16;
const BA_MEM_SIZE: usize = 24;
const BA_TOP_OF_KERNEL_DATA: usize = 32;
const BA_DEVICE_TREE: usize = 96;
const BA_DEVICE_TREE_LENGTH: usize = 104;
const BA_COMMAND_LINE: usize = 108;
const BOOT_ARGS_SIZE: usize = 1152;

/// One `LC_SEGMENT_64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
}

/// What the loader needs from a Mach-O kernel.
#[derive(Debug, PartialEq, Eq)]
pub struct Kernel {
    /// Link-time address of the Mach-O header (`virtBase` in `boot_args`).
    pub virt_base: u64,
    /// Link-time address of the entry point.
    pub entry: u64,
    pub segments: Vec<Segment>,
}

pub fn is_macho(image: &[u8]) -> bool {
    image.len() >= 4 && u32::from_le_bytes(image[..4].try_into().unwrap()) == MH_MAGIC_64
}

fn bad(reason: &'static str) -> Error {
    Error::BadMachO(reason)
}

/// Parse the load commands of an arm64 `MH_EXECUTE` image. The entry point is
/// `LC_MAIN` (relative to the header) or `LC_UNIXTHREAD`'s `pc`.
pub fn parse(image: &[u8]) -> Result<Kernel, Error> {
    if image.len() < HEADER_SIZE {
        return Err(Error::TooSmall);
    }
    if !is_macho(image) {
        return Err(bad("not a 64-bit little-endian Mach-O"));
    }
    let word = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().unwrap());
    let double = |at: usize| u64::from_le_bytes(image[at..at + 8].try_into().unwrap());
    if word(4) != CPU_TYPE_ARM64 {
        return Err(bad("not an arm64 Mach-O"));
    }
    if word(12) != MH_EXECUTE {
        return Err(bad("not an executable Mach-O"));
    }
    let commands_end = HEADER_SIZE
        .checked_add(word(20) as usize)
        .filter(|&end| end <= image.len())
        .ok_or_else(|| bad("load commands run past the end of the file"))?;
    let mut segments = Vec::new();
    let mut main_offset = None;
    let mut thread_pc = None;
    let mut at = HEADER_SIZE;
    for _ in 0..word(16) {
        if at + 8 > commands_end {
            return Err(bad("load command header runs past the load commands"));
        }
        let (command, size) = (word(at), word(at + 4) as usize);
        if size < 8 || size % 4 != 0 || at + size > commands_end {
            return Err(bad("load command size is out of range"));
        }
        match command {
            LC_SEGMENT_64 => {
                if size < SEGMENT_COMMAND_SIZE {
                    return Err(bad("segment command is too short"));
                }
                let segment = Segment {
                    vmaddr: double(at + 24),
                    vmsize: double(at + 32),
                    fileoff: double(at + 40),
                    filesize: double(at + 48),
                };
                let in_file = segment
                    .fileoff
                    .checked_add(segment.filesize)
                    .is_some_and(|end| end <= image.len() as u64);
                if !in_file {
                    return Err(bad("segment data runs past the end of the file"));
                }
                if segment.filesize > segment.vmsize {
                    return Err(bad("segment has more file data than memory"));
                }
                // Prelink placeholders are empty and linked at unrelated addresses.
                if segment.vmsize != 0 {
                    segments.push(segment);
                }
            }
            LC_MAIN if size >= 16 => main_offset = Some(double(at + 8)),
            LC_UNIXTHREAD => {
                if size < THREAD_PC_OFFSET + 8 || word(at + 8) != ARM_THREAD_STATE64 {
                    return Err(bad("LC_UNIXTHREAD is not an ARM_THREAD_STATE64"));
                }
                thread_pc = Some(double(at + THREAD_PC_OFFSET));
            }
            _ => {}
        }
        at += size;
    }
    let virt_base = segments
        .iter()
        .find(|s| s.fileoff == 0 && s.filesize > 0)
        .map(|s| s.vmaddr)
        .ok_or_else(|| bad("no segment maps the Mach-O header"))?;
    let entry = match (main_offset, thread_pc) {
        (Some(offset), _) => virt_base.checked_add(offset),
        (None, Some(pc)) => Some(pc),
        (None, None) => return Err(bad("no LC_MAIN or LC_UNIXTHREAD entry point")),
    }
    .ok_or_else(|| bad("entry point overflows"))?;
    Ok(Kernel {
        virt_base,
        entry,
        segments,
    })
}

fn align(value: u64, to: u64) -> Result<u64, Error> {
    value
        .checked_add(to - 1)
        .map(|v| v & !(to - 1))
        .ok_or(Error::NoRoom)
}

fn boot_args(
    virt_base: u64,
    phys_base: u64,
    mem_size: u64,
    top_of_kernel_data: u64,
    device_tree: u64,
    device_tree_length: u32,
    cmdline: &str,
) -> Result<[u8; BOOT_ARGS_SIZE], Error> {
    // The kernel reads a NUL-terminated string out of a fixed field; cutting a
    // command line short would silently change what the kernel is told.
    if cmdline.len() >= BOOT_LINE_LENGTH {
        return Err(Error::Unsupported("XNU boot-args exceed 1023 bytes"));
    }
    let mut args = [0u8; BOOT_ARGS_SIZE];
    args[0..2].copy_from_slice(&BOOT_ARGS_REVISION.to_le_bytes());
    args[2..4].copy_from_slice(&BOOT_ARGS_VERSION.to_le_bytes());
    let mut put = |at: usize, value: u64| args[at..at + 8].copy_from_slice(&value.to_le_bytes());
    put(BA_VIRT_BASE, virt_base);
    put(BA_PHYS_BASE, phys_base);
    put(BA_MEM_SIZE, mem_size);
    put(BA_TOP_OF_KERNEL_DATA, top_of_kernel_data);
    put(BA_DEVICE_TREE, device_tree);
    args[BA_DEVICE_TREE_LENGTH..BA_DEVICE_TREE_LENGTH + 4]
        .copy_from_slice(&device_tree_length.to_le_bytes());
    args[BA_COMMAND_LINE..BA_COMMAND_LINE + cmdline.len()].copy_from_slice(cmdline.as_bytes());
    Ok(args)
}

/// Load `image` in the first L2 block of RAM, at the same offset within the
/// block as its link-time base. `boot_args` and the device tree follow it.
/// `Layout::entry` is the physical address of the entry point and
/// `Layout::boot_info` the physical address of `boot_args`.
pub fn prepare_boot(
    memory: &mut impl GuestMemory,
    image: &[u8],
    cmdline: &str,
    ram_base: u64,
    ram_size: u64,
) -> Result<Layout, Error> {
    let kernel = parse(image)?;
    let ram_end = ram_base.checked_add(ram_size).ok_or(Error::NoRoom)?;
    // The device tree reserves the last of RAM for the panic log (`chosen/pram`),
    // so the kernel's allocator must stop short of it, as `bootxnu` does.
    let managed_end = ram_end
        .checked_sub(u64::from(super::afdt::PANIC_LOG_SIZE))
        .ok_or(Error::NoRoom)?;
    // `physBase` must agree with `virtBase` modulo an L2 block (see `L2_BLOCK`).
    let phys_base = align(ram_base, L2_BLOCK)?
        .checked_add(kernel.virt_base % L2_BLOCK)
        .ok_or(Error::NoRoom)?;
    let physical = |virt: u64| {
        virt.checked_sub(kernel.virt_base)
            .and_then(|offset| phys_base.checked_add(offset))
            .ok_or_else(|| bad("segment lies below the Mach-O header"))
    };
    let mut kernel_end = phys_base;
    for segment in &kernel.segments {
        let end = physical(segment.vmaddr)?
            .checked_add(segment.vmsize)
            .ok_or(Error::NoRoom)?;
        kernel_end = kernel_end.max(end);
    }
    let entry = physical(kernel.entry)?;
    if entry >= kernel_end {
        return Err(bad("entry point lies outside the kernel"));
    }

    let device_tree = align(kernel_end, PAGE)?;
    let tree = super::afdt::build(ram_base, ram_size, cmdline);
    let args_at = align(
        device_tree
            .checked_add(tree.len() as u64)
            .ok_or(Error::NoRoom)?,
        PAGE,
    )?;
    // XNU lays its physical aperture out from `topOfKernelData` and, in this
    // configuration, lets `physmap_end` leave no room for the random slide of the
    // aperture base: the first segment is shifted by up to one L2 block to match
    // `physBase`'s offset within a block, and only the region below `topOfKernelData`
    // absorbs that shift. Keep it at least one block above `physBase`, whatever the
    // seed makes of the slide.
    let top_of_kernel_data = align(args_at + BOOT_ARGS_SIZE as u64, PAGE)?
        .max(phys_base.checked_add(L2_BLOCK).ok_or(Error::NoRoom)?);
    if top_of_kernel_data > managed_end {
        return Err(Error::NoRoom);
    }

    let args = boot_args(
        kernel.virt_base,
        phys_base,
        managed_end - phys_base,
        top_of_kernel_data,
        // The kernel reads this pointer after it enables the MMU, through the
        // kernel virtual mapping of RAM.
        kernel.virt_base + (device_tree - phys_base),
        tree.len() as u32,
        cmdline,
    )?;
    // RAM starts zeroed, so only the file-backed part of each segment is copied.
    for segment in &kernel.segments {
        let data = &image[segment.fileoff as usize..(segment.fileoff + segment.filesize) as usize];
        memory
            .write(physical(segment.vmaddr)?, data)
            .map_err(Error::Memory)?;
    }
    memory.write(device_tree, &tree).map_err(Error::Memory)?;
    memory.write(args_at, &args).map_err(Error::Memory)?;
    Ok(Layout {
        entry,
        boot_info: args_at,
        initrd: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{PhysicalMemory, Region};

    /// A Mach-O with `__TEXT` (header and code) and `__DATA` (bss only).
    fn image(entry: Entry) -> Vec<u8> {
        let mut image = vec![0u8; 0x1000];
        let put32 = |image: &mut Vec<u8>, at: usize, v: u32| {
            image[at..at + 4].copy_from_slice(&v.to_le_bytes())
        };
        let put64 = |image: &mut Vec<u8>, at: usize, v: u64| {
            image[at..at + 8].copy_from_slice(&v.to_le_bytes())
        };
        put32(&mut image, 0, MH_MAGIC_64);
        put32(&mut image, 4, CPU_TYPE_ARM64);
        put32(&mut image, 12, MH_EXECUTE);
        let mut at = HEADER_SIZE;
        let mut commands = 0;
        let mut segment = |image: &mut Vec<u8>, vmaddr, vmsize, fileoff, filesize| {
            put32(image, at, LC_SEGMENT_64);
            put32(image, at + 4, SEGMENT_COMMAND_SIZE as u32);
            put64(image, at + 24, vmaddr);
            put64(image, at + 32, vmsize);
            put64(image, at + 40, fileoff);
            put64(image, at + 48, filesize);
            at += SEGMENT_COMMAND_SIZE;
            commands += 1;
        };
        segment(&mut image, 0xffff_fe00_0700_4000, 0x4000, 0, 0x1000);
        segment(&mut image, 0xffff_fe00_0700_8000, 0x8000, 0, 0);
        segment(&mut image, 0xffff_fe00_0400_4000, 0, 0, 0);
        match entry {
            Entry::Main(offset) => {
                put32(&mut image, at, LC_MAIN);
                put32(&mut image, at + 4, 24);
                put64(&mut image, at + 8, offset);
                at += 24;
            }
            Entry::Thread(pc) => {
                put32(&mut image, at, LC_UNIXTHREAD);
                put32(&mut image, at + 4, (THREAD_PC_OFFSET + 8) as u32);
                put32(&mut image, at + 8, ARM_THREAD_STATE64);
                put64(&mut image, at + THREAD_PC_OFFSET, pc);
                at += THREAD_PC_OFFSET + 8;
            }
        }
        commands += 1;
        put32(&mut image, 16, commands);
        put32(&mut image, 20, (at - HEADER_SIZE) as u32);
        image[0x800..0x804].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        image
    }
    #[derive(Clone, Copy)]
    enum Entry {
        Main(u64),
        Thread(u64),
    }

    #[test]
    fn lc_main_is_relative_to_the_header_and_lc_unixthread_is_absolute() {
        let main = parse(&image(Entry::Main(0x800))).unwrap();
        let thread = parse(&image(Entry::Thread(0xffff_fe00_0700_4800))).unwrap();
        assert_eq!(main.virt_base, 0xffff_fe00_0700_4000);
        assert_eq!(main.entry, 0xffff_fe00_0700_4800);
        assert_eq!(main, thread);
        assert_eq!(
            main.segments.len(),
            2,
            "the empty prelink placeholder is dropped"
        );
    }

    #[test]
    fn truncated_or_inconsistent_images_are_refused() {
        let good = image(Entry::Main(0x800));
        assert!(matches!(parse(&good[..20]), Err(Error::TooSmall)));
        // The load commands claim more bytes than the file has.
        assert!(matches!(parse(&good[..100]), Err(Error::BadMachO(_))));
        let mut data_past_eof = good.clone();
        data_past_eof[HEADER_SIZE + 48..HEADER_SIZE + 56].copy_from_slice(&0x2000u64.to_le_bytes());
        assert!(matches!(parse(&data_past_eof), Err(Error::BadMachO(_))));
        let mut x86 = good.clone();
        x86[4..8].copy_from_slice(&0x0100_0007u32.to_le_bytes());
        assert!(matches!(parse(&x86), Err(Error::BadMachO(_))));
        let mut zero_size_command = good;
        zero_size_command[HEADER_SIZE + 4..HEADER_SIZE + 8].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(parse(&zero_size_command), Err(Error::BadMachO(_))));
    }

    #[test]
    fn segments_land_at_phys_base_plus_their_link_offset_and_boot_args_describe_them() {
        const RAM: u64 = 0x4000_0000;
        // `virtBase` sits 0x1004000 into its 32 MiB block, so `physBase` does too.
        const PHYS: u64 = RAM + 0x1004000;
        let region = Region::ram(RAM, vec![0; 128 << 20]).unwrap();
        let mut memory = PhysicalMemory::new(vec![region]).unwrap();
        let layout = prepare_boot(
            &mut memory,
            &image(Entry::Main(0x800)),
            "-v",
            RAM,
            128 << 20,
        )
        .unwrap();
        assert_eq!(layout.entry, PHYS + 0x800);
        let mut code = [0u8; 4];
        memory.read(layout.entry, &mut code).unwrap();
        assert_eq!(code, [0xde, 0xad, 0xbe, 0xef]);

        let mut args = [0u8; BOOT_ARGS_SIZE];
        memory.read(layout.boot_info, &mut args).unwrap();
        let field = |at: usize| u64::from_le_bytes(args[at..at + 8].try_into().unwrap());
        assert_eq!(&args[0..4], &[2, 0, 2, 0], "Revision 2, Version 2");
        assert_eq!(field(BA_VIRT_BASE), 0xffff_fe00_0700_4000);
        assert_eq!(field(BA_PHYS_BASE), PHYS);
        assert_eq!(
            field(BA_PHYS_BASE) % L2_BLOCK,
            field(BA_VIRT_BASE) % L2_BLOCK
        );
        // The panic log in the last 512 KiB of RAM (`chosen/pram`) is not the kernel's.
        assert_eq!(
            field(BA_MEM_SIZE),
            (RAM + (128 << 20) - u64::from(crate::system::afdt::PANIC_LOG_SIZE)) - PHYS
        );
        // `__DATA` ends 0xc000 above the header; the tree and boot_args follow.
        let device_tree_pa = field(BA_DEVICE_TREE) - field(BA_VIRT_BASE) + PHYS;
        assert_eq!(device_tree_pa, PHYS + 0xc000);
        assert!(layout.boot_info > device_tree_pa);
        assert_eq!(layout.boot_info % PAGE, 0);
        assert_eq!(field(BA_TOP_OF_KERNEL_DATA) % PAGE, 0);
        assert!(field(BA_TOP_OF_KERNEL_DATA) >= layout.boot_info + BOOT_ARGS_SIZE as u64);
        // XNU's physical aperture has no slack for a random slide in this
        // configuration: its first segment is shifted by up to one L2 block to match
        // `physBase`'s offset, and that shift is absorbed by the kernel region.
        assert!(field(BA_TOP_OF_KERNEL_DATA) - PHYS >= L2_BLOCK);
        assert_eq!(&args[BA_COMMAND_LINE..BA_COMMAND_LINE + 3], b"-v\0");
        // The tree the kernel is told about is the tree in memory.
        let length = u32::from_le_bytes(args[104..108].try_into().unwrap()) as usize;
        let mut tree = vec![0u8; length];
        memory.read(device_tree_pa, &mut tree).unwrap();
        assert_eq!(tree, super::super::afdt::build(RAM, 128 << 20, "-v"));
    }

    #[test]
    fn a_kernel_that_does_not_fit_and_an_overlong_command_line_are_refused() {
        const RAM: u64 = 0x4000_0000;
        let mut memory =
            PhysicalMemory::new(vec![Region::ram(RAM, vec![0; 0x8000]).unwrap()]).unwrap();
        let good = image(Entry::Main(0x800));
        assert!(matches!(
            prepare_boot(&mut memory, &good, "", RAM, 0x8000),
            Err(Error::NoRoom)
        ));
        let mut memory =
            PhysicalMemory::new(vec![Region::ram(RAM, vec![0; 64 << 20]).unwrap()]).unwrap();
        let long = "x".repeat(BOOT_LINE_LENGTH);
        assert!(matches!(
            prepare_boot(&mut memory, &good, &long, RAM, 64 << 20),
            Err(Error::Unsupported(_))
        ));
    }
}
