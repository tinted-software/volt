//! The little-endian AArch64 Linux Image boot protocol (Mirage arm64/boot.zig).
use crate::memory::{GuestMemory, MemoryError};

pub const MAGIC: u32 = 0x644d5241;
pub const ALIGNMENT: u64 = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageSize {
    Unspecified,
    Page4K,
    Page16K,
    Page64K,
}
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub code0: u32,
    pub code1: u32,
    pub text_offset: u64,
    pub image_size: u64,
    pub flags: u64,
}
impl Header {
    pub fn page_size(self) -> PageSize {
        match (self.flags >> 1) & 3 {
            0 => PageSize::Unspecified,
            1 => PageSize::Page4K,
            2 => PageSize::Page16K,
            _ => PageSize::Page64K,
        }
    }
    pub fn place_anywhere(self) -> bool {
        self.flags & 8 != 0
    }
}
#[derive(Debug)]
pub enum Error {
    TooSmall,
    NotAnImage,
    BigEndianKernel,
    NoRoom,
    /// A Mach-O kernel that is truncated or inconsistent.
    BadMachO(&'static str),
    /// A boot option the selected kernel format cannot honour.
    Unsupported(&'static str),
    Memory(MemoryError),
    /// The device tree the caller supplied cannot be patched.
    DeviceTree(super::dtb::Error),
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl core::error::Error for Error {}

pub fn parse(image: &[u8]) -> Result<Header, Error> {
    if image.len() < 64 {
        return Err(Error::TooSmall);
    }
    let word = |at| u32::from_le_bytes(image[at..at + 4].try_into().unwrap());
    let double = |at| u64::from_le_bytes(image[at..at + 8].try_into().unwrap());
    if word(56) != MAGIC {
        return Err(Error::NotAnImage);
    }
    let header = Header {
        code0: word(0),
        code1: word(4),
        text_offset: double(8),
        image_size: double(16),
        flags: double(24),
    };
    if header.flags & 1 != 0 {
        return Err(Error::BigEndianKernel);
    }
    Ok(header)
}
fn align(value: u64) -> Result<u64, Error> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|v| v & !(ALIGNMENT - 1))
        .ok_or(Error::NoRoom)
}
pub fn load_address(ram_base: u64, header: Header) -> Result<u64, Error> {
    align(ram_base)?
        .checked_add(header.text_offset)
        .ok_or(Error::NoRoom)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub entry: u64,
    /// Guest-physical address handed to the kernel in `x0`: the device tree
    /// blob for a Linux Image, the `boot_args` block for XNU.
    pub boot_info: u64,
    pub initrd: Option<(u64, u64)>,
}
pub fn prepare(
    memory: &mut impl GuestMemory,
    image: &[u8],
    cmdline: &str,
    ram_base: u64,
    ram_size: u64,
) -> Result<Layout, Error> {
    prepare_boot(memory, image, None, cmdline, ram_base, ram_size, 1, false)
}

pub fn prepare_smp(
    memory: &mut impl GuestMemory,
    image: &[u8],
    cmdline: &str,
    ram_base: u64,
    ram_size: u64,
    cpus: u32,
) -> Result<Layout, Error> {
    prepare_boot(
        memory, image, None, cmdline, ram_base, ram_size, cpus, false,
    )
}

pub fn prepare_boot(
    memory: &mut impl GuestMemory,
    image: &[u8],
    initrd: Option<&[u8]>,
    cmdline: &str,
    ram_base: u64,
    ram_size: u64,
    cpus: u32,
    has_virtio: bool,
) -> Result<Layout, Error> {
    let header = parse(image)?;
    let entry = load_address(ram_base, header)?;
    let occupied = header.image_size.max(image.len() as u64);
    let mut above = entry.checked_add(occupied).ok_or(Error::NoRoom)?;

    let initrd_info = if let Some(initrd_data) = initrd {
        let start = align(above)?;
        let end = start
            .checked_add(initrd_data.len() as u64)
            .ok_or(Error::NoRoom)?;
        above = end;
        Some((start, end))
    } else {
        None
    };

    let device_tree = align(above)?;
    let tree = super::fdt::build_smp(ram_base, ram_size, cmdline, cpus, initrd_info, has_virtio);
    let end = ram_base.checked_add(ram_size).ok_or(Error::NoRoom)?;
    if above > end
        || device_tree
            .checked_add(tree.len() as u64)
            .ok_or(Error::NoRoom)?
            > end
    {
        return Err(Error::NoRoom);
    }
    memory.write(entry, image).map_err(Error::Memory)?;
    if let (Some(initrd_data), Some((start, _))) = (initrd, initrd_info) {
        memory.write(start, initrd_data).map_err(Error::Memory)?;
    }
    memory.write(device_tree, &tree).map_err(Error::Memory)?;
    Ok(Layout {
        entry,
        boot_info: device_tree,
        initrd: initrd_info,
    })
}

/// Place a Linux `Image` at its text offset above `ram_base`, with `tree` (a device
/// tree the caller already built or patched) 2 MiB-aligned after it. The returned
/// layout's `boot_info` is the tree's address, which the kernel receives in `x0`.
pub fn place_with_tree(
    memory: &mut impl GuestMemory,
    image: &[u8],
    tree: &[u8],
    ram_base: u64,
    ram_size: u64,
) -> Result<Layout, Error> {
    let header = parse(image)?;
    let entry = load_address(ram_base, header)?;
    let occupied = header.image_size.max(image.len() as u64);
    let tree_at = align(entry.checked_add(occupied).ok_or(Error::NoRoom)?)?;
    let end = ram_base.checked_add(ram_size).ok_or(Error::NoRoom)?;
    if tree_at
        .checked_add(tree.len() as u64)
        .is_none_or(|tree_end| tree_end > end)
    {
        return Err(Error::NoRoom);
    }
    memory.write(entry, image).map_err(Error::Memory)?;
    memory.write(tree_at, tree).map_err(Error::Memory)?;
    Ok(Layout {
        entry,
        boot_info: tree_at,
        initrd: None,
    })
}

/// Where a firmware image is mapped: flash at address 0, where the core resets.
pub const FIRMWARE_BASE: u64 = 0;

/// Place a device tree at the base of RAM and enter the firmware at [`FIRMWARE_BASE`] with
/// `x0` zero. That is what QEMU's `-bios` does: its reset path sets only the PC, and the
/// tree goes to RAM base for a firmware that looks there. Without a tree in `x0` the
/// tinted-boot firmware assumes its built-in platform description, which places RAM up to
/// 4 GiB; passing the tree there would have it reserve memory it already owns.
pub fn prepare_firmware(
    memory: &mut impl GuestMemory,
    ram_base: u64,
    ram_size: u64,
) -> Result<Layout, Error> {
    let tree = super::fdt::build(ram_base, ram_size, "");
    let end = ram_base.checked_add(ram_size).ok_or(Error::NoRoom)?;
    if ram_base
        .checked_add(tree.len() as u64)
        .is_none_or(|tree_end| tree_end > end)
    {
        return Err(Error::NoRoom);
    }
    memory.write(ram_base, &tree).map_err(Error::Memory)?;
    Ok(Layout {
        entry: FIRMWARE_BASE,
        boot_info: 0,
        initrd: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_arm64_header_fields_and_placement() {
        let header = [
            0x4d, 0x5a, 0x40, 0xfa, 0x27, 0x04, 0xc6, 0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
            0, 0, 0, 0x0e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0x41, 0x52, 0x4d, 0x64, 0x40, 0, 0, 0,
        ];
        let parsed = parse(&header).unwrap();
        assert_eq!(parsed.page_size(), PageSize::Page64K);
        assert_eq!(load_address(0x40001000, parsed).unwrap(), 0x40200000);
        assert!(parsed.place_anywhere());
        assert!(matches!(parse(&header[..32]), Err(Error::TooSmall)));
    }
    #[test]
    fn prepare_boot_places_initrd_correctly() {
        use crate::memory::{PhysicalMemory, Region};
        let mut header = [0u8; 64];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes()); // magic ARM\x64
        header[16..24].copy_from_slice(&0x200000u64.to_le_bytes()); // image_size 2MiB
        header[56..60].copy_from_slice(&MAGIC.to_le_bytes());

        let ram_base = 0x4000_0000;
        let ram_size = 32 << 20;
        let region = Region::ram(ram_base, vec![0; ram_size as usize]).unwrap();
        let mut mem = PhysicalMemory::new(vec![region]).unwrap();
        let initrd_data = b"mock cpio payload";

        let layout = prepare_boot(
            &mut mem,
            &header,
            Some(initrd_data),
            "console=ttyAMA0",
            ram_base,
            ram_size,
            1,
            false,
        )
        .unwrap();

        assert!(layout.initrd.is_some());
        let (initrd_start, initrd_end) = layout.initrd.unwrap();
        assert!(initrd_start >= layout.entry + 0x200000);
        assert_eq!(initrd_end, initrd_start + initrd_data.len() as u64);
        assert!(layout.boot_info >= initrd_end);

        let mut read_back = vec![0u8; initrd_data.len()];
        mem.read(initrd_start, &mut read_back).unwrap();
        assert_eq!(&read_back, initrd_data);
    }
}
