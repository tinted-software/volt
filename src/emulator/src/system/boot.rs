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
    Memory(MemoryError),
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
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub entry: u64,
    pub device_tree: u64,
}
pub fn prepare(
    memory: &mut impl GuestMemory,
    image: &[u8],
    cmdline: &str,
    ram_base: u64,
    ram_size: u64,
) -> Result<Layout, Error> {
    let header = parse(image)?;
    let entry = load_address(ram_base, header)?;
    let occupied = header.image_size.max(image.len() as u64);
    let above = entry.checked_add(occupied).ok_or(Error::NoRoom)?;
    let device_tree = align(above)?;
    let tree = super::fdt::build(ram_base, ram_size, cmdline);
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
    memory.write(device_tree, &tree).map_err(Error::Memory)?;
    Ok(Layout { entry, device_tree })
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
}
