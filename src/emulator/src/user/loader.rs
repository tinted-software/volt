// Copyright 2026 LilithSemi
// SPDX-License-Identifier: Apache-2.0
// Modified: Rust port of Mirage lib/mirage-jit/user/Loader.zig.
//! ELF64/AArch64 loading and Linux initial process stack.
//! PT_INTERP is opened at its absolute host path, without an interpreter prefix.
use crate::{Error, user::Space};
use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

const PAGE: u64 = 4096;
const IMAGE_LIMIT: u64 = 1 << 30;
const STACK_SIZE: usize = 8 << 20;
const STACK_TOP: u64 = 0x7fff_fff0_0000;
const STACK_BASE: u64 = STACK_TOP - STACK_SIZE as u64;
const USER_LIMIT: u64 = STACK_BASE - 0x10000;

#[derive(Debug)]
pub struct Entry {
    pub pc: u64,
    pub sp: u64,
    pub brk: u64,
}
struct Image {
    entry: u64,
    end: u64,
    bias: u64,
    phdr: u64,
    phnum: u16,
    interpreter: Option<OsString>,
}
struct Segment {
    address: u64,
    end: u64,
    offset: u64,
    filesz: u64,
    flags: u32,
}

pub fn load(
    space: &mut Space,
    path: &Path,
    args: &[OsString],
    env: &[(OsString, OsString)],
) -> Result<Entry, Error> {
    let previous = space.regions.len();
    let result = (|| {
        let bytes = read_image(path)?;
        let executable = load_image(space, &bytes, 0x4000_0000, true)?;
        let mut pc = executable.entry;
        let mut interpreter_bias = 0;
        if let Some(path) = &executable.interpreter {
            let bytes = read_image(Path::new(path))?;
            let interpreter = load_image(space, &bytes, 0x10_0000_0000, false)?;
            pc = interpreter.entry;
            interpreter_bias = interpreter.bias;
        }
        space.map(STACK_BASE, STACK_SIZE)?;
        let sp = make_stack(
            space,
            path.as_os_str().as_bytes(),
            args,
            env,
            &executable,
            interpreter_bias,
        )?;
        Ok(Entry {
            pc,
            sp,
            brk: round_page(executable.end)?,
        })
    })();
    if result.is_err() {
        space.truncate(previous);
    }
    result
}
fn read_image(path: &Path) -> Result<Vec<u8>, Error> {
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() > IMAGE_LIMIT {
        return Err(Error::ImageTooLarge);
    }
    let mut bytes = Vec::new();
    file.take(IMAGE_LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > IMAGE_LIMIT {
        return Err(Error::ImageTooLarge);
    }
    Ok(bytes)
}
fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}
fn file_span(bytes: &[u8], offset: u64, len: u64) -> Result<&[u8], Error> {
    let end = offset
        .checked_add(len)
        .ok_or(Error::InvalidElf("file span overflow"))?;
    if end > bytes.len() as u64 {
        return Err(Error::InvalidElf("truncated file span"));
    }
    Ok(&bytes[offset as usize..end as usize])
}
pub(crate) fn round_page(value: u64) -> Result<u64, Error> {
    Ok(value.checked_add(PAGE - 1).ok_or(Error::AddressOverflow)? & !(PAGE - 1))
}
fn load_image(
    space: &mut Space,
    bytes: &[u8],
    dynamic_bias: u64,
    allow_interpreter: bool,
) -> Result<Image, Error> {
    if bytes.len() < 64 || &bytes[..4] != b"\x7fELF" {
        return Err(Error::InvalidElf("ELF header"));
    }
    if bytes[4] != 2 || bytes[5] != 1 || bytes[6] != 1 || u32_at(bytes, 20) != 1 {
        return Err(Error::InvalidElf("requires ELF64 little-endian version 1"));
    }
    let kind = u16_at(bytes, 16);
    if u16_at(bytes, 18) != 183 || (kind != 2 && kind != 3) {
        return Err(Error::InvalidElf("requires AArch64 ET_EXEC or ET_DYN"));
    }
    if u16_at(bytes, 52) != 64 || u16_at(bytes, 54) != 56 {
        return Err(Error::InvalidElf("header sizes"));
    }
    let phnum = u16_at(bytes, 56);
    if phnum == 0 || phnum > 1024 {
        return Err(Error::InvalidElf("program header count"));
    }
    let phoff = u64_at(bytes, 32);
    let table_size = u64::from(phnum) * 56;
    let headers = file_span(bytes, phoff, table_size)?;
    let bias = if kind == 3 { dynamic_bias } else { 0 };
    let entry = bias
        .checked_add(u64_at(bytes, 24))
        .ok_or(Error::InvalidElf("entry overflow"))?;
    let mut segments = Vec::new();
    segments
        .try_reserve(phnum as usize)
        .map_err(|_| Error::OutOfMemory)?;
    let mut interpreter = None;
    let mut phdr = None;
    let mut image_end = 0;
    let mut entry_loaded = false;
    for header in headers.chunks_exact(56) {
        let tag = u32_at(header, 0);
        let offset = u64_at(header, 8);
        let vaddr = u64_at(header, 16);
        let filesz = u64_at(header, 32);
        let memsz = u64_at(header, 40);
        if tag == 3 {
            if !allow_interpreter {
                return Err(Error::InvalidElf("interpreter has PT_INTERP"));
            }
            if interpreter.is_some() || !(2..=4096).contains(&filesz) {
                return Err(Error::InvalidElf("PT_INTERP size or duplicate"));
            }
            let name = file_span(bytes, offset, filesz)?;
            if name[0] != b'/' || name[name.len() - 1] != 0 || name[..name.len() - 1].contains(&0) {
                return Err(Error::InvalidElf(
                    "interpreter must be an absolute NUL-terminated path",
                ));
            }
            interpreter = Some(OsStr::from_bytes(&name[..name.len() - 1]).to_os_string());
            continue;
        }
        if tag != 1 {
            continue;
        }
        let flags = u32_at(header, 4);
        let alignment = u64_at(header, 48);
        if filesz > memsz || memsz > IMAGE_LIMIT || flags & !7 != 0 {
            return Err(Error::InvalidElf("segment sizes or flags"));
        }
        if filesz != 0 {
            file_span(bytes, offset, filesz)?;
        }
        if alignment > 1
            && (!alignment.is_power_of_two() || vaddr & (alignment - 1) != offset & (alignment - 1))
        {
            return Err(Error::InvalidElf("segment alignment"));
        }
        if vaddr & (PAGE - 1) != offset & (PAGE - 1) {
            return Err(Error::InvalidElf("segment page alignment"));
        }
        let address = bias
            .checked_add(vaddr)
            .ok_or(Error::InvalidElf("segment address overflow"))?;
        let end = address
            .checked_add(memsz)
            .ok_or(Error::InvalidElf("segment end overflow"))?;
        if end > USER_LIMIT {
            return Err(Error::InvalidElf("segment above user limit"));
        }
        if memsz == 0 {
            continue;
        }
        if entry >= address && entry < end && flags & 1 != 0 {
            entry_loaded = true;
        }
        image_end = image_end.max(end);
        let file_end = offset
            .checked_add(filesz)
            .ok_or(Error::InvalidElf("segment file end overflow"))?;
        let table_end = phoff
            .checked_add(table_size)
            .ok_or(Error::InvalidElf("program header overflow"))?;
        if phoff >= offset && table_end <= file_end {
            phdr = Some(address + (phoff - offset));
        }
        segments.push(Segment {
            address,
            end,
            offset,
            filesz,
            flags,
        });
    }
    if !entry_loaded || entry & 3 != 0 || segments.is_empty() {
        return Err(Error::InvalidElf("entry not in executable segment"));
    }
    segments.sort_unstable_by_key(|s| s.address);
    let mut mapping_start = 0;
    let mut mapping_end = 0;
    let mut total = 0;
    for (i, segment) in segments.iter().enumerate() {
        if i != 0 && segments[i - 1].end > segment.address {
            return Err(Error::InvalidElf("overlapping segments"));
        }
        let start = segment.address & !(PAGE - 1);
        let end = round_page(segment.end)?;
        if i == 0 {
            mapping_start = start;
            mapping_end = end;
        } else if start <= mapping_end {
            mapping_end = mapping_end.max(end);
        } else {
            total = map_image_span(space, mapping_start, mapping_end, total)?;
            mapping_start = start;
            mapping_end = end;
        }
    }
    map_image_span(space, mapping_start, mapping_end, total)?;
    for segment in &segments {
        if segment.filesz != 0 {
            space.initialize(
                segment.address,
                file_span(bytes, segment.offset, segment.filesz)?,
            )?;
        }
    }
    for segment in &segments {
        let start = segment.address & !(PAGE - 1);
        let end = round_page(segment.end)?;
        let permission =
            (((segment.flags & 4) >> 2) | (segment.flags & 2) | ((segment.flags & 1) << 2)) as u8;
        space.protect(start, (end - start) as usize, permission)?;
    }
    let phdr = match phdr {
        Some(address) => address,
        None => map_headers(space, headers)?,
    };
    Ok(Image {
        entry,
        end: image_end,
        bias,
        phdr,
        phnum,
        interpreter,
    })
}
fn map_image_span(space: &mut Space, start: u64, end: u64, previous: u64) -> Result<u64, Error> {
    let total = previous
        .checked_add(end - start)
        .ok_or(Error::ImageTooLarge)?;
    if total > IMAGE_LIMIT {
        return Err(Error::ImageTooLarge);
    }
    space.map(start, (end - start) as usize)?;
    Ok(total)
}
fn map_headers(space: &mut Space, headers: &[u8]) -> Result<u64, Error> {
    let size = round_page(headers.len() as u64)?;
    let mut address = 0x6000_0000_0000u64;
    loop {
        if let Some(region) = space
            .regions
            .iter()
            .find(|r| address < r.end() && address + size > r.address)
        {
            address = (region.address & !(PAGE - 1))
                .checked_sub(size)
                .ok_or(Error::AddressOverflow)?;
        } else {
            break;
        }
    }
    space.map(address, size as usize)?;
    space.initialize(address, headers)?;
    space.protect(address, size as usize, 1)?;
    Ok(address)
}
struct Stack<'a> {
    space: &'a mut Space,
    at: u64,
}
impl Stack<'_> {
    fn reserve(&mut self, len: usize) -> Result<u64, Error> {
        if len as u64 > self.at - STACK_BASE {
            return Err(Error::ArgumentsTooLarge);
        }
        self.at -= len as u64;
        Ok(self.at)
    }
    fn string(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        if bytes.contains(&0) {
            return Err(Error::InvalidArgument);
        }
        let size = bytes.len().checked_add(1).ok_or(Error::ArgumentsTooLarge)?;
        let at = self.reserve(size)?;
        let destination = self.space.slice_mut(at, size)?;
        destination[..bytes.len()].copy_from_slice(bytes);
        destination[bytes.len()] = 0;
        Ok(at)
    }
}
fn make_stack(
    space: &mut Space,
    path: &[u8],
    args: &[OsString],
    env: &[(OsString, OsString)],
    image: &Image,
    interpreter_bias: u64,
) -> Result<u64, Error> {
    let count = args
        .len()
        .checked_add(env.len())
        .ok_or(Error::ArgumentsTooLarge)?;
    if count > STACK_SIZE / 8 {
        return Err(Error::ArgumentsTooLarge);
    }
    let mut argv = Vec::new();
    let mut envp = Vec::new();
    argv.try_reserve(args.len())
        .map_err(|_| Error::OutOfMemory)?;
    envp.try_reserve(env.len())
        .map_err(|_| Error::OutOfMemory)?;
    let mut stack = Stack {
        space,
        at: STACK_TOP,
    };
    let execfn = stack.string(path)?;
    for (key, value) in env {
        let key = key.as_bytes();
        let value = value.as_bytes();
        if key.contains(&0) || value.contains(&0) {
            return Err(Error::InvalidArgument);
        }
        let len = key
            .len()
            .checked_add(value.len())
            .and_then(|n| n.checked_add(2))
            .ok_or(Error::ArgumentsTooLarge)?;
        let at = stack.reserve(len)?;
        let destination = stack.space.slice_mut(at, len)?;
        destination[..key.len()].copy_from_slice(key);
        destination[key.len()] = b'=';
        destination[key.len() + 1..len - 1].copy_from_slice(value);
        destination[len - 1] = 0;
        envp.push(at);
    }
    for arg in args {
        argv.push(stack.string(arg.as_bytes())?);
    }
    let mut random = [0; 16];
    secure_random(&mut random)?;
    let random_address = stack.reserve(random.len())?;
    stack.space.initialize(random_address, &random)?;
    let aux = [
        (3, image.phdr),
        (4, 56),
        (5, u64::from(image.phnum)),
        (6, PAGE),
        (7, interpreter_bias),
        (9, image.entry),
        (11, unsafe { libc::getuid() } as u64),
        (12, unsafe { libc::geteuid() } as u64),
        (13, unsafe { libc::getgid() } as u64),
        (14, unsafe { libc::getegid() } as u64),
        (16, 0),
        (23, 0),
        (25, random_address),
        (26, 0),
        (31, execfn),
        (0, 0),
    ];
    let words = count
        .checked_add(3 + aux.len() * 2)
        .ok_or(Error::ArgumentsTooLarge)?;
    let table_size = words.checked_mul(8).ok_or(Error::ArgumentsTooLarge)?;
    stack.reserve(table_size)?;
    stack.at &= !15;
    if stack.at < STACK_BASE {
        return Err(Error::ArgumentsTooLarge);
    }
    let table = stack.space.slice_mut(stack.at, table_size)?;
    let mut offset = 0;
    let mut word = |value: u64| {
        table[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        offset += 8;
    };
    word(args.len() as u64);
    for pointer in argv {
        word(pointer);
    }
    word(0);
    for pointer in envp {
        word(pointer);
    }
    word(0);
    for (tag, value) in aux {
        word(tag);
        word(value);
    }
    Ok(stack.at)
}
fn secure_random(bytes: &mut [u8]) -> Result<(), Error> {
    let mut offset = 0;
    while offset < bytes.len() {
        let n = unsafe {
            libc::getrandom(bytes[offset..].as_mut_ptr().cast(), bytes.len() - offset, 0)
        };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err.into());
        }
        if n == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        offset += n as usize;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::memory::GuestMemory;

    fn image(kind: u16, address: u64) -> Vec<u8> {
        let mut bytes = vec![0; 256];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4..7].copy_from_slice(&[2, 1, 1]);
        bytes[16..18].copy_from_slice(&kind.to_le_bytes());
        bytes[18..20].copy_from_slice(&183u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&(address + 128).to_le_bytes());
        bytes[32..40].copy_from_slice(&64u64.to_le_bytes());
        bytes[52..54].copy_from_slice(&64u16.to_le_bytes());
        bytes[54..56].copy_from_slice(&56u16.to_le_bytes());
        bytes[56..58].copy_from_slice(&1u16.to_le_bytes());
        bytes[64..68].copy_from_slice(&1u32.to_le_bytes());
        bytes[68..72].copy_from_slice(&5u32.to_le_bytes());
        bytes[80..88].copy_from_slice(&address.to_le_bytes());
        bytes[96..104].copy_from_slice(&256u64.to_le_bytes());
        bytes[104..112].copy_from_slice(&4096u64.to_le_bytes());
        bytes[112..120].copy_from_slice(&4096u64.to_le_bytes());
        bytes
    }
    fn word(space: &Space, address: u64) -> u64 {
        let mut bytes = [0; 8];
        space.read(address, &mut bytes).unwrap();
        u64::from_le_bytes(bytes)
    }
    #[test]
    fn executable_dynamic_bss_and_stack_bytes() {
        for (kind, address, bias) in [(2, 0x100000, 0), (3, 0, 0x40000000)] {
            let mut space = Space::new();
            let executable =
                load_image(&mut space, &image(kind, address), 0x40000000, true).unwrap();
            assert_eq!(executable.entry, address + bias + 128);
            assert_eq!(executable.phdr, address + bias + 64);
            assert_eq!(executable.bias, bias);
            let mut bss = [1; 16];
            space.read(address + bias + 256, &mut bss).unwrap();
            assert_eq!(bss, [0; 16]);
            assert!(space.write(address + bias + 256, &[1]).is_err());
            space.check(executable.entry, 4, 4).unwrap();
            space.map(STACK_BASE, STACK_SIZE).unwrap();
            let args = [
                OsString::from("guest"),
                OsStr::from_bytes(b"arg\xff").to_os_string(),
            ];
            let env = [(
                OsString::from("KEY"),
                OsStr::from_bytes(b"value\xfe").to_os_string(),
            )];
            let sp =
                make_stack(&mut space, b"guest", &args, &env, &executable, 0x1000000000).unwrap();
            assert_eq!(sp & 15, 0);
            assert_eq!(word(&space, sp), 2);
            let mut argument = [0; 5];
            space.read(word(&space, sp + 16), &mut argument).unwrap();
            assert_eq!(&argument, b"arg\xff\0");
            assert_eq!(word(&space, sp + 24), 0);
            let mut environment = [0; 11];
            space.read(word(&space, sp + 32), &mut environment).unwrap();
            assert_eq!(&environment, b"KEY=value\xfe\0");
            assert_eq!(word(&space, sp + 40), 0);
            assert_eq!(word(&space, sp + 48), 3);
            assert_eq!(word(&space, sp + 56), executable.phdr);
            assert_eq!(word(&space, sp + 48 + 4 * 16), 7);
            assert_eq!(word(&space, sp + 56 + 4 * 16), 0x1000000000);
        }
    }
    #[test]
    fn malformed_images_and_nul_arguments_are_rejected() {
        let mut space = Space::new();
        let mut bytes = image(2, 0x100000);
        bytes[96..104].copy_from_slice(&4097u64.to_le_bytes());
        assert!(load_image(&mut space, &bytes, 0, true).is_err());
        let executable = load_image(&mut space, &image(2, 0x100000), 0, true).unwrap();
        space.map(STACK_BASE, STACK_SIZE).unwrap();
        let args = [OsStr::from_bytes(b"bad\0argument").to_os_string()];
        assert!(matches!(
            make_stack(&mut space, b"guest", &args, &[], &executable, 0),
            Err(Error::InvalidArgument)
        ));
    }
}

#[cfg(test)]
mod header_tests {
    use alloc::vec;

    use super::*;
    use crate::memory::GuestMemory;

    fn header_image() -> Vec<u8> {
        let mut bytes = vec![0; 512];
        bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        bytes[16..18].copy_from_slice(&2u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&183u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&0x100100u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&64u64.to_le_bytes());
        bytes[52..54].copy_from_slice(&64u16.to_le_bytes());
        bytes[54..56].copy_from_slice(&56u16.to_le_bytes());
        bytes[56..58].copy_from_slice(&1u16.to_le_bytes());
        bytes[64..68].copy_from_slice(&1u32.to_le_bytes());
        bytes[68..72].copy_from_slice(&5u32.to_le_bytes());
        bytes[72..80].copy_from_slice(&256u64.to_le_bytes());
        bytes[80..88].copy_from_slice(&0x100100u64.to_le_bytes());
        bytes[96..104].copy_from_slice(&256u64.to_le_bytes());
        bytes[104..112].copy_from_slice(&4096u64.to_le_bytes());
        bytes[112..120].copy_from_slice(&4096u64.to_le_bytes());
        bytes
    }
    #[test]
    fn unmapped_program_headers_get_read_only_backing() {
        let bytes = header_image();
        let mut space = Space::new();
        let image = load_image(&mut space, &bytes, 0, true).unwrap();
        let mut headers = [0; 56];
        space.read(image.phdr, &mut headers).unwrap();
        assert_eq!(&headers, &bytes[64..120]);
        assert!(space.write(image.phdr, &[0]).is_err());
    }
    #[test]
    fn interpreter_paths_are_absolute_and_not_recursive() {
        let mut bytes = header_image();
        bytes[56..58].copy_from_slice(&2u16.to_le_bytes());
        bytes[120..124].copy_from_slice(&3u32.to_le_bytes());
        bytes[128..136].copy_from_slice(&300u64.to_le_bytes());
        bytes[152..160].copy_from_slice(&13u64.to_le_bytes());
        bytes[300..313].copy_from_slice(b"/host/loader\0");
        let image = load_image(&mut Space::new(), &bytes, 0, true).unwrap();
        assert_eq!(image.interpreter.unwrap(), OsString::from("/host/loader"));
        assert!(load_image(&mut Space::new(), &bytes, 0, false).is_err());
        bytes[300] = b'.';
        assert!(load_image(&mut Space::new(), &bytes, 0, true).is_err());
    }
}
