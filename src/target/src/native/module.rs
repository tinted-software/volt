//! Native linked module images, ported and modified from Vulcan (Apache-2.0).
//! Copyright 2026 LilithSemi.

use super::{Error as NativeError, JittedFunction, sync_icache};
use core::{fmt, ptr::NonNull};
use volt_ir::function::Function;

#[cfg(target_arch = "aarch64")]
use crate::aarch64::isel;
#[cfg(target_arch = "x86_64")]
use crate::x86_64::isel;

#[derive(Debug)]
pub enum Error {
    Native(NativeError),
    Compile(Box<dyn core::error::Error + Send + Sync>),
    UndefinedSymbol(String),
    DuplicateSymbol(String),
    InvalidData,
    InvalidRelocation,
    AddressOverflow,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(e) => e.fmt(f),
            Self::Compile(e) => write!(f, "module compilation: {e}"),
            Self::UndefinedSymbol(name) => write!(f, "undefined module symbol {name}"),
            Self::DuplicateSymbol(name) => write!(f, "duplicate module symbol {name}"),
            Self::InvalidData => f.write_str("invalid module data extent"),
            Self::InvalidRelocation => f.write_str("invalid or out-of-range module relocation"),
            Self::AddressOverflow => f.write_str("module address overflow"),
        }
    }
}
impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Native(e) => Some(e),
            Self::Compile(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}
impl From<NativeError> for Error {
    fn from(value: NativeError) -> Self {
        Self::Native(value)
    }
}

pub struct ModuleFunction<'a> {
    pub name: &'a str,
    pub function: &'a Function,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataKind {
    Rodata,
    Data,
    Bss,
}
impl DataKind {
    fn section(self) -> usize {
        match self {
            Self::Rodata => 0,
            Self::Data => 1,
            Self::Bss => 2,
        }
    }
}

pub struct DataReloc<'a> {
    pub offset: usize,
    pub symbol: &'a str,
}

pub struct ModuleData<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
    pub kind: DataKind,
    pub size: usize,
    pub relocs: &'a [DataReloc<'a>],
}

/// One contiguous image. Code is RX, constants R, mutable globals and BSS RW.
pub struct JittedModule {
    image: JittedFunction,
    functions: Vec<(String, usize)>,
    data: Vec<(String, usize)>,
}
impl JittedModule {
    /// # Safety
    /// T must match the generated function signature, and the returned entry
    /// must not outlive the module. See JittedFunction::entry.
    pub unsafe fn entry<T: Copy>(&self, name: &str) -> Option<T> {
        let offset = self.functions.iter().find(|(symbol, _)| symbol == name)?.1;
        unsafe { self.image.entry_at(offset) }
    }

    /// # Safety
    /// Same requirements as `entry`; offset must identify a function entry.
    pub unsafe fn entry_at<T: Copy>(&self, offset: usize) -> Option<T> {
        unsafe { self.image.entry_at(offset) }
    }

    /// Pointer into a named global. Only Data and Bss objects are writable.
    /// The pointer is valid only while this module remains alive.
    pub fn data_addr(&self, name: &str) -> Option<*mut u8> {
        let offset = self.data.iter().find(|(symbol, _)| symbol == name)?.1;
        Some(unsafe { self.image.address.as_ptr().add(offset) })
    }

    pub fn code(&self) -> &[u8] {
        self.image.code()
    }
}

fn align(value: usize, alignment: usize) -> Result<usize, Error> {
    Ok(value
        .checked_add(alignment - 1)
        .ok_or(Error::AddressOverflow)?
        & !(alignment - 1))
}
fn symbol_offset(
    functions: &[(String, usize)],
    data: &[(String, usize)],
    name: &str,
) -> Result<usize, Error> {
    data.iter()
        .chain(functions.iter())
        .find(|(symbol, _)| symbol == name)
        .map(|(_, offset)| *offset)
        .ok_or_else(|| Error::UndefinedSymbol(name.into()))
}

pub fn jit_module(functions: &[ModuleFunction<'_>]) -> Result<JittedModule, Error> {
    jit_module_data(functions, &[])
}

pub fn jit_module_data(
    functions: &[ModuleFunction<'_>],
    data: &[ModuleData<'_>],
) -> Result<JittedModule, Error> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (functions, data);
        return Err(NativeError::UnsupportedHost.into());
    }
    #[cfg(target_os = "linux")]
    {
        let mut compiled = Vec::with_capacity(functions.len());
        let mut symbols = Vec::with_capacity(functions.len());
        let mut code_len = 0usize;
        for function in functions {
            #[cfg(target_arch = "aarch64")]
            let part = isel::compile_function(function.function, &isel::ModelCaps::default())
                .map_err(|e| Error::Compile(Box::new(e)))?;
            #[cfg(target_arch = "x86_64")]
            let part =
                isel::compile_object(function.function).map_err(|e| Error::Compile(Box::new(e)))?;
            symbols.push((function.name.to_owned(), code_len));
            #[cfg(target_arch = "aarch64")]
            let length = part
                .code
                .len()
                .checked_mul(4)
                .ok_or(Error::AddressOverflow)?;
            #[cfg(target_arch = "x86_64")]
            let length = part.code.len();
            if length == 0 {
                return Err(NativeError::EmptyCode.into());
            }
            code_len = align(
                code_len.checked_add(length).ok_or(Error::AddressOverflow)?,
                16,
            )?;
            compiled.push(part);
        }
        map_module(
            code_len,
            symbols,
            data,
            |memory, base, symbols, data_symbols| {
                #[cfg(target_arch = "x86_64")]
                memory[..code_len].fill(0x90);
                #[cfg(target_arch = "aarch64")]
                for padding in memory[..code_len].chunks_exact_mut(4) {
                    padding.copy_from_slice(&0xd503201fu32.to_le_bytes());
                }
                for (part, (_, offset)) in compiled.iter().zip(symbols.iter()) {
                    #[cfg(target_arch = "x86_64")]
                    memory[*offset..*offset + part.code.len()].copy_from_slice(&part.code);
                    #[cfg(target_arch = "aarch64")]
                    for (index, word) in part.code.iter().enumerate() {
                        memory[*offset + index * 4..*offset + index * 4 + 4]
                            .copy_from_slice(&word.to_le_bytes());
                    }
                }
                for (part, (_, function_offset)) in compiled.iter().zip(symbols.iter()) {
                    for reloc in &part.relocs {
                        let target = symbol_offset(&symbols, &data_symbols, &reloc.symbol)?;
                        #[cfg(target_arch = "x86_64")]
                        {
                            let site = function_offset
                                .checked_add(reloc.offset)
                                .ok_or(Error::AddressOverflow)?;
                            if site.checked_add(4).is_none_or(|end| end > code_len) {
                                return Err(Error::InvalidRelocation);
                            }
                            match reloc.kind {
                                isel::RelocKind::Call | isel::RelocKind::PcRelative => {
                                    if reloc.kind == isel::RelocKind::Call
                                        && !symbols.iter().any(|(name, _)| name == &reloc.symbol)
                                    {
                                        return Err(Error::InvalidRelocation);
                                    }
                                    let delta =
                                        target as i128 + reloc.addend as i128 - site as i128;
                                    let delta = i32::try_from(delta)
                                        .map_err(|_| Error::InvalidRelocation)?;
                                    memory[site..site + 4].copy_from_slice(&delta.to_le_bytes());
                                }
                                isel::RelocKind::Absolute => {
                                    if site.checked_add(8).is_none_or(|end| end > code_len) {
                                        return Err(Error::InvalidRelocation);
                                    }
                                    let value = (base + target) as i128 + reloc.addend as i128;
                                    let value = u64::try_from(value)
                                        .map_err(|_| Error::InvalidRelocation)?;
                                    memory[site..site + 8].copy_from_slice(&value.to_le_bytes());
                                }
                                isel::RelocKind::GotPcRelative => {
                                    return Err(Error::InvalidRelocation);
                                }
                            }
                        }
                        #[cfg(target_arch = "aarch64")]
                        {
                            let site = function_offset
                                .checked_add(
                                    reloc.offset.checked_mul(4).ok_or(Error::AddressOverflow)?,
                                )
                                .ok_or(Error::AddressOverflow)?;
                            if site.checked_add(4).is_none_or(|end| end > code_len) {
                                return Err(Error::InvalidRelocation);
                            }
                            let word =
                                u32::from_le_bytes(memory[site..site + 4].try_into().unwrap());
                            let patched = match reloc.kind {
                                isel::RelocKind::Call => {
                                    if !symbols.iter().any(|(name, _)| name == &reloc.symbol) {
                                        return Err(Error::InvalidRelocation);
                                    }
                                    let delta = target as i128 - site as i128;
                                    if delta % 4 != 0
                                        || !(-(128 << 20)..(128 << 20)).contains(&delta)
                                    {
                                        return Err(Error::InvalidRelocation);
                                    }
                                    0x94000000 | ((delta as i64 >> 2) as u32 & 0x03ffffff)
                                }
                                isel::RelocKind::AdrpPg => {
                                    let delta = (((base + target) & !4095) as i128
                                        - ((base + site) & !4095) as i128)
                                        / 4096;
                                    if !(-(1 << 20)..(1 << 20)).contains(&delta) {
                                        return Err(Error::InvalidRelocation);
                                    }
                                    let value = delta as u32;
                                    (word & !((3 << 29) | (0x7ffff << 5)))
                                        | ((value & 3) << 29)
                                        | (((value >> 2) & 0x7ffff) << 5)
                                }
                                isel::RelocKind::AddPgOff => {
                                    (word & !(0xfff << 10))
                                        | (((base + target) as u32 & 0xfff) << 10)
                                }
                                isel::RelocKind::GotPg | isel::RelocKind::GotLo12 => {
                                    return Err(Error::InvalidRelocation);
                                }
                            };
                            memory[site..site + 4].copy_from_slice(&patched.to_le_bytes());
                        }
                    }
                }
                Ok(())
            },
        )
    }
}

#[cfg(target_os = "linux")]
fn map_module(
    code_len: usize,
    symbols: Vec<(String, usize)>,
    data: &[ModuleData<'_>],
    fill_code: impl FnOnce(
        &mut [u8],
        usize,
        &[(String, usize)],
        &[(String, usize)],
    ) -> Result<(), Error>,
) -> Result<JittedModule, Error> {
    if code_len == 0 {
        return Err(NativeError::EmptyCode.into());
    }
    let mut names = std::collections::BTreeSet::new();
    for name in symbols
        .iter()
        .map(|(name, _)| name.as_str())
        .chain(data.iter().map(|d| d.name))
    {
        if !names.insert(name) {
            return Err(Error::DuplicateSymbol(name.into()));
        }
    }
    let mut extents = [0usize; 3];
    let mut placements = Vec::with_capacity(data.len());
    for object in data {
        if (object.kind != DataKind::Bss && object.size != object.bytes.len())
            || (object.kind == DataKind::Bss && !object.bytes.is_empty())
        {
            return Err(Error::InvalidData);
        }
        let section = object.kind.section();
        let natural = object
            .size
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(8)
            .min(8);
        let offset = align(extents[section], natural)?;
        extents[section] = offset
            .checked_add(object.size)
            .ok_or(Error::AddressOverflow)?;
        for reloc in object.relocs {
            if reloc
                .offset
                .checked_add(8)
                .is_none_or(|end| end > object.size)
            {
                return Err(Error::InvalidRelocation);
            }
        }
        placements.push(offset);
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return Err(NativeError::Mapping(std::io::Error::last_os_error()).into());
    }
    let page = page as usize;
    let mut starts = [0usize; 3];
    let mut mapped_len = align(code_len, page)?;
    for section in 0..3 {
        starts[section] = mapped_len;
        mapped_len = align(
            mapped_len
                .checked_add(extents[section])
                .ok_or(Error::AddressOverflow)?,
            page,
        )?;
    }
    let raw = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            mapped_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        return Err(NativeError::Mapping(std::io::Error::last_os_error()).into());
    }
    let address = NonNull::new(raw.cast::<u8>()).expect("mmap returned null");
    let image = JittedFunction {
        address,
        mapped_len,
        code_len,
    };
    // Anonymous mappings start zeroed, including BSS and padding.
    let memory = unsafe { core::slice::from_raw_parts_mut(address.as_ptr(), mapped_len) };
    let mut data_symbols = Vec::with_capacity(data.len());
    for (object, placement) in data.iter().zip(&placements) {
        let offset = starts[object.kind.section()] + placement;
        memory[offset..offset + object.bytes.len()].copy_from_slice(object.bytes);
        data_symbols.push((object.name.to_owned(), offset));
    }
    let base = address.as_ptr() as usize;
    for (object, (_, offset)) in data.iter().zip(&data_symbols) {
        for reloc in object.relocs {
            let target = base + symbol_offset(&symbols, &data_symbols, reloc.symbol)?;
            let site = offset + reloc.offset;
            memory[site..site + 8].copy_from_slice(&(target as u64).to_le_bytes());
        }
    }
    fill_code(&mut memory[..code_len], base, &symbols, &data_symbols)?;
    // No Rust mutable view survives the permission transition.
    if unsafe {
        libc::mprotect(
            raw,
            align(code_len, page)?,
            libc::PROT_READ | libc::PROT_EXEC,
        )
    } != 0
    {
        return Err(NativeError::Protection(std::io::Error::last_os_error()).into());
    }
    if extents[0] != 0
        && unsafe {
            libc::mprotect(
                address.as_ptr().add(starts[0]).cast(),
                align(extents[0], page)?,
                libc::PROT_READ,
            )
        } != 0
    {
        return Err(NativeError::Protection(std::io::Error::last_os_error()).into());
    }
    sync_icache(image.code());
    Ok(JittedModule {
        image,
        functions: symbols,
        data: data_symbols,
    })
}

/// Map a linked AArch64 module, including its globals and runtime relocations.
/// The resulting entry points may only be executed on an AArch64 host.
#[cfg(all(feature = "aarch64", target_os = "linux"))]
pub fn map_linked_aarch64(linked: &crate::aarch64::link::Linked) -> Result<JittedModule, Error> {
    use crate::aarch64::{isel::RelocKind, link};
    let code_len = linked
        .code
        .len()
        .checked_mul(4)
        .ok_or(Error::AddressOverflow)?;
    let symbols: Vec<_> = linked
        .symbols
        .iter()
        .map(|s| (s.name.clone(), s.offset))
        .collect();
    for (_, offset) in &symbols {
        if *offset >= code_len || offset % 4 != 0 {
            return Err(Error::InvalidRelocation);
        }
    }
    let data_relocs: Vec<Vec<_>> = linked
        .data
        .iter()
        .map(|object| {
            object
                .relocs
                .iter()
                .map(|r| DataReloc {
                    offset: r.off,
                    symbol: r.symbol.as_str(),
                })
                .collect()
        })
        .collect();
    let objects: Vec<_> = linked
        .data
        .iter()
        .zip(&data_relocs)
        .map(|(object, relocs)| ModuleData {
            name: &object.name,
            bytes: &object.bytes,
            kind: match object.kind {
                link::DataKind::Rodata => DataKind::Rodata,
                link::DataKind::Data => DataKind::Data,
                link::DataKind::Bss => DataKind::Bss,
            },
            size: object.size,
            relocs,
        })
        .collect();
    map_module(
        code_len,
        symbols,
        &objects,
        |memory, base, symbols, data| {
            for (chunk, word) in memory.chunks_exact_mut(4).zip(&linked.code) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            for reloc in &linked.relocs {
                if reloc.offset >= linked.code.len() {
                    return Err(Error::InvalidRelocation);
                }
                let site = reloc.offset * 4;
                let target = symbol_offset(symbols, data, &reloc.symbol)?;
                match reloc.kind {
                    RelocKind::AdrpPg => {
                        let delta = (((base + target) & !4095) as i128
                            - ((base + site) & !4095) as i128)
                            / 4096;
                        if !(-(1 << 20)..(1 << 20)).contains(&delta) {
                            return Err(Error::InvalidRelocation);
                        }
                    }
                    RelocKind::AddPgOff => {}
                    RelocKind::Call | RelocKind::GotPg | RelocKind::GotLo12 => {
                        return Err(Error::InvalidRelocation);
                    }
                }
                // SAFETY: memory is an mmap-aligned writable code section, its
                // length is exactly a whole number of A64 words, and Rust host
                // targets supported by this crate are little-endian.
                let words = unsafe {
                    core::slice::from_raw_parts_mut(
                        memory.as_mut_ptr().cast::<u32>(),
                        linked.code.len(),
                    )
                };
                link::apply_global_reloc(words, reloc, base + site, base + target);
            }
            Ok(())
        },
    )
}
