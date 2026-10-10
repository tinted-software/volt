//! A loaded binary: mapped segments and symbols.
use object::{Object, ObjectSection, ObjectSymbol, SectionKind, SymbolKind};

#[derive(Clone, Debug)]
pub struct Segment {
    pub name: String,
    pub addr: u64,
    pub data: Vec<u8>,
    pub exec: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub addr: u64,
    pub size: u64,
    pub func: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Image {
    /// Sorted by address.
    pub segments: Vec<Segment>,
    /// Sorted by address.
    pub symbols: Vec<Symbol>,
    pub entry: Option<u64>,
}

impl Image {
    /// Parse an ELF or Mach-O image. Raw files are not detected; use
    /// [`Image::raw`].
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let file = object::File::parse(bytes).map_err(|e| e.to_string())?;
        if file.architecture() != object::Architecture::Aarch64 {
            return Err(format!(
                "unsupported architecture {:?}",
                file.architecture()
            ));
        }
        let mut segments = Vec::new();
        for s in file.sections() {
            let exec = s.kind() == SectionKind::Text;
            if !(exec
                || matches!(
                    s.kind(),
                    SectionKind::Data | SectionKind::ReadOnlyData | SectionKind::ReadOnlyString
                ))
            {
                continue;
            }
            let Ok(data) = s.data() else { continue };
            if data.is_empty() {
                continue;
            }
            segments.push(Segment {
                name: s.name().unwrap_or("").to_string(),
                addr: s.address(),
                data: data.to_vec(),
                exec,
            });
        }
        segments.sort_by_key(|s| s.addr);
        let mut symbols: Vec<Symbol> = file
            .symbols()
            .chain(file.dynamic_symbols())
            .filter(|s| s.address() != 0 && !s.name().unwrap_or("").is_empty())
            .filter(|s| {
                matches!(
                    s.kind(),
                    SymbolKind::Text | SymbolKind::Data | SymbolKind::Unknown
                )
            })
            .map(|s| Symbol {
                name: s.name().unwrap_or("").to_string(),
                addr: s.address(),
                size: s.size(),
                func: s.kind() == SymbolKind::Text,
            })
            .collect();
        symbols.sort_by(|a, b| a.addr.cmp(&b.addr).then_with(|| a.name.cmp(&b.name)));
        symbols.dedup();
        let entry = Some(file.entry()).filter(|&e| e != 0);
        Ok(Self {
            segments,
            symbols,
            entry,
        })
    }

    /// An image of one executable blob at `addr`.
    pub fn raw(addr: u64, code: Vec<u8>) -> Self {
        Self {
            segments: vec![Segment {
                name: "raw".into(),
                addr,
                data: code,
                exec: true,
            }],
            symbols: Vec::new(),
            entry: Some(addr),
        }
    }

    fn segment_at(&self, addr: u64) -> Option<&Segment> {
        let i = self
            .segments
            .partition_point(|s| s.addr <= addr)
            .checked_sub(1)?;
        let s = &self.segments[i];
        (addr - s.addr < s.data.len() as u64).then_some(s)
    }

    pub fn is_exec(&self, addr: u64) -> bool {
        self.segment_at(addr).is_some_and(|s| s.exec)
    }

    pub fn read_u32(&self, addr: u64) -> Option<u32> {
        let s = self.segment_at(addr)?;
        let o = (addr - s.addr) as usize;
        Some(u32::from_le_bytes(s.data.get(o..o + 4)?.try_into().ok()?))
    }

    /// The symbol containing `addr` (or the nearest preceding one when sizes
    /// are unknown), with the offset into it.
    pub fn symbolize(&self, addr: u64) -> Option<(&Symbol, u64)> {
        let i = self
            .symbols
            .partition_point(|s| s.addr <= addr)
            .checked_sub(1)?;
        let s = &self.symbols[i];
        let off = addr - s.addr;
        (s.size == 0 || off < s.size).then_some((s, off))
    }
}
