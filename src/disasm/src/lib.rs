//! Disassembly framework: binary images, linear listing, and control-flow
//! recovery over `volt-isa-aarch64`'s decoder and formatter.
pub mod analysis;
pub mod image;

pub use analysis::{Analysis, Block, Function, Xref, XrefKind};
pub use image::{Image, Segment, Symbol};

use volt_isa_aarch64::decode::decode;
use volt_isa_aarch64::flow::{Flow, flow};
use volt_isa_aarch64::format::{self, Symbols};

/// One decoded instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Insn {
    pub addr: u64,
    pub word: u32,
    pub text: String,
    pub flow: Flow,
}

impl Insn {
    /// Decode `word` at `addr`; targets print absolute, annotated by `symbols`.
    pub fn decode(addr: u64, word: u32, symbols: &dyn Symbols) -> Self {
        // One decode serves both the text and the flow; a word the decoder does not
        // recognise prints raw and is classified by `flow`.
        match decode(word) {
            Ok(instruction) => Self {
                addr,
                word,
                text: instruction.display_with(addr, symbols).to_string(),
                flow: instruction.flow(addr),
            },
            Err(_) => Self {
                addr,
                word,
                text: format::word_with(word, addr, symbols).to_string(),
                flow: flow(word, addr),
            },
        }
    }
}

impl Symbols for Image {
    fn lookup(&self, addr: u64) -> Option<(&str, u64)> {
        self.symbolize(addr)
            .map(|(s, offset)| (s.name.as_str(), offset))
    }
}

impl Image {
    /// Decode `count` instructions starting at `addr`, stopping at the end of
    /// the mapped segment.
    pub fn disassemble(&self, addr: u64, count: usize) -> Vec<Insn> {
        (0..count as u64)
            .map_while(|i| {
                let a = addr + i * 4;
                self.read_u32(a).map(|w| Insn::decode(a, w, self))
            })
            .collect()
    }
}
