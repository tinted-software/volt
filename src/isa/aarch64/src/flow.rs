//! Control-flow classification of A64 instructions, for analysis.
//!
//! [`Instruction::flow`] answers the CFG question: where can execution go
//! next. It is architectural. Whether a particular translator cuts its blocks
//! at loads, stores or system instructions is that translator's policy and is
//! not decided here.
use crate::decode::{IndirectKind, Instruction, decode, is_undefined};

/// Where execution can go after an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Continues at `pc + 4`.
    Next,
    /// Unconditional direct branch.
    Jump(u64),
    /// Conditional direct branch: taken target, otherwise `pc + 4`.
    CondJump(u64),
    /// Direct call (`bl`); returns to `pc + 4`.
    Call(u64),
    /// Indirect call (`blr`); returns to `pc + 4`.
    CallIndirect,
    /// Indirect jump (`br`); no known successor.
    JumpIndirect,
    /// `ret`, `eret`.
    Return,
    /// Does not continue (`brk`, permanently undefined words).
    Stop,
    /// The word is not an instruction this crate decodes, so where it goes is
    /// unknown. Analysis that must keep walking treats it as falling through
    /// (see [`Flow::falls_through`]); it never ends a block.
    Unrecognized,
}

impl Flow {
    /// Whether the instruction after this one (`pc + 4`) is a successor.
    pub fn falls_through(self) -> bool {
        matches!(
            self,
            Self::Next
                | Self::CondJump(_)
                | Self::Call(_)
                | Self::CallIndirect
                | Self::Unrecognized
        )
    }
    /// The statically known branch or call target.
    pub fn target(self) -> Option<u64> {
        match self {
            Self::Jump(t) | Self::CondJump(t) | Self::Call(t) => Some(t),
            _ => None,
        }
    }
    /// Whether this instruction ends a basic block.
    pub fn ends_block(self) -> bool {
        !matches!(self, Self::Next | Self::Unrecognized)
    }
}

impl Instruction {
    /// Where execution can go after this instruction when it is located at `pc`.
    pub fn flow(&self, pc: u64) -> Flow {
        let rel = |off: i64| pc.wrapping_add(off as u64);
        match self {
            Self::B(o) => Flow::Jump(rel(*o)),
            Self::BCond(b) => Flow::CondJump(rel(b.offset)),
            Self::TestBranch(t) => Flow::CondJump(rel(t.offset)),
            Self::Call(c) if c.link => Flow::Call(rel(c.target)),
            Self::Call(c) => Flow::Jump(rel(c.target)),
            Self::Indirect(i) => match i.kind {
                IndirectKind::Jump => Flow::JumpIndirect,
                IndirectKind::Call => Flow::CallIndirect,
                IndirectKind::Return => Flow::Return,
            },
            Self::Eret => Flow::Return,
            Self::Brk(_) => Flow::Stop,
            _ => Flow::Next,
        }
    }
}

/// Classify the instruction `word` located at `pc`. Permanently undefined
/// words are [`Flow::Stop`]; words the decoder does not recognise are
/// [`Flow::Unrecognized`].
pub fn flow(word: u32, pc: u64) -> Flow {
    match decode(word) {
        Ok(instruction) => instruction.flow(pc),
        Err(_) if is_undefined(word) => Flow::Stop,
        Err(_) => Flow::Unrecognized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies() {
        assert_eq!(flow(0xd65f_03c0, 0x100), Flow::Return);
        assert_eq!(flow(0xd61f_0200, 0x100), Flow::JumpIndirect); // br x16
        assert_eq!(flow(0xd63f_0100, 0x100), Flow::CallIndirect); // blr x8
        assert_eq!(flow(0x1400_0004, 0x100), Flow::Jump(0x110)); // b .+16
        assert_eq!(flow(0x97ff_ffff, 0x100), Flow::Call(0xfc)); // bl .-4
        assert_eq!(flow(0x3400_00e8, 0x100), Flow::CondJump(0x11c)); // cbz w8, .+28
        assert_eq!(flow(0xd503_201f, 0x100), Flow::Next); // nop
        assert_eq!(flow(0xd420_0020, 0x100), Flow::Stop); // brk #1
        assert_eq!(flow(0xd400_0021, 0x100), Flow::Next); // svc #1
        assert_eq!(flow(0xe7ff_deff, 0x100), Flow::Stop); // udf: permanently undefined
    }

    #[test]
    fn ret_and_br_stay_distinct_in_the_decoded_instruction() {
        use crate::decode::Indirect;
        let ret = decode(0xd65f_03c0).unwrap();
        let br = decode(0xd61f_0200).unwrap();
        assert_eq!(
            ret,
            Instruction::Indirect(Indirect {
                rn: 30,
                kind: IndirectKind::Return
            })
        );
        assert_eq!(
            br,
            Instruction::Indirect(Indirect {
                rn: 16,
                kind: IndirectKind::Jump
            })
        );
    }

    #[test]
    fn unrecognized_words_neither_end_a_block_nor_stop_a_walk() {
        assert!(Flow::Unrecognized.falls_through());
        assert!(!Flow::Unrecognized.ends_block());
        assert!(!Flow::Stop.falls_through());
    }
}
