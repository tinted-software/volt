//! AArch64 ESR syndrome decoding (Mirage arm64/esr.zig).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataAbort {
    pub valid: bool,
    pub write: bool,
    pub size: u8,
    pub register: u8,
    pub sign_extend: bool,
    pub sixty_four: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemRegister {
    pub op0: u8,
    pub op1: u8,
    pub crn: u8,
    pub crm: u8,
    pub op2: u8,
    pub transfer: u8,
    pub write: bool,
}

impl SystemRegister {
    pub fn debug(self) -> bool {
        self.op0 == 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exception {
    DataAbort(DataAbort),
    SystemRegister(SystemRegister),
    Hvc(u16),
    Wfi,
    Wfe,
    Unhandled(u8),
}

pub fn class(esr: u64) -> u8 {
    ((esr >> 26) & 63) as u8
}

pub fn decode(esr: u64) -> Exception {
    match class(esr) {
        0x24 => Exception::DataAbort(DataAbort {
            valid: esr & (1 << 24) != 0,
            write: esr & (1 << 6) != 0,
            size: 1 << ((esr >> 22) & 3),
            register: ((esr >> 16) & 31) as u8,
            sign_extend: esr & (1 << 21) != 0,
            sixty_four: esr & (1 << 15) != 0,
        }),
        0x18 => Exception::SystemRegister(SystemRegister {
            op0: ((esr >> 20) & 3) as u8,
            op2: ((esr >> 17) & 7) as u8,
            op1: ((esr >> 14) & 7) as u8,
            crn: ((esr >> 10) & 15) as u8,
            transfer: ((esr >> 5) & 31) as u8,
            crm: ((esr >> 1) & 15) as u8,
            write: esr & 1 == 0,
        }),
        0x16 => Exception::Hvc(esr as u16),
        1 => {
            if esr & 1 == 0 {
                Exception::Wfi
            } else {
                Exception::Wfe
            }
        }
        other => Exception::Unhandled(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abort_width_sign_and_register_are_independent() {
        let syndrome = (0x24 << 26) | (1 << 24) | (2 << 22) | (1 << 21) | (3 << 16) | (1 << 6);
        assert_eq!(
            decode(syndrome),
            Exception::DataAbort(DataAbort {
                valid: true,
                write: true,
                size: 4,
                register: 3,
                sign_extend: true,
                sixty_four: false
            })
        );
    }
}
