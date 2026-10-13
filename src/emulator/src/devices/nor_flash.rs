//! Volatile variable flash: the second QEMU virt bank, not the firmware ROM.
//!
//! The bank is two x16 Intel-command-set chips on a little-endian x32 bus.
//! Operations complete synchronously; storage and the write buffer are allocated
//! only when the device is constructed. Command decoding follows QEMU's bank-wide
//! low-byte commands, with status and identification repeated in both x16 lanes.

use alloc::vec;
use alloc::vec::Vec;

use super::bus::Device;

pub const BASE: u64 = 0x0400_0000;
pub const LEN: u64 = 64 * 1024 * 1024;
pub const BLOCK_SIZE: u64 = 256 * 1024;

const BLOCK_COUNT: usize = (LEN / BLOCK_SIZE) as usize;
// QEMU pflash_cfi01 advertises 2 KiB per chip, hence 4 KiB for the bank.
// EDK2's P30 driver uses a smaller, compatible 128-byte buffer.
const BUFFER_SIZE: usize = 4096;
const BUFFER_WORDS: usize = BUFFER_SIZE / 4;
const READY: u8 = 0x80;
const ERASE_ERROR: u8 = 0x20;
const PROGRAM_ERROR: u8 = 0x10;
const BLOCK_LOCKED: u8 = 0x02;
const SEQUENCE_ERROR: u8 = ERASE_ERROR | PROGRAM_ERROR;

#[derive(Clone, Copy)]
enum ReadMode {
    Array,
    Status,
    Id,
    Cfi,
}

#[derive(Clone, Copy)]
enum WriteState {
    Idle,
    Program,
    Erase {
        block: usize,
    },
    Lock {
        block: usize,
    },
    BufferCount {
        block: usize,
    },
    BufferData {
        block: usize,
        remaining: usize,
        window: Option<usize>,
    },
    BufferConfirm,
}

#[derive(Clone, Copy)]
struct BufferedWord {
    offset: u32,
    value: u32,
}

pub struct NorFlash {
    storage: Vec<u8>,
    locked: [bool; BLOCK_COUNT],
    mode: ReadMode,
    state: WriteState,
    status: u8,
    buffer: [BufferedWord; BUFFER_WORDS],
    buffered_words: usize,
}

impl NorFlash {
    /// Construct an erased, unlocked variable bank. Its contents are not persisted.
    pub fn new() -> Self {
        Self {
            storage: vec![0xff; LEN as usize],
            locked: [false; BLOCK_COUNT],
            mode: ReadMode::Array,
            state: WriteState::Idle,
            status: READY,
            buffer: [BufferedWord {
                offset: 0,
                value: 0,
            }; BUFFER_WORDS],
            buffered_words: 0,
        }
    }

    fn valid_access(offset: u64, size: u8) -> bool {
        matches!(size, 1 | 2 | 4 | 8)
            && offset
                .checked_add(u64::from(size))
                .is_some_and(|end| end <= LEN)
    }

    fn block(offset: usize) -> usize {
        offset / BLOCK_SIZE as usize
    }

    fn read_array(&mut self) {
        self.mode = ReadMode::Array;
        self.state = WriteState::Idle;
        self.buffered_words = 0;
    }

    fn fail(&mut self, bits: u8) {
        self.status |= READY | bits;
        self.mode = ReadMode::Status;
        self.state = WriteState::Idle;
        self.buffered_words = 0;
    }

    fn id(&self, offset: usize) -> u16 {
        // QEMU virt IDs, from hw/arm/virt.c. ID address 2 is block lock status.
        match (offset / 4) & 0xff {
            0 => 0x89,
            1 => 0x18,
            2 => u16::from(self.locked[Self::block(offset)]),
            _ => 0,
        }
    }

    fn cfi(address: usize) -> u8 {
        // Geometry and Intel PRI table from QEMU hw/block/pflash_cfi01.c.
        // The CFI geometry is per chip: 32 MiB, 256 blocks of 128 KiB each.
        match address {
            0x10 => b'Q',
            0x11 => b'R',
            0x12 => b'Y',
            0x13 => 1,
            0x15 => 0x31,
            0x1b => 0x45,
            0x1c => 0x55,
            0x1f | 0x20 => 7,
            0x21 => 0x0a,
            0x23..=0x25 => 4,
            0x27 => 25,
            0x28 => 2,
            0x2a => 11,
            0x2c => 1,
            0x2d => 0xff,
            0x30 => 2,
            0x31 => b'P',
            0x32 => b'R',
            0x33 => b'I',
            0x34 => b'1',
            0x35 => b'0',
            0x3f => 1,
            _ => 0,
        }
    }

    fn program(&mut self, offset: usize, size: u8, value: u64) {
        if self.locked[Self::block(offset)]
            || self.locked[Self::block(offset + usize::from(size) - 1)]
        {
            self.status |= PROGRAM_ERROR | BLOCK_LOCKED;
            return;
        }
        for byte in 0..usize::from(size) {
            let new = (value >> (byte * 8)) as u8;
            let old = self.storage[offset + byte];
            if new & !old != 0 {
                self.status |= PROGRAM_ERROR;
            }
            // Programming can clear bits only; even erroneous writes never erase.
            self.storage[offset + byte] = old & new;
        }
    }

    fn command(&mut self, offset: usize, command: u8) {
        self.state = WriteState::Idle;
        match command {
            // 0x00 and 0xf0 are QEMU's historical read-array/probe aliases.
            0x00 | 0xff | 0xf0 => self.read_array(),
            0x10 | 0x40 => {
                self.mode = ReadMode::Status;
                self.state = WriteState::Program;
            }
            0x20 => {
                self.mode = ReadMode::Status;
                self.state = WriteState::Erase {
                    block: Self::block(offset),
                };
            }
            0x50 => {
                // Clear sticky errors, not the ready bit. Like QEMU, return to array.
                self.status = READY;
                self.read_array();
            }
            0x60 => {
                self.mode = ReadMode::Status;
                self.state = WriteState::Lock {
                    block: Self::block(offset),
                };
            }
            0x70 => self.mode = ReadMode::Status,
            0x90 => self.mode = ReadMode::Id,
            0x98 => self.mode = ReadMode::Cfi,
            0xe8 => {
                self.mode = ReadMode::Status;
                self.buffered_words = 0;
                self.state = WriteState::BufferCount {
                    block: Self::block(offset),
                };
            }
            _ => self.fail(SEQUENCE_ERROR),
        }
    }
}

impl Default for NorFlash {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for NorFlash {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        if !Self::valid_access(offset, size) {
            return 0;
        }
        let offset = offset as usize;
        let mut value = 0;
        for byte in 0..usize::from(size) {
            let at = offset + byte;
            let data = match self.mode {
                ReadMode::Array => self.storage[at],
                ReadMode::Status => {
                    if at & 1 == 0 {
                        self.status
                    } else {
                        0
                    }
                }
                ReadMode::Id => (self.id(at) >> ((at & 1) * 8)) as u8,
                ReadMode::Cfi => {
                    if at & 1 == 0 {
                        Self::cfi(at / 4)
                    } else {
                        0
                    }
                }
            };
            value |= u64::from(data) << (byte * 8);
        }
        value
    }

    fn write(&mut self, offset: u64, size: u8, value: u64) {
        // A bank write is at most 32 bits. 64-bit array reads are supported, but
        // 64-bit writes are not a single command/data cycle on the x32 bus.
        if !Self::valid_access(offset, size) || size == 8 {
            self.fail(SEQUENCE_ERROR);
            return;
        }
        let offset = offset as usize;
        let command = value as u8;
        match self.state {
            WriteState::Idle => self.command(offset, command),
            WriteState::Program => {
                // This cycle is data, even if its low byte resembles a command.
                self.program(offset, size, value);
                self.status |= READY;
                self.mode = ReadMode::Status;
                self.state = WriteState::Idle;
            }
            WriteState::Erase { block } => {
                if command == 0xff {
                    self.read_array();
                } else if command != 0xd0 || Self::block(offset) != block {
                    self.fail(SEQUENCE_ERROR);
                } else {
                    if self.locked[block] {
                        self.status |= ERASE_ERROR | BLOCK_LOCKED;
                    } else {
                        let start = block * BLOCK_SIZE as usize;
                        self.storage[start..start + BLOCK_SIZE as usize].fill(0xff);
                    }
                    self.status |= READY;
                    self.mode = ReadMode::Status;
                    self.state = WriteState::Idle;
                }
            }
            WriteState::Lock { block } => {
                if command == 0xff {
                    self.read_array();
                } else if !matches!(command, 0x01 | 0xd0) || Self::block(offset) != block {
                    self.fail(SEQUENCE_ERROR);
                } else {
                    self.locked[block] = command == 0x01;
                    self.status |= READY;
                    self.mode = ReadMode::Status;
                    self.state = WriteState::Idle;
                }
            }
            WriteState::BufferCount { block } => {
                // Counts are per x16 chip and zero means one bank word.
                let words = usize::from(value as u16) + 1;
                if size != 4
                    || offset & 3 != 0
                    || Self::block(offset) != block
                    || words > BUFFER_WORDS
                {
                    self.fail(SEQUENCE_ERROR);
                } else {
                    self.state = WriteState::BufferData {
                        block,
                        remaining: words,
                        window: None,
                    };
                }
            }
            WriteState::BufferData {
                block,
                remaining,
                window,
            } => {
                let data_window = offset & !(BUFFER_SIZE - 1);
                if size != 4
                    || offset & 3 != 0
                    || Self::block(offset) != block
                    || window.is_some_and(|start| start != data_window)
                {
                    self.fail(SEQUENCE_ERROR);
                    return;
                }
                self.buffer[self.buffered_words] = BufferedWord {
                    offset: offset as u32,
                    value: value as u32,
                };
                self.buffered_words += 1;
                self.state = if remaining == 1 {
                    WriteState::BufferConfirm
                } else {
                    WriteState::BufferData {
                        block,
                        remaining: remaining - 1,
                        window: Some(data_window),
                    }
                };
            }
            WriteState::BufferConfirm => {
                if command == 0xff {
                    self.read_array();
                } else if command != 0xd0 {
                    self.fail(SEQUENCE_ERROR);
                } else {
                    // EDK2 confirms at DeviceBaseAddress, not the target block.
                    // Do not restrict the confirmation address to the data window.
                    for index in 0..self.buffered_words {
                        let word = self.buffer[index];
                        self.program(word.offset as usize, 4, u64::from(word.value));
                    }
                    self.buffered_words = 0;
                    self.status |= READY;
                    self.mode = ReadMode::Status;
                    self.state = WriteState::Idle;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(flash: &mut NorFlash, offset: u64, command: u16) {
        flash.write(offset, 4, u64::from(command) | (u64::from(command) << 16));
    }

    fn status(flash: &mut NorFlash) -> u64 {
        command(flash, 0, 0x70);
        flash.read(0, 4)
    }

    fn program(flash: &mut NorFlash, offset: u64, value: u32) {
        command(flash, offset, 0x40);
        flash.write(offset, 4, u64::from(value));
    }

    #[test]
    fn erased_array_and_programming_preserve_little_endian_accesses() {
        let mut flash = NorFlash::new();
        assert_eq!(flash.read(0, 8), u64::MAX);
        assert_eq!(flash.read(LEN - 1, 1), 0xff);
        program(&mut flash, 0x100, 0x1234_5678);
        assert_eq!(status(&mut flash), 0x0080_0080);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(0x100, 1), 0x78);
        assert_eq!(flash.read(0x101, 2), 0x3456);
        assert_eq!(flash.read(0x100, 4), 0x1234_5678);
        assert_eq!(flash.read(0x100, 8), 0xffff_ffff_1234_5678);
        assert_eq!(flash.read(0x103, 4), 0xffff_ff12);

        command(&mut flash, 0x104, 0x10);
        flash.write(0x104, 2, 0x70ff); // Command-looking bytes are still data.
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(0x104, 4), 0xffff_70ff);
        command(&mut flash, 0x106, 0x40);
        flash.write(0x106, 1, 0x90);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(0x104, 4), 0xff90_70ff);
    }

    #[test]
    fn programming_cannot_set_bits_and_errors_are_sticky_until_cleared() {
        let mut flash = NorFlash::new();
        program(&mut flash, 0, 0);
        program(&mut flash, 0, 0xffff_ffff);
        assert_eq!(status(&mut flash), 0x0090_0090);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(0, 4), 0);
        program(&mut flash, 4, 0x1234);
        assert_eq!(status(&mut flash), 0x0090_0090);
        command(&mut flash, 0, 0x50);
        assert_eq!(flash.read(4, 4), 0x1234);
        assert_eq!(status(&mut flash), 0x0080_0080);
    }

    #[test]
    fn erase_requires_confirmation_and_changes_only_the_selected_block() {
        let mut flash = NorFlash::new();
        program(&mut flash, 0, 0);
        program(&mut flash, BLOCK_SIZE, 0);
        command(&mut flash, 0, 0x20);
        assert_eq!(flash.storage[0], 0);
        command(&mut flash, 0, 0xff); // Cancel before confirmation.
        assert_eq!(flash.read(0, 4), 0);
        command(&mut flash, 0x100, 0x20);
        command(&mut flash, 0, 0xd0);
        assert_eq!(status(&mut flash), 0x0080_0080);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(0, 4), 0xffff_ffff);
        assert_eq!(flash.read(BLOCK_SIZE - 4, 4), 0xffff_ffff);
        assert_eq!(flash.read(BLOCK_SIZE, 4), 0);

        command(&mut flash, BLOCK_SIZE, 0x20);
        command(&mut flash, 0, 0xd0); // Wrong block must not erase anything.
        assert_eq!(status(&mut flash), 0x00b0_00b0);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(BLOCK_SIZE, 4), 0);
    }

    #[test]
    fn ids_lock_unlock_and_cfi_describe_the_interleaved_bank() {
        let mut flash = NorFlash::new();
        command(&mut flash, 0, 0x90);
        assert_eq!(flash.read(0, 4), 0x0089_0089);
        assert_eq!(flash.read(4, 4), 0x0018_0018);
        assert_eq!(flash.read(0, 8), 0x0018_0018_0089_0089);
        assert_eq!(flash.read(BLOCK_SIZE + 8, 4), 0);

        command(&mut flash, BLOCK_SIZE, 0x60);
        command(&mut flash, BLOCK_SIZE, 0x01);
        command(&mut flash, BLOCK_SIZE + 8, 0x90);
        assert_eq!(flash.read(BLOCK_SIZE + 8, 4), 0x0001_0001);
        program(&mut flash, BLOCK_SIZE, 0);
        assert_eq!(status(&mut flash), 0x0092_0092);
        command(&mut flash, 0, 0x50);
        command(&mut flash, BLOCK_SIZE, 0x20);
        command(&mut flash, BLOCK_SIZE, 0xd0);
        assert_eq!(status(&mut flash), 0x00a2_00a2);
        command(&mut flash, 0, 0x50);
        command(&mut flash, BLOCK_SIZE, 0x60);
        command(&mut flash, BLOCK_SIZE, 0xd0);
        program(&mut flash, BLOCK_SIZE, 0);
        assert_eq!(status(&mut flash), 0x0080_0080);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(BLOCK_SIZE, 4), 0);

        command(&mut flash, 0x55 * 4, 0x98);
        assert_eq!(flash.read(0x10 * 4, 4), 0x0051_0051);
        assert_eq!(flash.read(0x11 * 4, 4), 0x0052_0052);
        assert_eq!(flash.read(0x12 * 4, 4), 0x0059_0059);
        assert_eq!(flash.read(0x27 * 4, 4), 0x0019_0019);
        assert_eq!(flash.read(0x2a * 4, 4), 0x000b_000b);
        assert_eq!(flash.read(0x2d * 4, 4), 0x00ff_00ff);
        assert_eq!(flash.read(0x30 * 4, 4), 0x0002_0002);
        assert_eq!(flash.read(0x52 * 4, 4), 0);
    }

    #[test]
    fn buffered_program_stages_data_until_base_address_confirmation() {
        let mut flash = NorFlash::new();
        let target = BLOCK_SIZE + 128;
        command(&mut flash, target, 0xe8);
        assert_eq!(flash.read(target, 4), 0x0080_0080);
        command(&mut flash, target, 1); // Two words.
        flash.write(target, 4, 0x1234_5678);
        flash.write(target + 4, 4, 0xd0ff_70e8);
        assert_eq!(flash.storage[target as usize], 0xff);
        command(&mut flash, 0, 0xd0); // EDK2 confirms at the bank base.
        assert_eq!(status(&mut flash), 0x0080_0080);
        command(&mut flash, 0, 0x50);
        assert_eq!(flash.read(target, 8), 0xd0ff_70e8_1234_5678);

        command(&mut flash, target + 8, 0xe8);
        command(&mut flash, target + 8, 0); // Count zero is one word.
        flash.write(target + 8, 4, 0x1122_33ff);
        command(&mut flash, 0, 0xd0);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(target + 8, 4), 0x1122_33ff);

        command(&mut flash, target, 0xe8);
        command(&mut flash, target, 0);
        flash.write(target, 4, 0xffff_ffff);
        command(&mut flash, 0, 0xd0);
        assert_eq!(status(&mut flash), 0x0090_0090);
        command(&mut flash, 0, 0x50);
        assert_eq!(flash.read(target, 4), 0x1234_5678);
    }

    #[test]
    fn full_edk2_buffer_and_last_bank_word_are_programmable() {
        let mut flash = NorFlash::new();
        let target = LEN - 128;
        command(&mut flash, target, 0xe8);
        command(&mut flash, target, 31);
        for index in 0..32 {
            flash.write(target + index * 4, 4, 0xa500_0000 | index);
        }
        command(&mut flash, 0, 0xd0);
        assert_eq!(status(&mut flash), 0x0080_0080);
        command(&mut flash, 0, 0xff);
        for index in 0..32 {
            assert_eq!(flash.read(target + index * 4, 4), 0xa500_0000 | index);
        }

        let target = LEN - BUFFER_SIZE as u64;
        command(&mut flash, target, 0xe8);
        command(&mut flash, target, (BUFFER_WORDS - 1) as u16);
        for index in 0..BUFFER_WORDS as u64 {
            // Keep the earlier P30 data unchanged while testing the larger
            // buffer capacity advertised by QEMU's CFI table.
            let value = if index < BUFFER_WORDS as u64 - 32 {
                0x5a00_0000 | index
            } else {
                0xa500_0000 | (index - (BUFFER_WORDS as u64 - 32))
            };
            flash.write(target + index * 4, 4, value);
        }
        command(&mut flash, 0, 0xd0);
        assert_eq!(status(&mut flash), 0x0080_0080);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(target, 4), 0x5a00_0000);
        assert_eq!(flash.read(LEN - 4, 4), 0xa500_001f);
    }

    #[test]
    fn invalid_buffer_sequences_discard_staged_data() {
        let mut flash = NorFlash::new();
        command(&mut flash, 0, 0xe8);
        command(&mut flash, 0, BUFFER_WORDS as u16); // One word too many.
        assert_eq!(status(&mut flash), 0x00b0_00b0);
        command(&mut flash, 0, 0x50);

        command(&mut flash, 0, 0xe8);
        command(&mut flash, 0, 1);
        flash.write(0, 4, 0);
        flash.write(BUFFER_SIZE as u64, 4, 0); // Cross the write-buffer window.
        assert_eq!(status(&mut flash), 0x00b0_00b0);
        command(&mut flash, 0, 0x50);
        assert_eq!(flash.read(0, 8), u64::MAX);
        assert_eq!(flash.read(BUFFER_SIZE as u64, 4), 0xffff_ffff);

        command(&mut flash, 0, 0xe8);
        command(&mut flash, 0, 0);
        flash.write(0, 4, 0);
        command(&mut flash, 0, 0x70); // Not a confirmation.
        assert_eq!(status(&mut flash), 0x00b0_00b0);
        command(&mut flash, 0, 0x50);
        assert_eq!(flash.read(0, 4), 0xffff_ffff);

        command(&mut flash, 0, 0xe8);
        command(&mut flash, 0, 0);
        flash.write(0, 4, 0);
        command(&mut flash, 0, 0xff); // Explicit cancellation.
        assert_eq!(flash.read(0, 4), 0xffff_ffff);
    }

    #[test]
    fn invalid_commands_and_accesses_are_bounded_and_recoverable() {
        let mut flash = NorFlash::new();
        assert_eq!(flash.read(LEN, 1), 0);
        assert_eq!(flash.read(LEN - 3, 4), 0);
        assert_eq!(flash.read(u64::MAX, 8), 0);
        assert_eq!(flash.read(0, 0), 0);
        assert_eq!(flash.read(0, 3), 0);
        flash.write(LEN - 3, 4, 0);
        assert_eq!(status(&mut flash), 0x00b0_00b0);
        command(&mut flash, 0, 0x50);
        assert_eq!(flash.read(LEN - 4, 4), 0xffff_ffff);
        command(&mut flash, 0, 0xab);
        assert_eq!(status(&mut flash), 0x00b0_00b0);
        command(&mut flash, 0, 0x50);
        program(&mut flash, LEN - 4, 0x1234_5678);
        command(&mut flash, 0, 0xff);
        assert_eq!(flash.read(LEN - 4, 4), 0x1234_5678);

        command(&mut flash, 0, 0x40);
        flash.write(u64::MAX, 4, 0);
        assert_eq!(status(&mut flash), 0x00b0_00b0);
        command(&mut flash, 0, 0x50);
        assert_eq!(flash.read(0, 4), 0xffff_ffff);
        flash.write(0, 8, 0);
        assert_eq!(status(&mut flash), 0x00b0_00b0);
    }
}
