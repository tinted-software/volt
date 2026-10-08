//! SIMD structure loads and stores: `LD1`..`LD4`, `ST1`..`ST4`, their single-lane
//! forms and the replicating `LD1R`..`LD4R`.
//!
//! Decode describes one in a [`StructDesc`]. The JIT copies the descriptor into
//! [`Cpu::structure`](super::Cpu) and traps; the machine (or user mode) then runs
//! [`execute`] against its own memory, one element at a time, so each backend keeps
//! its own translation, device and fault handling.

/// Which elements of the registers a structure access touches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Shape {
    /// Whole vectors: `ld1 {v0.4s - v3.4s}` and the interleaved `ld2`..`ld4`.
    #[default]
    Multiple,
    /// One element of each register: `ld1 {v0.s}[1]`. Other lanes are kept.
    Lane,
    /// One element broadcast to every lane: `ld1r`. Loads only.
    Replicate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct StructDesc {
    pub store: bool,
    pub shape: Shape,
    /// Registers touched, starting at `rt` and wrapping from v31 to v0.
    pub registers: u8,
    /// `ld2`..`ld4`: consecutive memory elements alternate between the registers.
    /// Otherwise (`ld1`) each register is a contiguous block.
    pub interleaved: bool,
    /// Element size in bytes: 1, 2, 4 or 8.
    pub esize: u8,
    /// Element index of a [`Shape::Lane`] access.
    pub lane: u8,
    /// A 128-bit vector (otherwise 64-bit, with the upper half zeroed on loads).
    pub q: bool,
    pub rt: u8,
}

impl StructDesc {
    fn vector_bytes(&self) -> u8 {
        if self.q { 16 } else { 8 }
    }
    /// Bytes of memory the access spans; the post-index immediate advances by this.
    pub fn bytes(&self) -> u64 {
        let registers = u64::from(self.registers);
        match self.shape {
            Shape::Multiple => registers * u64::from(self.vector_bytes()),
            Shape::Lane | Shape::Replicate => registers * u64::from(self.esize),
        }
    }
}

fn mask(esize: u8) -> u64 {
    if esize >= 8 {
        u64::MAX
    } else {
        (1 << (u32::from(esize) * 8)) - 1
    }
}
fn extract(register: &[u64; 2], esize: u8, lane: u8) -> u64 {
    let bit = u32::from(lane) * u32::from(esize) * 8;
    (register[(bit / 64) as usize] >> (bit % 64)) & mask(esize)
}
fn insert(register: &mut [u64; 2], esize: u8, lane: u8, value: u64) {
    let bit = u32::from(lane) * u32::from(esize) * 8;
    let (word, shift) = ((bit / 64) as usize, bit % 64);
    register[word] = (register[word] & !(mask(esize) << shift)) | ((value & mask(esize)) << shift);
}

/// Run `desc` at `base` over the register file `v`. `access(address, width, store)`
/// moves one element: it writes `store` when given, otherwise it reads and returns
/// the element, zero-extended. The first error stops the access; `v` may then be
/// partly updated, so callers work on a copy and keep it only on success.
pub fn execute<E>(
    desc: &StructDesc,
    v: &mut [[u64; 2]; 32],
    base: u64,
    mut access: impl FnMut(u64, u8, Option<u64>) -> Result<u64, E>,
) -> Result<(), E> {
    let esize = desc.esize;
    let registers = usize::from(desc.registers);
    let register = |r: usize| usize::from(desc.rt) + r & 31;
    let mut element = |v: &mut [[u64; 2]; 32], r: usize, lane: u8, address: u64| {
        let reg = register(r);
        if desc.store {
            access(address, esize, Some(extract(&v[reg], esize, lane))).map(drop)
        } else {
            access(address, esize, None).map(|value| insert(&mut v[reg], esize, lane, value))
        }
    };
    match desc.shape {
        Shape::Multiple => {
            let lanes = usize::from(desc.vector_bytes() / esize);
            if !desc.store {
                for r in 0..registers {
                    v[register(r)] = [0; 2];
                }
            }
            for slot in 0..registers * lanes {
                let (r, lane) = if desc.interleaved {
                    (slot % registers, slot / registers)
                } else {
                    (slot / lanes, slot % lanes)
                };
                let address = base.wrapping_add(slot as u64 * u64::from(esize));
                element(v, r, lane as u8, address)?;
            }
        }
        Shape::Lane => {
            for r in 0..registers {
                let address = base.wrapping_add(r as u64 * u64::from(esize));
                element(v, r, desc.lane, address)?;
            }
        }
        Shape::Replicate => {
            let lanes = desc.vector_bytes() / esize;
            for r in 0..registers {
                let address = base.wrapping_add(r as u64 * u64::from(esize));
                let value = access(address, esize, None)?;
                let mut broadcast = [0; 2];
                for lane in 0..lanes {
                    insert(&mut broadcast, esize, lane, value);
                }
                v[register(r)] = broadcast;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;

    fn memory(len: u8) -> Vec<u8> {
        (0..len).collect()
    }
    fn run(desc: &StructDesc, v: &mut [[u64; 2]; 32], memory: &mut [u8]) {
        let result: Result<(), Infallible> = execute(desc, v, 0, |address, width, store| {
            let at = address as usize..address as usize + usize::from(width);
            Ok(match store {
                Some(value) => {
                    memory[at].copy_from_slice(&value.to_le_bytes()[..usize::from(width)]);
                    0
                }
                None => {
                    let mut bytes = [0; 8];
                    bytes[..usize::from(width)].copy_from_slice(&memory[at]);
                    u64::from_le_bytes(bytes)
                }
            })
        });
        result.unwrap();
    }
    fn bytes(register: [u64; 2]) -> Vec<u8> {
        register.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn ld2_deinterleaves_elements() {
        // ld2 {v0.4h, v1.4h}: v0 gets halfwords 0, 2, 4, 6; v1 gets 1, 3, 5, 7.
        let desc = StructDesc {
            registers: 2,
            interleaved: true,
            esize: 2,
            ..StructDesc::default()
        };
        let mut v = [[u64::MAX; 2]; 32];
        run(&desc, &mut v, &mut memory(16));
        assert_eq!(
            bytes(v[0]),
            [0, 1, 4, 5, 8, 9, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            bytes(v[1]),
            [2, 3, 6, 7, 10, 11, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }
    #[test]
    fn st3_interleaves_and_wraps_registers() {
        // st3 {v31.16b, v0.16b, v1.16b}: register numbers wrap, memory alternates.
        let desc = StructDesc {
            store: true,
            registers: 3,
            interleaved: true,
            esize: 1,
            q: true,
            rt: 31,
            ..StructDesc::default()
        };
        let mut v = [[0; 2]; 32];
        v[31] = [u64::from_le_bytes([1; 8]); 2];
        v[0] = [u64::from_le_bytes([2; 8]); 2];
        v[1] = [u64::from_le_bytes([3; 8]); 2];
        let mut out = vec![0; 48];
        run(&desc, &mut v, &mut out);
        assert!(out.chunks(3).all(|c| c == [1, 2, 3]));
    }
    #[test]
    fn ld1_registers_are_contiguous_blocks() {
        let desc = StructDesc {
            registers: 2,
            esize: 4,
            q: true,
            ..StructDesc::default()
        };
        let mut v = [[0; 2]; 32];
        run(&desc, &mut v, &mut memory(32));
        assert_eq!(bytes(v[0]), (0..16).collect::<Vec<u8>>());
        assert_eq!(bytes(v[1]), (16..32).collect::<Vec<u8>>());
    }
    #[test]
    fn lane_load_keeps_other_lanes_and_store_reads_the_lane() {
        // ld1 {v3.s}[1] replaces bits 63:32 only.
        let desc = StructDesc {
            shape: Shape::Lane,
            registers: 1,
            esize: 4,
            lane: 1,
            rt: 3,
            ..StructDesc::default()
        };
        let mut v = [[0; 2]; 32];
        v[3] = [0x1111_1111_2222_2222, 0x3333_3333_4444_4444];
        run(&desc, &mut v, &mut [0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(v[3], [0xddcc_bbaa_2222_2222, 0x3333_3333_4444_4444]);
        // st1 {v3.d}[1] writes the upper doubleword.
        let store = StructDesc {
            store: true,
            esize: 8,
            lane: 1,
            ..desc
        };
        let mut out = [0; 8];
        run(&store, &mut v, &mut out);
        assert_eq!(u64::from_le_bytes(out), 0x3333_3333_4444_4444);
    }
    #[test]
    fn lane_form_spans_consecutive_elements_per_register() {
        // ld3 {v0.s, v1.s, v2.s}[3]: elements are adjacent in memory, lane 3 of each.
        let desc = StructDesc {
            shape: Shape::Lane,
            registers: 3,
            esize: 4,
            lane: 3,
            q: true,
            ..StructDesc::default()
        };
        let mut v = [[0; 2]; 32];
        run(&desc, &mut v, &mut memory(12));
        assert_eq!(v[0][1] >> 32, 0x0302_0100);
        assert_eq!(v[1][1] >> 32, 0x0706_0504);
        assert_eq!(v[2][1] >> 32, 0x0b0a_0908);
        assert_eq!(desc.bytes(), 12);
    }
    #[test]
    fn replicate_fills_every_lane_and_zeroes_the_upper_half() {
        // ld1r {v0.8b}: one byte everywhere in the low half, upper half cleared.
        let desc = StructDesc {
            shape: Shape::Replicate,
            registers: 1,
            esize: 1,
            ..StructDesc::default()
        };
        let mut v = [[u64::MAX; 2]; 32];
        run(&desc, &mut v, &mut [0x5a]);
        assert_eq!(v[0], [0x5a5a_5a5a_5a5a_5a5a, 0]);
        // ld1r {v0.2d}: both doublewords.
        let wide = StructDesc {
            esize: 8,
            q: true,
            ..desc
        };
        run(&wide, &mut v, &mut 7u64.to_le_bytes());
        assert_eq!(v[0], [7, 7]);
    }
    #[test]
    fn bytes_is_the_post_index_advance() {
        let ld4 = StructDesc {
            registers: 4,
            interleaved: true,
            esize: 4,
            q: true,
            ..StructDesc::default()
        };
        assert_eq!(ld4.bytes(), 64);
        assert_eq!(StructDesc { q: false, ..ld4 }.bytes(), 32);
    }
    #[test]
    fn first_error_stops_the_access() {
        let desc = StructDesc {
            registers: 2,
            interleaved: true,
            esize: 8,
            q: true,
            ..StructDesc::default()
        };
        let mut v = [[0; 2]; 32];
        let mut calls = 0;
        let result = execute(&desc, &mut v, 0, |_, _, _| {
            calls += 1;
            if calls == 2 { Err("fault") } else { Ok(1) }
        });
        assert_eq!(result, Err("fault"));
        assert_eq!(calls, 2);
    }
}
