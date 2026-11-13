//! GFX10 PM4 command stream encoder and standard packet definitions.
#![allow(dead_code)]

use alloc::vec::Vec;

pub const PKT3_NOP: u32 = 0x10;
pub const PKT3_SET_BASE: u32 = 0x11;
pub const PKT3_CLEAR_STATE: u32 = 0x12;
pub const PKT3_DISPATCH_DIRECT: u32 = 0x15;
pub const PKT3_DISPATCH_INDIRECT: u32 = 0x16;
pub const PKT3_DRAW_INDEX_2: u32 = 0x27;
pub const PKT3_CONTEXT_CONTROL: u32 = 0x28;
pub const PKT3_INDEX_TYPE: u32 = 0x2A;
pub const PKT3_DRAW_INDEX_AUTO: u32 = 0x2D;
pub const PKT3_NUM_INSTANCES: u32 = 0x2F;
pub const PKT3_EVENT_WRITE: u32 = 0x46;
pub const PKT3_SET_CONFIG_REG: u32 = 0x68;
pub const PKT3_SET_CONTEXT_REG: u32 = 0x69;
pub const PKT3_SET_SH_REG: u32 = 0x76;
pub const PKT3_SET_UCONFIG_REG: u32 = 0x79;
pub const PKT3_SET_UCONFIG_REG_INDEX: u32 = 0x7A;
pub const PKT3_RELEASE_MEM: u32 = 0x49;
pub const PKT3_ACQUIRE_MEM: u32 = 0x58;

/// `EVENT_WRITE` event types (values of `VGT_EVENT_TYPE`).
///
/// The payload encoding is `EVENT_TYPE(x) | EVENT_INDEX(i << 8)`; index 0 is
/// the plain "flush, no timestamp destination" form.
pub const EVENT_CACHE_FLUSH_AND_INV_TS: u32 = 20;
pub const EVENT_FLUSH_AND_INV_CB_DATA: u32 = 45;

pub const SI_CONFIG_REG_OFFSET: u32 = 0x08000;
pub const SI_CONTEXT_REG_OFFSET: u32 = 0x28000;
pub const SI_SH_REG_OFFSET: u32 = 0x0B000;
pub const SI_UCONFIG_REG_OFFSET: u32 = 0x30000;

/// GFX10 header-only NOP; count 0x3fff is special (Mesa PKT3_NOP_PAD).
pub const PM4_NOP_1DW: u32 = packet3(PKT3_NOP, 0x3fff);

#[inline(always)]
pub const fn packet3(op: u32, count_minus_one: u32) -> u32 {
    (3 << 30) | ((count_minus_one & 0x3FFF) << 16) | ((op & 0xFF) << 8)
}

/// Dynamic buffer of PM4 dwords.
#[derive(Default, Debug, Clone)]
pub struct Pm4 {
    pub dwords: Vec<u32>,
}

impl Pm4 {
    pub fn new() -> Self {
        Self { dwords: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            dwords: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.dwords.clear();
    }

    pub fn len(&self) -> usize {
        self.dwords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dwords.is_empty()
    }

    pub fn push(&mut self, dw: u32) {
        self.dwords.push(dw);
    }

    pub fn extend(&mut self, words: &[u32]) {
        self.dwords.extend_from_slice(words);
    }

    pub fn set_context_reg(&mut self, reg_byte_addr: u32, val: u32) {
        debug_assert!(reg_byte_addr >= SI_CONTEXT_REG_OFFSET);
        let reg_dw = (reg_byte_addr - SI_CONTEXT_REG_OFFSET) >> 2;
        self.dwords.push(packet3(PKT3_SET_CONTEXT_REG, 1));
        self.dwords.push(reg_dw);
        self.dwords.push(val);
    }

    pub fn set_context_reg_seq(&mut self, start_byte_addr: u32, vals: &[u32]) {
        debug_assert!(start_byte_addr >= SI_CONTEXT_REG_OFFSET);
        let reg_dw = (start_byte_addr - SI_CONTEXT_REG_OFFSET) >> 2;
        self.dwords
            .push(packet3(PKT3_SET_CONTEXT_REG, vals.len() as u32));
        self.dwords.push(reg_dw);
        self.dwords.extend_from_slice(vals);
    }

    pub fn set_context_reg_idx(&mut self, reg_byte_addr: u32, idx: u32, val: u32) {
        debug_assert!(reg_byte_addr >= SI_CONTEXT_REG_OFFSET);
        let reg_dw = (reg_byte_addr - SI_CONTEXT_REG_OFFSET) >> 2;
        self.dwords.push(packet3(PKT3_SET_CONTEXT_REG, 1));
        // Multi-dword registers select their dword through the high bits of the
        // register offset, like SET_UCONFIG_REG_INDEX.
        self.dwords.push(reg_dw | ((idx & 0xF) << 28));
        self.dwords.push(val);
    }

    pub fn set_sh_reg(&mut self, reg_byte_addr: u32, val: u32) {
        debug_assert!(reg_byte_addr >= SI_SH_REG_OFFSET);
        let reg_dw = (reg_byte_addr - SI_SH_REG_OFFSET) >> 2;
        self.dwords.push(packet3(PKT3_SET_SH_REG, 1));
        self.dwords.push(reg_dw);
        self.dwords.push(val);
    }

    pub fn set_sh_reg_seq(&mut self, start_byte_addr: u32, vals: &[u32]) {
        debug_assert!(start_byte_addr >= SI_SH_REG_OFFSET);
        let reg_dw = (start_byte_addr - SI_SH_REG_OFFSET) >> 2;
        self.dwords
            .push(packet3(PKT3_SET_SH_REG, vals.len() as u32));
        self.dwords.push(reg_dw);
        self.dwords.extend_from_slice(vals);
    }

    pub fn set_uconfig_reg(&mut self, reg_byte_addr: u32, val: u32) {
        debug_assert!(reg_byte_addr >= SI_UCONFIG_REG_OFFSET);
        let reg_dw = (reg_byte_addr - SI_UCONFIG_REG_OFFSET) >> 2;
        self.dwords.push(packet3(PKT3_SET_UCONFIG_REG, 1));
        self.dwords.push(reg_dw);
        self.dwords.push(val);
    }

    pub fn set_uconfig_reg_idx(&mut self, reg_byte_addr: u32, idx: u32, val: u32) {
        debug_assert!(reg_byte_addr >= SI_UCONFIG_REG_OFFSET);
        let reg_dw = (reg_byte_addr - SI_UCONFIG_REG_OFFSET) >> 2;
        self.dwords.push(packet3(PKT3_SET_UCONFIG_REG_INDEX, 1));
        self.dwords.push(reg_dw | ((idx & 0xF) << 28));
        self.dwords.push(val);
    }

    pub fn context_control(&mut self) {
        self.dwords.push(packet3(PKT3_CONTEXT_CONTROL, 1));
        self.dwords.push(0x80000000); // S_281_UPDATE_LOAD_ENABLES(1)
        self.dwords.push(0x80000000); // S_282_UPDATE_SHADOW_ENABLES(1)
    }

    pub fn clear_state(&mut self) {
        self.dwords.push(packet3(PKT3_CLEAR_STATE, 0));
        self.dwords.push(0);
    }

    pub fn num_instances(&mut self, count: u32) {
        self.dwords.push(packet3(PKT3_NUM_INSTANCES, 0));
        self.dwords.push(count);
    }

    /// `DRAW_INDEX_AUTO(count, DI_SRC_SEL_AUTO_INDEX)`.
    pub fn draw_index_auto(&mut self, count: u32) {
        self.dwords.push(packet3(PKT3_DRAW_INDEX_AUTO, 1));
        self.dwords.push(count);
        self.dwords.push(0x00000002); // V_0287F0_DI_SRC_SEL_AUTO_INDEX
    }

    /// `DRAW_INDEX_2(max_size, va_lo, va_hi, count, DI_SRC_SEL_DMA)`.
    pub fn draw_index_2(&mut self, max_size: u32, va: u64, count: u32) {
        self.dwords.push(packet3(PKT3_DRAW_INDEX_2, 4));
        self.dwords.push(max_size);
        self.dwords.push(va as u32);
        self.dwords.push((va >> 32) as u32);
        self.dwords.push(count);
        self.dwords.push(0x00000000); // V_0287F0_DI_SRC_SEL_DMA
    }

    /// `EVENT_WRITE(event_type, index)`: flushes/idles the pipeline stages
    /// named by `event_type`. `index` 0 performs the flush without writing a
    /// timestamp to memory (the kernel encodes the same way:
    /// `EVENT_TYPE(x) = x << 0`, `EVENT_INDEX(i) = i << 8`).
    pub fn event_write(&mut self, event_type: u32, index: u32) {
        self.dwords.push(packet3(PKT3_EVENT_WRITE, 0));
        self.dwords.push((event_type & 0xFF) | ((index & 0xF) << 8));
    }
    /// `RELEASE_MEM` with EOP fence and cache flush/invalidation.
    /// Emits a 7-dword packet (PKT3_RELEASE_MEM, count 6).
    pub fn release_mem(
        &mut self,
        event_type: u32,
        event_index: u32,
        gcr_cntl: u32,
        dst_sel: u32,
        int_sel: u32,
        data_sel: u32,
        gpu_va: u64,
        data: u32,
    ) {
        self.dwords.push(packet3(PKT3_RELEASE_MEM, 6));
        let w0 = (event_type & 0xff) | ((event_index & 0xf) << 8) | ((gcr_cntl & 0xfffff) << 12);
        let w1 = (dst_sel & 0x3) << 16 | (int_sel & 0x7) << 24 | (data_sel & 0x7) << 29;
        self.dwords.push(w0);
        self.dwords.push(w1);
        self.dwords.push(gpu_va as u32);
        self.dwords.push((gpu_va >> 32) as u32);
        self.dwords.push(data);
        self.dwords.push(0); // immediate data hi
        self.dwords.push(0); // unused
    }

    /// Pad to a multiple of `align_dwords` using standard NOP packets.
    pub fn pad_to(&mut self, align_dwords: usize) {
        let rem = self.dwords.len() % align_dwords;
        if rem == 0 {
            return;
        }
        let pad = align_dwords - rem;
        for _ in 0..pad {
            self.dwords.push(PM4_NOP_1DW);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_uses_header_only_nops() {
        let mut pm4 = Pm4::new();
        pm4.num_instances(1);
        pm4.pad_to(32);
        assert_eq!(pm4.len(), 32);
        assert!(pm4.dwords[2..].iter().all(|&dw| dw == 0xffff_1000));
    }
}
