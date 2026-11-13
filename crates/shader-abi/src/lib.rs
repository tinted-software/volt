//! Versioned binary ABI for precompiled GPU shaders.
//!
//! Layout (all multi-byte integers little-endian):
//! - `[0..4]`:   magic `b"VSHD"`
//! - `[4..6]`:   format version (`u16`, currently `1`)
//! - `[6]`:      stage (`0` = Vertex, `1` = Pixel)
//! - `[7]`:      flags (`FLAG_USES_KILL = 1 << 0`)
//! - `[8..10]`:  `user_sgprs` (`u16`)
//! - `[10..12]`: `param_exports` (`u16`)
//! - `[12..16]`: `config_count` (`u32`, number of `(reg, value)` pairs)
//! - `[16..20]`: `code_len` (`u32`, byte length of machine code)
//! - `[20..20 + config_count * 8]`: `(reg: u32, value: u32)` loader config pairs
//! - `[.. + code_len]`: raw machine code bytes (4-byte aligned length)

#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

pub const MAGIC: [u8; 4] = *b"VSHD";
pub const VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 20;
pub const FLAG_USES_KILL: u8 = 1 << 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShaderStage {
    Vertex = 0,
    Pixel = 1,
}

impl ShaderStage {
    pub const fn from_u8(raw: u8) -> Result<Self, BlobError> {
        match raw {
            0 => Ok(Self::Vertex),
            1 => Ok(Self::Pixel),
            _ => Err(BlobError::InvalidStage(raw)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u16),
    InvalidStage(u8),
    Truncated,
    UnalignedCode(u32),
    TrailingBytes,
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => f.write_str("shader blob shorter than header"),
            Self::BadMagic => f.write_str("invalid shader blob magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported shader blob version {v}"),
            Self::InvalidStage(s) => write!(f, "invalid shader stage {s}"),
            Self::Truncated => f.write_str("shader blob payload truncated"),
            Self::UnalignedCode(n) => {
                write!(f, "shader machine code length {n} not 4-byte aligned")
            }
            Self::TrailingBytes => f.write_str("unexpected trailing bytes in shader blob"),
        }
    }
}

impl core::error::Error for BlobError {}

/// Zero-copy view over a parsed binary shader blob.
#[derive(Clone, Copy, Debug)]
pub struct ShaderBlob<'a> {
    stage: ShaderStage,
    flags: u8,
    user_sgprs: u16,
    param_exports: u16,
    config_bytes: &'a [u8],
    code: &'a [u8],
}

impl<'a> ShaderBlob<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, BlobError> {
        if bytes.len() < HEADER_SIZE {
            return Err(BlobError::TooShort);
        }
        if bytes[0..4] != MAGIC {
            return Err(BlobError::BadMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(BlobError::UnsupportedVersion(version));
        }
        let stage = ShaderStage::from_u8(bytes[6])?;
        let flags = bytes[7];
        let user_sgprs = u16::from_le_bytes([bytes[8], bytes[9]]);
        let param_exports = u16::from_le_bytes([bytes[10], bytes[11]]);
        let config_count =
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let code_len_u32 = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        if !code_len_u32.is_multiple_of(4) {
            return Err(BlobError::UnalignedCode(code_len_u32));
        }
        let code_len = code_len_u32 as usize;
        let config_bytes_len = config_count.checked_mul(8).ok_or(BlobError::Truncated)?;
        let config_end = HEADER_SIZE
            .checked_add(config_bytes_len)
            .ok_or(BlobError::Truncated)?;
        let total = config_end
            .checked_add(code_len)
            .ok_or(BlobError::Truncated)?;
        if bytes.len() < total {
            return Err(BlobError::Truncated);
        }
        if bytes.len() > total {
            return Err(BlobError::TrailingBytes);
        }
        Ok(Self {
            stage,
            flags,
            user_sgprs,
            param_exports,
            config_bytes: &bytes[HEADER_SIZE..config_end],
            code: &bytes[config_end..total],
        })
    }

    pub const fn stage(&self) -> ShaderStage {
        self.stage
    }

    pub const fn flags(&self) -> u8 {
        self.flags
    }

    pub const fn uses_kill(&self) -> bool {
        self.flags & FLAG_USES_KILL != 0
    }

    pub const fn user_sgprs(&self) -> u16 {
        self.user_sgprs
    }

    pub const fn param_exports(&self) -> u16 {
        self.param_exports
    }

    pub fn config_len(&self) -> usize {
        self.config_bytes.len() / 8
    }

    pub fn configs(&self) -> impl Iterator<Item = (u32, u32)> + 'a {
        self.config_bytes.as_chunks::<8>().0.iter().map(|chunk| {
            let reg = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let val = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            (reg, val)
        })
    }

    pub fn reg(&self, reg: u32) -> Option<u32> {
        self.configs().find(|(r, _)| *r == reg).map(|(_, v)| v)
    }

    pub const fn code(&self) -> &'a [u8] {
        self.code
    }

    pub fn to_owned(&self) -> ShaderBlobOwned {
        ShaderBlobOwned {
            stage: self.stage,
            flags: self.flags,
            user_sgprs: self.user_sgprs,
            param_exports: self.param_exports,
            config: self.configs().collect(),
            code: self.code.to_vec(),
        }
    }
}

/// Owned shader blob representation for build-time emission and runtime caches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderBlobOwned {
    pub stage: ShaderStage,
    pub flags: u8,
    pub user_sgprs: u16,
    pub param_exports: u16,
    pub config: Vec<(u32, u32)>,
    pub code: Vec<u8>,
}

impl ShaderBlobOwned {
    pub fn reg(&self, reg: u32) -> Option<u32> {
        self.config.iter().find(|(r, _)| *r == reg).map(|(_, v)| *v)
    }

    pub fn encode(&self) -> Vec<u8> {
        encode_blob(
            self.stage,
            self.flags,
            self.user_sgprs,
            self.param_exports,
            &self.config,
            &self.code,
        )
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, BlobError> {
        Ok(ShaderBlob::parse(bytes)?.to_owned())
    }
}

pub fn encode_blob(
    stage: ShaderStage,
    flags: u8,
    user_sgprs: u16,
    param_exports: u16,
    configs: &[(u32, u32)],
    code: &[u8],
) -> Vec<u8> {
    assert!(
        code.len().is_multiple_of(4),
        "shader code must be 4-byte aligned"
    );
    let config_count = u32::try_from(configs.len()).expect("config count fits u32");
    let code_len = u32::try_from(code.len()).expect("code length fits u32");
    let mut out = Vec::with_capacity(HEADER_SIZE + configs.len() * 8 + code.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.push(stage as u8);
    out.push(flags);
    out.extend_from_slice(&user_sgprs.to_le_bytes());
    out.extend_from_slice(&param_exports.to_le_bytes());
    out.extend_from_slice(&config_count.to_le_bytes());
    out.extend_from_slice(&code_len.to_le_bytes());
    for &(reg, val) in configs {
        out.extend_from_slice(&reg.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    }
    out.extend_from_slice(code);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn round_trip_shader_blob() {
        let original = ShaderBlobOwned {
            stage: ShaderStage::Pixel,
            flags: FLAG_USES_KILL,
            user_sgprs: 7,
            param_exports: 0,
            config: vec![(0xB028, 0x0010_0001), (0x286CC, 0x2), (0x286E8, 0)],
            code: vec![0x01, 0x02, 0x03, 0x04, 0x10, 0x20, 0x30, 0x40],
        };
        let bytes = original.encode();
        let parsed = ShaderBlob::parse(&bytes).expect("parse encoded blob");
        assert_eq!(parsed.stage(), ShaderStage::Pixel);
        assert!(parsed.uses_kill());
        assert_eq!(parsed.user_sgprs(), 7);
        assert_eq!(parsed.param_exports(), 0);
        assert_eq!(parsed.reg(0xB028), Some(0x0010_0001));
        assert_eq!(parsed.reg(0x286CC), Some(0x2));
        assert_eq!(parsed.reg(0xFFFF), None);
        assert_eq!(parsed.code(), original.code.as_slice());
        assert_eq!(parsed.to_owned(), original);
    }

    #[test]
    fn rejects_invalid_blobs() {
        assert_eq!(
            ShaderBlob::parse(&[0; 10]).unwrap_err(),
            BlobError::TooShort
        );

        let mut bad_magic = encode_blob(ShaderStage::Vertex, 0, 2, 3, &[], &[0; 4]);
        bad_magic[0] = b'X';
        assert_eq!(
            ShaderBlob::parse(&bad_magic).unwrap_err(),
            BlobError::BadMagic
        );

        let mut bad_ver = encode_blob(ShaderStage::Vertex, 0, 2, 3, &[], &[0; 4]);
        bad_ver[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(
            ShaderBlob::parse(&bad_ver).unwrap_err(),
            BlobError::UnsupportedVersion(99)
        );

        let mut bad_stage = encode_blob(ShaderStage::Vertex, 0, 2, 3, &[], &[0; 4]);
        bad_stage[6] = 7;
        assert_eq!(
            ShaderBlob::parse(&bad_stage).unwrap_err(),
            BlobError::InvalidStage(7)
        );

        let mut truncated = encode_blob(ShaderStage::Vertex, 0, 2, 3, &[(1, 2)], &[0; 8]);
        truncated.pop();
        assert_eq!(
            ShaderBlob::parse(&truncated).unwrap_err(),
            BlobError::Truncated
        );

        let mut trailing = encode_blob(ShaderStage::Vertex, 0, 2, 3, &[], &[0; 4]);
        trailing.push(0);
        assert_eq!(
            ShaderBlob::parse(&trailing).unwrap_err(),
            BlobError::TrailingBytes
        );
    }
}
