use core::f32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Format {
    Bf16 = 0,
    F8E4M3 = 1,
    F8E5M2 = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InvalidPayload;

impl Format {
    pub fn payload_bits(self) -> u16 {
        match self {
            Format::Bf16 => 16,
            Format::F8E4M3 | Format::F8E5M2 => 8,
        }
    }
}

pub fn decode(format: Format, payload: u16) -> Result<f32, InvalidPayload> {
    match format {
        Format::Bf16 => Ok(f32::from_bits((payload as u32) << 16)),
        Format::F8E4M3 => Ok(decode_e4m3(payload8(payload)?)),
        Format::F8E5M2 => Ok(decode_e5m2(payload8(payload)?)),
    }
}

pub fn encode(format: Format, value: f32) -> u16 {
    match format {
        Format::Bf16 => encode_bf16(value),
        Format::F8E4M3 => encode_e4m3(value) as u16,
        Format::F8E5M2 => encode_e5m2(value) as u16,
    }
}

fn payload8(payload: u16) -> Result<u8, InvalidPayload> {
    if payload > u8::MAX as u16 {
        return Err(InvalidPayload);
    }
    Ok(payload as u8)
}

fn encode_bf16(value: f32) -> u16 {
    let source = value.to_bits();
    let exponent = source & 0x7f80_0000;
    let fraction = source & 0x007f_ffff;
    let mut retained = (source >> 16) as u16;
    if exponent == 0x7f80_0000 {
        if fraction != 0 {
            retained |= 0x0040;
        }
        return retained;
    }
    ((source + 0x7fff + (retained & 1) as u32) >> 16) as u16
}

fn encode_e4m3(value: f32) -> u8 {
    let source = value.to_bits();
    let sign = ((source >> 24) & 0x80) as u8;
    let source_exponent = ((source >> 23) & 0xff) as u8;
    let source_fraction = source & 0x007f_ffff;
    if source_exponent == 0xff {
        if source_fraction != 0 {
            return sign | 0x7f;
        }
        return sign | 0x7e;
    }
    if source_exponent == 0 {
        return sign;
    }
    let exponent = source_exponent as i16 - 127;
    let significand = 0x0080_0000 | source_fraction;
    let magnitude: u32 = if exponent < -6 {
        let shift = (-9 - exponent + 23) as u8;
        round_right(significand, shift)
    } else {
        let mut rounded = round_right(significand, 20);
        let mut target_exponent = exponent;
        if rounded == 16 {
            rounded = 8;
            target_exponent += 1;
        }
        (target_exponent + 7) as u32 * 8 + (rounded - 8)
    };
    sign | magnitude.min(0x7e) as u8
}

fn decode_e4m3(payload: u8) -> f32 {
    let sign = ((payload & 0x80) as u32) << 24;
    let exponent = (payload >> 3) & 0x0f;
    let fraction = payload & 0x07;
    if exponent == 0x0f && fraction == 0x07 {
        return f32::from_bits(sign | 0x7fc0_0000);
    }
    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        // Zig `@clz` counts within u8: leading = 7 - clz8(fraction).
        let leading: u32 = 7 - fraction.leading_zeros();
        let f32_exponent = 118 + leading;
        let remainder = fraction as u32 - (1 << leading);
        return f32::from_bits(sign | (f32_exponent << 23) | (remainder << (23 - leading)));
    }
    let f32_exponent = exponent as u32 + 120;
    f32::from_bits(sign | (f32_exponent << 23) | ((fraction as u32) << 20))
}

fn encode_e5m2(value: f32) -> u8 {
    let source = value.to_bits();
    let sign = ((source >> 24) & 0x80) as u8;
    let source_exponent = ((source >> 23) & 0xff) as u8;
    let source_fraction = source & 0x007f_ffff;
    if source_exponent == 0xff {
        if source_fraction == 0 {
            return sign | 0x7c;
        }
        let payload = (source_fraction >> 21) as u8;
        return sign | 0x7c | payload | 0x02;
    }
    if source_exponent == 0 {
        return sign;
    }
    let exponent = source_exponent as i16 - 127;
    let significand = 0x0080_0000 | source_fraction;
    let magnitude: u32 = if exponent < -14 {
        let shift = (-16 - exponent + 23) as u8;
        round_right(significand, shift)
    } else {
        let mut rounded = round_right(significand, 21);
        let mut target_exponent = exponent;
        if rounded == 8 {
            rounded = 4;
            target_exponent += 1;
        }
        (target_exponent + 15) as u32 * 4 + (rounded - 4)
    };
    sign | magnitude.min(0x7b) as u8
}

fn decode_e5m2(payload: u8) -> f32 {
    let sign = ((payload & 0x80) as u32) << 24;
    let exponent = (payload >> 2) & 0x1f;
    let fraction = payload & 0x03;
    if exponent == 0x1f {
        if fraction == 0 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        return f32::from_bits(sign | 0x7f80_0000 | (((fraction | 0x02) as u32) << 21));
    }
    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        let leading: u32 = 7 - fraction.leading_zeros();
        let f32_exponent = 111 + leading;
        let remainder = fraction as u32 - (1 << leading);
        return f32::from_bits(sign | (f32_exponent << 23) | (remainder << (23 - leading)));
    }
    let f32_exponent = exponent as u32 + 112;
    f32::from_bits(sign | (f32_exponent << 23) | ((fraction as u32) << 21))
}

fn round_right(value: u32, shift: u8) -> u32 {
    if shift > 24 {
        return 0;
    }
    if shift == 0 {
        return value;
    }
    let retained = value >> shift;
    let mask = (1u32 << shift) - 1;
    let discarded = value & mask;
    let halfway = 1u32 << (shift - 1);
    retained + u32::from(discarded > halfway || (discarded == halfway && retained & 1 != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_boundaries() {
        let cases: &[(u32, u16)] = &[
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            (0x3f80_0000, 0x3f80),
            (0x3f80_8001, 0x3f81),
            (0x7f80_0000, 0x7f80),
            (0x7f81_2345, 0x7fc1),
        ];
        for (source, expected) in cases {
            assert_eq!(
                *expected,
                encode(Format::Bf16, f32::from_bits(*source)),
                "{source:08x}"
            );
        }
    }

    #[test]
    fn bf16_round_trip() {
        for payload in [0x0000u16, 0x3f80, 0x8000, 0x7f80, 0xff80] {
            let v = decode(Format::Bf16, payload).unwrap();
            assert_eq!(payload, encode(Format::Bf16, v));
        }
    }

    #[test]
    fn e4m3_e5m2_decode_spot() {
        // 0x38 e4m3 = 1.0, 0x3c e5m2 = 1.0.
        assert_eq!(
            1.0f32.to_bits(),
            decode(Format::F8E4M3, 0x38).unwrap().to_bits()
        );
        assert_eq!(
            1.0f32.to_bits(),
            decode(Format::F8E5M2, 0x3c).unwrap().to_bits()
        );
        assert_eq!(0u32, decode(Format::F8E4M3, 0x00).unwrap().to_bits());
        assert_eq!(0u32, decode(Format::F8E5M2, 0x00).unwrap().to_bits());
    }
}
