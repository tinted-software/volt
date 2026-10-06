use super::low_float;
use core::f32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum ScaleApplication {
    Multiply = 0,
    Divide = 1,
}

pub fn decode_e2m1(payload: u8) -> f32 {
    let nibble = payload & 0x0f;
    let sign = ((nibble & 0x08) as u32) << 28;
    let magnitude: [u32; 8] = [
        0x0000_0000,
        0x3f00_0000,
        0x3f80_0000,
        0x3fc0_0000,
        0x4000_0000,
        0x4040_0000,
        0x4080_0000,
        0x40c0_0000,
    ];
    f32::from_bits(sign | magnitude[(nibble & 0x07) as usize])
}

pub fn encode_e2m1(value: f32) -> u8 {
    let source = value.to_bits();
    let sign = ((source >> 28) & 0x08) as u8;
    let magnitude = source & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return 0x07;
    }
    if magnitude == 0x7f80_0000 {
        return sign | 0x07;
    }
    let result: u8 = if magnitude <= 0x3e80_0000 {
        0x00
    } else if magnitude < 0x3f40_0000 {
        0x01
    } else if magnitude <= 0x3fa0_0000 {
        0x02
    } else if magnitude < 0x3fe0_0000 {
        0x03
    } else if magnitude <= 0x4020_0000 {
        0x04
    } else if magnitude < 0x4060_0000 {
        0x05
    } else if magnitude <= 0x40a0_0000 {
        0x06
    } else {
        0x07
    };
    sign | result
}

fn apply(value: f32, scale: f32, application: ScaleApplication) -> f32 {
    match application {
        ScaleApplication::Multiply => value * scale,
        ScaleApplication::Divide => value / scale,
    }
}

fn undo(value: f32, scale: f32, application: ScaleApplication) -> f32 {
    match application {
        ScaleApplication::Multiply => value / scale,
        ScaleApplication::Divide => value * scale,
    }
}

pub fn dequantize(
    payload: u8,
    block_scale: u8,
    global_scale: f32,
    block_application: ScaleApplication,
    global_application: ScaleApplication,
) -> f32 {
    let decoded_scale = low_float::decode(low_float::Format::F8E4M3, block_scale as u16)
        .unwrap_or(f32::from_bits(0x7fc0_0000));
    let local = apply(decode_e2m1(payload), decoded_scale, block_application);
    apply(local, global_scale, global_application)
}

pub fn quantize(
    value: f32,
    block_scale: u8,
    global_scale: f32,
    block_application: ScaleApplication,
    global_application: ScaleApplication,
) -> u8 {
    let decoded_scale = low_float::decode(low_float::Format::F8E4M3, block_scale as u16)
        .unwrap_or(f32::from_bits(0x7fc0_0000));
    let global_unscaled = undo(value, global_scale, global_application);
    let local_unscaled = undo(global_unscaled, decoded_scale, block_application);
    encode_e2m1(local_unscaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_e2m1_payloads_round_trip() {
        let positive: [u32; 8] = [
            0,
            0x3f00_0000,
            0x3f80_0000,
            0x3fc0_0000,
            0x4000_0000,
            0x4040_0000,
            0x4080_0000,
            0x40c0_0000,
        ];
        for payload in 0..16u8 {
            let expected = positive[(payload & 7) as usize] | u32::from(payload & 8 != 0) << 31;
            assert_eq!(expected, decode_e2m1(payload).to_bits());
            assert_eq!(payload, encode_e2m1(decode_e2m1(payload)));
        }
    }

    #[test]
    fn tag_values_pinned() {
        assert_eq!(0u8, ScaleApplication::Multiply as u8);
        assert_eq!(1u8, ScaleApplication::Divide as u8);
    }
}
