//! How much of an AArch64 image's executable sections the decoder recognises, and
//! which encodings it does not (the most frequent first, bucketed by opcode bits).
//! Formatting is checked separately by the `fmtdiff` example.
use std::collections::HashMap;
use volt_disasm::Image;
use volt_isa_aarch64::decode::decode;

fn main() {
    let path = std::env::args().nth(1).expect("usage: coverage <binary>");
    let img = Image::parse(&std::fs::read(path).unwrap()).unwrap();
    let (mut total, mut undecoded) = (0u64, 0u64);
    let mut buckets: HashMap<u32, (u64, u32)> = HashMap::new();
    for s in img.segments.iter().filter(|s| s.exec) {
        for c in s.data.chunks_exact(4) {
            let w = u32::from_le_bytes(c.try_into().unwrap());
            total += 1;
            if decode(w).is_err() {
                undecoded += 1;
                buckets.entry(w & 0xffe0_fc00).or_insert((0, w)).0 += 1;
            }
        }
    }
    println!("total {total} undecoded {undecoded}");
    let mut v: Vec<_> = buckets.into_iter().collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.1.0));
    for (_, (n, w)) in v.iter().take(40) {
        println!("{n:6} {w:08x}");
    }
}
