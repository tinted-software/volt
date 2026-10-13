use std::time::Instant;
use volt_emulator::aarch64::compile::compile;

fn block(n: usize, seed: u32) -> Vec<u8> {
    let mut w = Vec::new();
    for i in 0..n as u32 {
        let r = (i + seed) % 8;
        w.push(match i % 4 {
            0 => 0x9100_0400 | ((seed & 0x3ff) << 10) | r,
            1 => 0x8b03_0041 | (r << 5),
            2 => 0xd280_0000 | ((seed & 0xffff) << 5) | r,
            3 => 0xca02_0021,
            _ => 0xea02_0021 | (r << 5), // ands x1,x1,x2
        });
    }
    w.push(0x1400_0000); // b .
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    dispatch();
    {
        use volt_emulator::aarch64::host::ends_block;
        use volt_isa_aarch64::decode::decode;
        let b = block(63, 1);
        let n = b
            .chunks_exact(4)
            .take_while(|c| {
                let word = u32::from_le_bytes((*c).try_into().unwrap());
                !ends_block(&decode(word).unwrap(), false)
            })
            .count();
        println!("non-terminating prefix of 63-block: {n}");
    }
    for n in [4usize, 16, 63] {
        let iters = 2000;
        let blocks: Vec<_> = (0..iters).map(|s| block(n, s as u32)).collect();
        let t = Instant::now();
        for (s, b) in blocks.iter().enumerate() {
            std::hint::black_box(compile(0x1000 + s as u64 * 4, b).unwrap());
        }
        let e = t.elapsed();
        println!(
            "{n:>3} insns: {:>8.1} us/block",
            e.as_secs_f64() * 1e6 / iters as f64
        );
    }
}

#[allow(dead_code)]
pub fn dispatch() {
    use volt_emulator::{Cpu, aarch64::Cache};
    for (label, pc) in [
        ("fast path (in-page)", 0x1000u64),
        ("slow path (page-crossing)", 0x1ff0u64),
    ] {
        for n in [4usize, 63] {
            let code = block(n, 1);
            let mut cache = Cache::new();
            let mut cpu = Cpu::default();
            let fetch = |_: &Cpu, at: u64, out: &mut [u8]| {
                let o = (at - pc) as usize;
                out.copy_from_slice(&code[o..o + out.len()]);
                Ok(out.len())
            };
            cpu.pc = pc;
            cache.run_block(&mut cpu, fetch).unwrap();
            let iters = 200_000;
            let t = Instant::now();
            for _ in 0..iters {
                cpu.pc = pc;
                std::hint::black_box(cache.run_block(&mut cpu, fetch).unwrap());
            }
            println!(
                "{label:>26} {n:>3} insns: {:>7.0} ns/dispatch",
                t.elapsed().as_nanos() as f64 / iters as f64
            );
        }
    }
}
