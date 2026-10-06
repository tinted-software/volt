//! Raw Linux arm64 Image boot through Volt's full-system JIT.

use std::{process::ExitCode, time::Duration};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("volt-boot: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn core::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--help") {
        println!(
            "usage: volt-boot IMAGE [SECONDS] [CMDLINE]\nOne-vCPU AArch64 Linux boot with 128 MiB RAM, PL011, and GICv2."
        );
        return Ok(());
    }
    if args.is_empty() || args.len() > 3 {
        return Err("usage: volt-boot IMAGE [SECONDS] [CMDLINE]".into());
    }
    let seconds = match args.get(1) {
        Some(value) => value
            .to_str()
            .ok_or("SECONDS must be UTF-8")?
            .parse::<u64>()?,
        None => 20,
    };
    let cmdline = match args.get(2) {
        Some(value) => value.to_str().ok_or("CMDLINE must be UTF-8")?,
        None => "console=ttyAMA0 earlycon=pl011,0x9000000 nokaslr loglevel=8",
    };
    let image = std::fs::read(&args[0])?;
    let report = volt_emulator::system::boot(&image, Duration::from_secs(seconds), cmdline)?;
    println!("\n{report}");
    Ok(())
}
