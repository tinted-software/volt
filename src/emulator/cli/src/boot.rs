//! Raw Linux arm64 Image boot through Volt's full-system JIT.

use std::{io::Write, process::ExitCode, time::Duration};

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
    let mut idle_mode = volt_emulator::system::IdleMode::default();
    let mut cpus = 1u32;
    let mut positionals = Vec::new();
    for arg in args {
        let text = match arg.to_str() {
            Some(t) => t,
            None => return Err("argument must be UTF-8".into()),
        };
        if text == "--help" {
            println!(
                "usage: volt-boot [OPTIONS] IMAGE [SECONDS] [CMDLINE]\n\
                 AArch64 Linux boot with 128 MiB RAM, PL011, and GICv2.\n\n\
                 Options:\n\
                   --cpus=N, -smp=N         Number of virtual CPUs (default: 1, max: 8)\n\
                   --idle=paced|fast|spin   CPU idle policy on WFI (default: paced)\n\
                   --paced                  Alias for --idle=paced\n\
                   --fast, --fast-forward   Alias for --idle=fast\n\
                   --spin                   Alias for --idle=spin"
            );
            return Ok(());
        } else if let Some(mode) = text.strip_prefix("--idle=") {
            idle_mode = match mode {
                "paced" => volt_emulator::system::IdleMode::Paced,
                "fast" | "fast-forward" => volt_emulator::system::IdleMode::FastForward,
                "spin" => volt_emulator::system::IdleMode::Spin,
                _ => return Err("invalid --idle mode: choose paced, fast, or spin".into()),
            };
        } else if text == "--paced" {
            idle_mode = volt_emulator::system::IdleMode::Paced;
        } else if text == "--fast" || text == "--fast-forward" {
            idle_mode = volt_emulator::system::IdleMode::FastForward;
        } else if text == "--spin" {
            idle_mode = volt_emulator::system::IdleMode::Spin;
        } else if let Some(count) = text.strip_prefix("--cpus=") {
            cpus = count.parse::<u32>()?;
        } else if let Some(count) = text.strip_prefix("-smp=") {
            cpus = count.parse::<u32>()?;
        } else {
            positionals.push(text.to_string());
        }
    }
    if positionals.is_empty() || positionals.len() > 3 {
        return Err("usage: volt-boot [OPTIONS] IMAGE [SECONDS] [CMDLINE]".into());
    }
    let seconds = match positionals.get(1) {
        Some(value) => value.parse::<u64>()?,
        None => 20,
    };
    let cmdline = match positionals.get(2) {
        Some(value) => value.as_str(),
        None => "console=ttyAMA0 earlycon=pl011,0x9000000 nokaslr loglevel=8",
    };
    let image = std::fs::read(&positionals[0])?;
    if cpus > 1 {
        let config = volt_emulator::system::smp::SmpConfig {
            cpus,
            timeout: Duration::from_secs(seconds),
            cmdline: cmdline.to_string(),
            idle_mode,
        };
        let mut output = std::io::BufWriter::new(std::io::stdout());
        let report = volt_emulator::system::smp::boot_smp(&image, config, move |byte| {
            let _ = output.write_all(&[byte]);
            if byte == b'\n' {
                let _ = output.flush();
            }
        })?;
        println!("\n{report}");
        return Ok(());
    }
    let report = volt_emulator::system::boot_options(
        &image,
        Duration::from_secs(seconds),
        cmdline,
        idle_mode,
    )?;
    println!("\n{report}");
    Ok(())
}
