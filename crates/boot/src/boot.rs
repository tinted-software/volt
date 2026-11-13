//! Raw Linux arm64 Image, or XNU Mach-O kernel, boot through Volt's full-system JIT.

use core::time::Duration;
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};
use volt_emulator::system::{
    DEFAULT_CMDLINE, DEFAULT_XNU_CMDLINE, IdleMode, RAM_SIZE, SystemConfig, XNU_RAM_SIZE,
    smp::SmpConfig,
};
use volt_host::{DumpSurface, FileBackend, StdoutConsole, WindowSurface};
use volt_platform::{Console, NullSurface, Surface};
use winnow_args::{Args, ValueEnum};

#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[arg(rename_all = "kebab-case")]
enum IdleModeArg {
    #[default]
    Paced,
    #[arg(alias = "fast-forward")]
    Fast,
    Spin,
}

impl From<IdleModeArg> for IdleMode {
    fn from(arg: IdleModeArg) -> Self {
        match arg {
            IdleModeArg::Paced => IdleMode::Paced,
            IdleModeArg::Fast => IdleMode::FastForward,
            IdleModeArg::Spin => IdleMode::Spin,
        }
    }
}

#[derive(Args, Debug)]
#[arg(
    name = "volt-boot",
    about = "AArch64 Linux or XNU boot with PL011, GICv2/GICv3, and VirtIO-Blk (Linux)"
)]
struct Cli {
    /// Path to a raw Linux AArch64 Image, or an XNU kernel (Mach-O).
    #[arg(positional)]
    image: PathBuf,

    /// Timeout in seconds (default: 20).
    #[arg(positional)]
    seconds: Option<u64>,

    /// Kernel command line (Linux) or boot-args (XNU; start with a non-dash word).
    #[arg(positional)]
    cmdline: Option<String>,

    /// Disk image for virtio-blk rootfs (e.g. rootfs.ext4, disk.img).
    #[arg(short, long, alias = "drive", alias = "rootfs")]
    disk: Option<PathBuf>,

    /// Initial ramdisk image (e.g. initramfs.cpio, initrd.img).
    #[arg(short, long, alias = "initramfs")]
    initrd: Option<PathBuf>,

    /// Guest RAM size (e.g. 128M, 256M, 512M, 1G).
    #[arg(short, long, alias = "ram")]
    memory: Option<String>,

    /// Number of virtual CPUs (default: 1, max: 8).
    #[arg(long, alias = "smp", default = "1")]
    cpus: u32,

    /// CPU idle policy on WFI (paced, fast, spin).
    #[arg(long)]
    idle: Option<IdleModeArg>,

    /// Alias for --idle=paced.
    #[arg(long)]
    paced: bool,

    /// Alias for --idle=fast.
    #[arg(long, alias = "fast-forward")]
    fast: bool,

    /// Alias for --idle=spin.
    #[arg(long)]
    spin: bool,

    /// Enable VirtIO-GPU 2D scanout device.
    #[arg(long)]
    gpu: bool,

    /// Display backend for VirtIO-GPU: null, dump, dump:<path>, or window.
    #[arg(long)]
    display: Option<String>,
}

fn main() -> ExitCode {
    // Normalize "-smp=N" or "-smp N" to "--smp=N" for winnow-args
    let raw_args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut args: Vec<std::ffi::OsString> = Vec::with_capacity(raw_args.len());
    for arg in raw_args {
        if let Some(s) = arg.to_str() {
            if let Some(rest) = s.strip_prefix("-smp=") {
                args.push(format!("--smp={rest}").into());
                continue;
            } else if s == "-smp" {
                args.push("--smp".into());
                continue;
            }
        }
        args.push(arg);
    }

    let cli = match Cli::parse_from(&args) {
        Ok(cli) => cli,
        Err(err) => {
            let code = winnow_args::report(&err, &Cli::program());
            return ExitCode::from(code as u8);
        }
    };

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("volt-boot: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let idle_mode = if cli.spin {
        IdleMode::Spin
    } else if cli.fast {
        IdleMode::FastForward
    } else if cli.paced {
        IdleMode::Paced
    } else if let Some(idle) = cli.idle {
        idle.into()
    } else {
        IdleMode::Paced
    };

    let seconds = cli.seconds.unwrap_or(20);
    let image = std::fs::read(&cli.image).map_err(|e| format!("{}: {e}", cli.image.display()))?;
    // A Mach-O image is an XNU kernel; anything else is a raw Linux Image.
    let xnu = volt_emulator::system::xnu::is_macho(&image);
    let default_cmdline = if xnu {
        DEFAULT_XNU_CMDLINE
    } else if cli.disk.is_some() {
        "root=/dev/vda rw console=ttyAMA0 earlycon=pl011,0x9000000 nokaslr loglevel=8"
    } else {
        DEFAULT_CMDLINE
    };
    let cmdline = cli.cmdline.unwrap_or_else(|| default_cmdline.to_string());

    let initrd_data = match &cli.initrd {
        Some(path) => Some(load_initrd(path)?),
        None => None,
    };

    let default_ram = if let Some(initrd) = &initrd_data {
        RAM_SIZE.max((initrd.len() * 3 + (128 << 20)).next_power_of_two())
    } else if xnu {
        XNU_RAM_SIZE
    } else {
        RAM_SIZE
    };

    let ram_size = match cli.memory.as_deref() {
        Some(s) => parse_memory_size(s)?,
        None => default_ram,
    };

    let disk_backend = match &cli.disk {
        Some(path) => Some(FileBackend::open(path)?),
        None => None,
    };

    let gpu_surface = build_gpu_surface(cli.gpu, cli.display.as_deref())?;

    if cli.cpus > 1 {
        let config = SmpConfig {
            cpus: cli.cpus,
            timeout: Duration::from_secs(seconds),
            cmdline,
            idle_mode,
            ram_size,
            initrd: initrd_data,
            disk: disk_backend,
            gpu: gpu_surface,
        };
        let mut output = StdoutConsole::new();
        let report = volt_emulator::system::smp::boot_smp(&image, config, move |byte| {
            output.write_byte(byte);
        })
        .map_err(|e| format!("{e}"))?;
        println!("\n{report}");
        return Ok(());
    }

    let config = SystemConfig {
        timeout: Duration::from_secs(seconds),
        cmdline,
        idle_mode,
        ram_size,
        initrd: initrd_data,
        disk: disk_backend,
        gpu: gpu_surface,
    };
    let mut output = StdoutConsole::new();
    let report = volt_emulator::system::boot_system(&image, config, move |byte| {
        output.write_byte(byte);
    })
    .map_err(|e| format!("{e}"))?;
    println!("\n{report}");
    Ok(())
}

fn build_gpu_surface(
    gpu_flag: bool,
    display: Option<&str>,
) -> Result<Option<Box<dyn Surface>>, String> {
    let spec = match (gpu_flag, display) {
        (false, None) => return Ok(None),
        (true, None) => "null",
        (_, Some(spec)) => spec,
    };
    if spec == "null" {
        Ok(Some(Box::new(NullSurface)))
    } else if spec == "dump" {
        Ok(Some(Box::new(DumpSurface::new("volt_scanout.ppm"))))
    } else if let Some(path) = spec
        .strip_prefix("dump:")
        .or_else(|| spec.strip_prefix("dump="))
    {
        Ok(Some(Box::new(DumpSurface::new(path))))
    } else if spec == "window" {
        let win = WindowSurface::open("Volt VirtIO-GPU", 1024, 768)?;
        Ok(Some(Box::new(win)))
    } else {
        Err(format!(
            "unknown --display mode `{spec}` (expected null, dump[:path], or window)"
        ))
    }
}

fn load_initrd(path: &Path) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() >= 4 && bytes[..4] == [0x28, 0xb5, 0x2f, 0xfd] {
        zrip::decompress(&bytes).map_err(|e| format!("zstd decompression error: {e}"))
    } else {
        Ok(bytes)
    }
}

/// Parse human-readable memory string (e.g. "128M", "512MB", "1G", "2GiB", "65536").
pub fn parse_memory_size(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("memory size cannot be empty".to_string());
    }
    let (digits, suffix) = match s.find(|c: char| !c.is_ascii_digit()) {
        Some(idx) => (&s[..idx], &s[idx..]),
        None => (s, ""),
    };
    let num: usize = digits
        .parse()
        .map_err(|e| format!("invalid number in memory size '{s}': {e}"))?;
    let multiplier = match suffix.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        other => return Err(format!("unknown memory unit '{other}' in '{s}'")),
    };
    Ok(num * multiplier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_emulator::devices::BlockBackend;

    #[test]
    fn test_parse_memory_size() {
        assert_eq!(parse_memory_size("128M").unwrap(), 128 << 20);
        assert_eq!(parse_memory_size("256mb").unwrap(), 256 << 20);
        assert_eq!(parse_memory_size("1G").unwrap(), 1 << 30);
        assert_eq!(parse_memory_size("2GiB").unwrap(), 2 << 30);
        assert_eq!(parse_memory_size("1048576").unwrap(), 1048576);
    }

    #[test]
    fn test_file_backend() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("volt_test_disk.img");
        std::fs::write(&test_file, [0xaa; 1024]).unwrap();

        let mut backend = FileBackend::open(&test_file).unwrap();
        assert_eq!(backend.len(), 1024);

        let mut buf = [0u8; 4];
        backend.read_at(0, &mut buf).unwrap();
        assert_eq!(buf, [0xaa; 4]);

        backend.write_at(512, &[0xbb; 4]).unwrap();
        backend.read_at(512, &mut buf).unwrap();
        assert_eq!(buf, [0xbb; 4]);

        std::fs::remove_file(&test_file).unwrap();
    }

    #[test]
    fn test_winnow_args_disk_and_initrd() {
        let cli = Cli::parse_from(["--disk=/var/rootfs.img", "Image"]).unwrap();
        assert_eq!(cli.disk, Some(PathBuf::from("/var/rootfs.img")));

        let cli = Cli::parse_from(["-d", "disk.img", "Image"]).unwrap();
        assert_eq!(cli.disk, Some(PathBuf::from("disk.img")));

        let cli = Cli::parse_from(["--rootfs", "root.img", "Image"]).unwrap();
        assert_eq!(cli.disk, Some(PathBuf::from("root.img")));

        let cli = Cli::parse_from(["--initrd=initramfs.cpio", "Image"]).unwrap();
        assert_eq!(cli.initrd, Some(PathBuf::from("initramfs.cpio")));
    }

    #[test]
    fn test_zstd_decompression() {
        let data = b"sample cpio content for zstd test";
        let compressed = zrip::compress(data, 1).unwrap();
        let path = std::env::temp_dir().join("volt_test_zstd.bin");
        std::fs::write(&path, &compressed).unwrap();
        let decompressed = load_initrd(&path).unwrap();
        assert_eq!(&decompressed, data);
        let _ = std::fs::remove_file(&path);
    }
}
