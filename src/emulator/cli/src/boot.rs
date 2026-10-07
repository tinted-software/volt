//! Raw Linux arm64 Image boot through Volt's full-system JIT with VirtIO-Blk.

use std::{
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};
use volt_emulator::devices::BlockBackend;
use volt_emulator::system::{DEFAULT_CMDLINE, IdleMode, RAM_SIZE, SystemConfig, smp::SmpConfig};
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
    about = "AArch64 Linux boot with PL011, GICv2, and VirtIO-Blk"
)]
struct Cli {
    /// Path to the raw Linux AArch64 Image.
    #[arg(positional)]
    image: PathBuf,

    /// Timeout in seconds (default: 20).
    #[arg(positional)]
    seconds: Option<u64>,

    /// Kernel command line.
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
}

/// Host file-backed block storage for virtio-blk, streaming sectors on demand.
pub struct FileBackend {
    file: std::fs::File,
    len: u64,
}

impl FileBackend {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .or_else(|_| std::fs::File::open(path))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        Ok(Self { file, len })
    }
}

impl BlockBackend for FileBackend {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), ()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset).map_err(|_| ())
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(offset)).map_err(|_| ())?;
            self.file.read_exact(buf).map_err(|_| ())
        }
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<(), ()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.write_all_at(buf, offset).map_err(|_| ())
        }
        #[cfg(not(unix))]
        {
            use std::io::{Seek, SeekFrom, Write};
            self.file.seek(SeekFrom::Start(offset)).map_err(|_| ())?;
            self.file.write_all(buf).map_err(|_| ())
        }
    }

    fn flush(&mut self) -> Result<(), ()> {
        self.file.sync_all().map_err(|_| ())
    }
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
    let default_cmdline = if cli.disk.is_some() {
        "root=/dev/vda rw console=ttyAMA0 earlycon=pl011,0x9000000 nokaslr loglevel=8"
    } else {
        DEFAULT_CMDLINE
    };
    let cmdline = cli.cmdline.unwrap_or_else(|| default_cmdline.to_string());

    let image = std::fs::read(&cli.image).map_err(|e| format!("{}: {e}", cli.image.display()))?;

    let initrd_data = match &cli.initrd {
        Some(path) => Some(load_initrd(path)?),
        None => None,
    };

    let default_ram = if let Some(initrd) = &initrd_data {
        RAM_SIZE.max((initrd.len() * 3 + (128 << 20)).next_power_of_two())
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

    if cli.cpus > 1 {
        let config = SmpConfig {
            cpus: cli.cpus,
            timeout: Duration::from_secs(seconds),
            cmdline,
            idle_mode,
            ram_size,
            initrd: initrd_data,
            disk: disk_backend,
        };
        let mut output = std::io::BufWriter::new(std::io::stdout());
        let report = volt_emulator::system::smp::boot_smp(&image, config, move |byte| {
            let _ = output.write_all(&[byte]);
            if byte == b'\n' {
                let _ = output.flush();
            }
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
    };
    let mut output = std::io::BufWriter::new(std::io::stdout());
    let report = volt_emulator::system::boot_system(&image, config, move |byte| {
        let _ = output.write_all(&[byte]);
        if byte == b'\n' {
            let _ = output.flush();
        }
    })
    .map_err(|e| format!("{e}"))?;
    println!("\n{report}");
    Ok(())
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
