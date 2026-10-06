//! Linux AArch64 ELF execution using Volt's native host JIT.

use std::{ffi::OsString, path::Path, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    if args.is_empty() || args[0] == "--help" {
        eprintln!(
            "usage: volt-aarch64 PROGRAM [ARGUMENT...]\nLinux AArch64 user-mode execution through the Volt JIT (scalar instructions only)."
        );
        return ExitCode::from(u8::from(args.is_empty()));
    }
    let env: Vec<_> = std::env::vars_os().collect();
    match volt_emulator::user::run(Path::new(&args[0]), &args, &env) {
        Ok(status) => ExitCode::from(status),
        Err(error) => {
            eprintln!("volt-aarch64: {}: {error}", Path::new(&args[0]).display());
            ExitCode::FAILURE
        }
    }
}
