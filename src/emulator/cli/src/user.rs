//! Linux AArch64 ELF execution using Volt's native host JIT.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::ExitCode,
};
use winnow_args::Args;

#[derive(Args, Debug)]
#[arg(
    name = "volt-aarch64",
    about = "Linux AArch64 user-mode execution through the Volt JIT (scalar instructions only)."
)]
struct UserCli {
    /// AArch64 ELF binary to execute.
    #[arg(positional, stop_flags)]
    program: PathBuf,

    /// Arguments passed to the target binary.
    #[arg(positional)]
    args: Vec<OsString>,
}

fn main() -> ExitCode {
    let cli = match UserCli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            let code = winnow_args::report(&err, &UserCli::program());
            return ExitCode::from(code as u8);
        }
    };

    let mut full_args = Vec::with_capacity(cli.args.len() + 1);
    full_args.push(cli.program.as_os_str().to_os_string());
    full_args.extend(cli.args);

    let env: Vec<_> = std::env::vars_os().collect();
    match volt_emulator::user::run(Path::new(&cli.program), &full_args, &env) {
        Ok(status) => ExitCode::from(status),
        Err(error) => {
            eprintln!("volt-aarch64: {}: {error}", cli.program.display());
            ExitCode::FAILURE
        }
    }
}
