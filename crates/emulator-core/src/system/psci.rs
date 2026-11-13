//! PSCI 0.2 firmware calls, following Mirage arm64/psci.zig.
pub const VERSION: u32 = 0x8400_0000;
pub const CPU_SUSPEND_32: u32 = 0x8400_0001;
pub const CPU_OFF: u32 = 0x8400_0002;
pub const CPU_ON_32: u32 = 0x8400_0003;
pub const MIGRATE_INFO_TYPE: u32 = 0x8400_0006;
pub const SYSTEM_OFF: u32 = 0x8400_0008;
pub const SYSTEM_RESET: u32 = 0x8400_0009;
pub const FEATURES: u32 = 0x8400_000a;
pub const CPU_ON: u32 = 0xc400_0003;
pub const CPU_SUSPEND: u32 = 0xc400_0001;
pub const NOT_SUPPORTED: u64 = u64::MAX;
pub const SUCCESS: u64 = 0;
pub const INVALID_PARAMETERS: u64 = (-2i64) as u64;
pub const DENIED: u64 = (-3i64) as u64;
pub const ALREADY_ON: u64 = (-4i64) as u64;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Value(u64),
    PowerOff,
    CpuOff,
    Reset,
    StartCpu {
        target: u64,
        entry: u64,
        context: u64,
    },
}
fn implemented(function: u32) -> bool {
    matches!(
        function,
        VERSION
            | SYSTEM_OFF
            | SYSTEM_RESET
            | FEATURES
            | MIGRATE_INFO_TYPE
            | CPU_ON
            | CPU_ON_32
            | CPU_OFF
    )
}
pub fn handle(function: u32, args: [u64; 3]) -> Outcome {
    match function {
        VERSION => Outcome::Value(2),
        SYSTEM_OFF => Outcome::PowerOff,
        SYSTEM_RESET => Outcome::Reset,
        MIGRATE_INFO_TYPE => Outcome::Value(2),
        FEATURES => Outcome::Value(if implemented(args[0] as u32) {
            SUCCESS
        } else {
            NOT_SUPPORTED
        }),
        CPU_ON | CPU_ON_32 => Outcome::StartCpu {
            target: args[0],
            entry: args[1],
            context: args[2],
        },
        CPU_OFF => Outcome::CpuOff,
        _ => Outcome::Value(NOT_SUPPORTED),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn firmware_features_match_actual_implemented_calls() {
        assert_eq!(handle(VERSION, [0; 3]), Outcome::Value(2));
        assert_eq!(
            handle(FEATURES, [SYSTEM_OFF as u64, 0, 0]),
            Outcome::Value(0)
        );
        assert_eq!(
            handle(FEATURES, [CPU_SUSPEND as u64, 0, 0]),
            Outcome::Value(NOT_SUPPORTED)
        );
        assert_eq!(handle(SYSTEM_OFF, [0; 3]), Outcome::PowerOff);
    }
}
