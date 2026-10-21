// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "volt",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "VoltIR", targets: ["VoltIR"]),
        .library(name: "VoltTarget", targets: ["VoltTarget"]),
        .library(name: "VoltEmulator", targets: ["VoltEmulator"]),
        .executable(name: "volt-jit", targets: ["VoltJIT"]),
        .executable(name: "volt-emulate", targets: ["VoltEmulate"]),
    ],
    traits: [
        .trait(name: "X86_64", description: "Enable the x86_64 encoder, JIT, and linker."),
        .trait(name: "AArch64", description: "Enable the AArch64 disassembler."),
        .trait(name: "ARMv7", description: "Enable the ARMv7 A32 decoder and disassembler."),
        .trait(
            name: "ARMv7Emulator",
            description: "Enable the ARMv7 A32 emulator and its decoder dependency.",
            enabledTraits: ["ARMv7", "X86_64"]),
        .default(enabledTraits: ["X86_64", "AArch64"]),
    ],
    targets: [
        .target(name: "VoltIR", dependencies: []),
        .target(name: "VoltTarget", dependencies: ["VoltIR"]),
        .target(name: "VoltEmulator", dependencies: ["VoltIR", "VoltTarget"]),
        .executableTarget(name: "VoltEmulate", dependencies: [
            .target(name: "VoltEmulator", condition: .when(traits: ["ARMv7Emulator"]))
        ]),
        .executableTarget(name: "VoltJIT", dependencies: ["VoltTarget", "VoltIR"]),
        .testTarget(name: "VoltTests", dependencies: [
            "VoltTarget", "VoltIR",
            .target(name: "VoltEmulator", condition: .when(traits: ["ARMv7Emulator"])),
        ]),
    ]
)
