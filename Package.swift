// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "volt",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "VoltIR", targets: ["VoltIR"]),
        .library(name: "VoltTarget", targets: ["VoltTarget"]),
        .library(name: "VoltTargetAArch64", targets: ["VoltTargetAArch64"]),
        .library(name: "VoltTargetARMv7", targets: ["VoltTargetARMv7"]),
        .library(name: "VoltTargetX86_64", targets: ["VoltTargetX86_64"]),
        .library(name: "VoltEmulator", targets: ["VoltEmulator"]),
        .executable(name: "volt-jit", targets: ["VoltJIT"]),
        .executable(name: "volt-emulate", targets: ["VoltEmulate"]),
    ],
    targets: [
        .target(name: "VoltIR", dependencies: []),
        .target(name: "VoltTarget", dependencies: ["VoltIR"]),
        .target(name: "VoltTargetAArch64", dependencies: ["VoltTarget", "VoltIR"]),
        .target(name: "VoltTargetARMv7", dependencies: ["VoltTarget"]),
        .target(name: "VoltTargetX86_64", dependencies: ["VoltTarget", "VoltIR"]),
        .target(name: "VoltEmulator", dependencies: ["VoltIR", "VoltTarget", "VoltTargetARMv7", "VoltTargetX86_64"]),
        .executableTarget(name: "VoltEmulate", dependencies: ["VoltEmulator"]),
        .executableTarget(name: "VoltJIT", dependencies: ["VoltTarget", "VoltTargetX86_64", "VoltIR"]),
        .testTarget(name: "VoltTests", dependencies: ["VoltTargetAArch64", "VoltTargetARMv7", "VoltTargetX86_64", "VoltIR", "VoltEmulator"]),
    ]
)
