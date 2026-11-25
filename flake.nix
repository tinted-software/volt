{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "riscv64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        with pkgs;
        {
          default = rustPlatform.buildRustPackage {
            pname = "volt";
            version = "0.1.0";
            src = self;
            cargoLock.lockFile = ./Cargo.lock;
          };
        }
      );
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          moldStdenv = pkgs.stdenvAdapters.useMoldLinker pkgs.clangStdenv;
          mkShell = pkgs.mkShell.override { stdenv = moldStdenv; };
        in
        {
          default = mkShell {
            name = "volt-devshell";
            packages = with pkgs; [
              rustc
              cargo
              clippy
              rustfmt
              rust-analyzer
              cargo-nextest
              qemu
            ];
          };
        }
      );
    };
}
