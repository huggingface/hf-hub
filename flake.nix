{
  inputs = {
    crate2nix = {
      url = "github:nix-community/crate2nix";
    };
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
    };
  };
  outputs = {
    self,
    crate2nix,
    nixpkgs,
    flake-utils,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        cargoNix = crate2nix.tools.${system}.appliedCargoNix {
          name = "hf-hub";
          src = ./.;
        };
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
          ];
        };
        hf-hub = cargoNix.rootCrate.build;
      in {
        devShells = with pkgs; rec {
          default = pure;

          pure = mkShell {
            buildInputs = [
              hf-hub
            ];
          };

          impure = mkShell {
            buildInputs = [
              openssl.dev
              pkg-config
              (rust-bin.stable.latest.default.override {
                extensions = [
                  "rust-analyzer"
                  "rust-src"
                ];
              })
            ];

            postShellHook = ''
              export PATH=$PATH:~/.cargo/bin
            '';
          };
        };
      }
    );
}
