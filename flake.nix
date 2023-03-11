{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  
  outputs = { self, nixpkgs, flake-utils , rust-overlay }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      overlays = [ rust-overlay.overlay ];
      pkgs = import nixpkgs { inherit overlays system; };
      rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      apple = pkgs.darwin.apple_sdk.frameworks;
      apple-deps = [ apple.Security apple.DiskArbitration apple.Foundation pkgs.libiconv ];
    in
    {
      defaultPackage = pkgs.rustPlatform.buildRustPackage {
        pname = "reuron";
        version = "0.1.0";
        src = ./.;

        cargoLock = {
          lockFile = ./Cargo.lock;
        };
      };
      devShell = pkgs.mkShell rec {
        buildInputs = [
          pkgs.wasm-bindgen-cli
          rust
          pkgs.autoconf
          pkgs.pkgconfig
          pkgs.openssl
          pkgs.udev pkgs.alsa-lib pkgs.vulkan-loader
          pkgs.xorg.libX11 pkgs.xorg.libXcursor pkgs.xorg.libXi
          pkgs.xorg.libXrandr pkgs.libxkbcommon pkgs.wayland
        ] ++ (if system == "aarch64-darwin" then apple-deps else []);
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
      };
    }
  );
}
