{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nix-community/naersk";
    nixpkgs.url = "nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  
  outputs = { self, nixpkgs, flake-utils , rust-overlay, naersk }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      overlays = [ rust-overlay.overlays.default ];
      pkgs = import nixpkgs { inherit overlays system; };
      rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      apple = pkgs.darwin.apple_sdk.frameworks;
      apple-deps = [ apple.AudioUnit apple.CoreAudio apple.CoreFoundation apple.CoreServices apple.SystemConfiguration apple.Security apple.DiskArbitration apple.Foundation pkgs.libiconv apple.AppKit apple.Cocoa ];
      linux-deps = [
          pkgs.udev pkgs.alsa-lib pkgs.vulkan-loader
          pkgs.xorg.libX11 pkgs.xorg.libXcursor pkgs.xorg.libXi
          pkgs.xorg.libXrandr pkgs.libxkbcommon pkgs.wayland

      ];

      buildInputs = [
          pkgs.wasm-bindgen-cli
          rust
          pkgs.autoconf
          pkgs.pkgconfig
          pkgs.openssl] ++ (if system == "aarch64-darwin" then apple-deps else linux-deps);

      naersk' = pkgs.callPackage naersk {};

    in
    {

      defaultPackage = naersk'.buildPackage {
        src = ./.;

        nativeBuildInputs = buildInputs;
        buildInputs = buildInputs;
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        COREAUDIO_SDK_PATH= if system == "aarch64-darwin" then "${pkgs.darwin.apple_sdk.MacOSX-SDK}" else "";
      };


      devShell = pkgs.mkShell rec {
        # buildInputs = buildInputs;
        buildInputs = [
          pkgs.wasm-bindgen-cli
          rust
          pkgs.autoconf
          pkgs.pkgconfig
          pkgs.openssl] ++ (if system == "aarch64-darwin" then apple-deps else linux-deps);

        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        COREAUDIO_SDK_PATH= if system == "aarch64-darwin" then "${pkgs.darwin.apple_sdk.MacOSX-SDK}" else "";
      };
    }
  );
}
