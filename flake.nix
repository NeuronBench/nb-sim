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
      rust = (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml).override {
        targets = [ "wasm32-unknown-unknown" ];
      };
      apple = pkgs.darwin.apple_sdk.frameworks;
      apple-deps = [ apple.AudioUnit apple.CoreAudio apple.CoreFoundation apple.CoreServices apple.SystemConfiguration apple.Security apple.DiskArbitration apple.Foundation apple.AppKit apple.Cocoa ];
      linux-deps = [
          pkgs.udev pkgs.alsaLib pkgs.vulkan-loader
          pkgs.xorg.libX11 pkgs.xorg.libXcursor pkgs.xorg.libXi
          pkgs.xorg.libXrandr pkgs.libxkbcommon pkgs.wayland

      ];

      buildInputs = [
          pkgs.wasm-bindgen-cli
          pkgs.trunk
          rust
          pkgs.autoconf
          pkgs.pkgconfig
          pkgs.openssl] ++ (if system == "aarch64-darwin" then apple-deps else linux-deps);

      naersk' = pkgs.callPackage naersk {};

    in
    {

      defaultPackage = pkgs.rustPlatform.buildRustPackage {
        src = ./.;
        name = "reuron";

        cargoLock = {
          lockFile = ./Cargo.lock;
          outputHashes = {
            "bevy_mod_picking-0.11.0" =
              "sha256-YkkBkgrd76nwJHUvEckN7M3dJ4TIzrP3RxyDNo5mkx0=";
            "bevy_mod_raycast-0.7.0" =
              "sha256-EGB9ZwkJwiRub6IaErg4qG6FzF7oyM1hyR4yLPwVnCE=";
          };
        };

        checkPhase = "echo 'Skipping tests'";

        nativeBuildInputs = buildInputs;
        buildInputs = buildInputs;
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        COREAUDIO_SDK_PATH= if system == "aarch64-darwin" then "${pkgs.darwin.apple_sdk.MacOSX-SDK}" else "";
      };


      packages.wasm-build = pkgs.rustPlatform.buildRustPackage {

        src = ./.;
        name = "reuron-wasm";

        cargoLock = {
          lockFile = ./Cargo.lock;
          outputHashes = {
            "bevy-0.11.0-dev" =
              "sha256-iSn+HsrMKJEnY8VqRj/dxDZKle3ozTmn097dR+ZuX1w=";
          };
        };

        buildPhase = ''
          cargo build --release --target=wasm32-unknown-unknown

          echo 'Creating out dir...'
          mkdir -p $out

           trunk build --release --dist $out index.html
        '';
        checkPhase = "echo 'Skipping tests'";
        installPhase = "echo 'Skipping install phase'";

        buildInputs = buildInputs;
        nativeBuildInputs = buildInputs;
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        COREAUDIO_SDK_PATH= if system == "aarch64-darwin" then "${pkgs.darwin.apple_sdk.MacOSX-SDK}" else "";
        VERGEN_GIT_SHA=self.sourceInfo.lastModifiedDate;
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
