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

      # wasm-bindgen-cli = pkgs.rustPlatform.buildRustPackage rec {
      #   pname = "wasm-bindgen-cli";
      #   version = "0.2.86";

      #   src = pkgs.fetchCrate {
      #     inherit pname version;
      #     sha256 = "sha256-56EOiLbdgAcoTrkyvB3t9TjtLaRvGxFUXx4haLwE2QY=";
      #   };

      #   cargoSha256 = "sha256-4CPBmz92PuPN6KeGDTdYPAf5+vTFk9EN5Cmx4QJy6yI=";

      #   nativeBuildInputs = [ pkgs.pkg-config ];

      #   buildInputs = [ pkgs.openssl ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.curl apple.Security ];

      #   doCheck = false;
      #   # nativeCheckInputs = [ pkgs.nodejs ];

      # };

      nbSimLockHashes = {
          lockFile = ./Cargo.lock;
          outputHashes = { };
        };


      buildInputs = [
          # wasm-bindgen-cli
          pkgs.wasm-bindgen-cli
          pkgs.wasm-pack
          pkgs.which
          rust
          pkgs.curl
          pkgs.autoconf
          pkgs.pkg-config
          pkgs.openssl
          pkgs.binaryen
          pkgs.sass
          ] ++ (if system == "aarch64-darwin" then apple-deps else linux-deps);

      naersk' = pkgs.callPackage naersk {};

    in
    {

      defaultPackage = pkgs.rustPlatform.buildRustPackage {
        src = ./.;
        name = "nb-sim";

        cargoLock = nbSimLockHashes;

        checkPhase = "echo 'Skipping tests'";

        nativeBuildInputs = buildInputs;
        buildInputs = buildInputs;
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        COREAUDIO_SDK_PATH= if system == "aarch64-darwin" then "${pkgs.darwin.apple_sdk.MacOSX-SDK}" else "";
      };

      # packages.wasm-bindgen-cli = wasm-bindgen-cli;

      packages.wasm-build = pkgs.rustPlatform.buildRustPackage {

        src = ./.;
        name = "nb-sim-wasm";

        cargoLock = nbSimLockHashes;

        buildPhase = ''
          HOME=$(mktemp -d fake-homeXXXX) RUSTFLAGS="--cfg=web_sys_unstable_apis" wasm-pack build --mode no-install --release --target web
        '';
        checkPhase = "echo 'Skipping tests'";
        installPhase = ''
          mkdir -p $out
          cp pkg/* $out/
        '';

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
          # wasm-bindgen-cli
          rust
          pkgs.autoconf
          pkgs.wasm-bindgen-cli
          pkgs.pkg-config
          pkgs.openssl
          pkgs.sass
          pkgs.binaryen
          pkgs.wasm-pack
          ] ++ (if system == "aarch64-darwin" then apple-deps else linux-deps);

        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        COREAUDIO_SDK_PATH= if system == "aarch64-darwin" then "${pkgs.darwin.apple_sdk.MacOSX-SDK}" else "";
      };
    }
  );
}
