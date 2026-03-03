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
      isDarwin = pkgs.stdenv.isDarwin;
      apple-deps = pkgs.lib.optionals isDarwin [ pkgs.apple-sdk pkgs.libiconv ];
      linux-deps = [
          pkgs.udev pkgs.alsa-lib pkgs.vulkan-loader
          pkgs.xorg.libX11 pkgs.xorg.libXcursor pkgs.xorg.libXi
          pkgs.xorg.libXrandr pkgs.libxkbcommon pkgs.wayland

      ];

      wasm-bindgen-cli = pkgs.buildWasmBindgenCli rec {
        src = pkgs.fetchCrate {
          pname = "wasm-bindgen-cli";
          version = "0.2.111";
          hash = "sha256-vCa7VIGmMB3baGQqhkd6r4XmUktt61ibcjDQtRW4PzA=";
        };
        cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
          inherit src;
          inherit (src) pname version;
          hash = "sha256-Sl/AJXq4NSryKIXXo2Fjy6ybVxB8ezka8VQBBxbWPCw=";
        };
      };

      nbSimLockHashes = {
          lockFile = ./Cargo.lock;
          outputHashes = { 
	    "uom-0.36.0" = "sha256-dCYFmwklzeQx/Kr3K9lsWGB6lxKBrRFlpLhfjm9HqFE=";
	    "neuroml-0.1.0" = "sha256-mU6KgN/O6H32ApPRMpCIWY8sojp0vHh1LjeMQtTM7zQ=";
	  };
        };


      buildInputs = [
          wasm-bindgen-cli
          pkgs.wasm-pack
          pkgs.which
          rust
          pkgs.curl
          pkgs.autoconf
          pkgs.pkg-config
          pkgs.openssl
          pkgs.binaryen
          pkgs.sass
          ] ++ (if isDarwin then apple-deps else linux-deps);

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
      };

      packages.wasm-build = pkgs.rustPlatform.buildRustPackage {

        src = ./.;
        name = "nb-sim-wasm";

        cargoLock = nbSimLockHashes;

        buildPhase = ''
          HOME=$(mktemp -d fake-homeXXXX) wasm-pack build --mode no-install --release --target web
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
        VERGEN_GIT_SHA=self.sourceInfo.lastModifiedDate;
      };


      devShell = pkgs.mkShell rec {
        buildInputs = [
          rust
          pkgs.git
          pkgs.autoconf
          wasm-bindgen-cli
          pkgs.pkg-config
          pkgs.openssl
          pkgs.sass
          pkgs.binaryen
          pkgs.wasm-pack
          ] ++ (if isDarwin then apple-deps else linux-deps);

        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
      };
    }
  );
}
