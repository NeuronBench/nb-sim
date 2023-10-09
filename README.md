# nb-sim

NeuronBench is a web-first neural network simulator. To read more about that that
means, please see the [organization readme](https://github.com/neuronbench).

This repository contains the main simulator client, which can be compiled as either
a native application, or a wasm web app. Both the native client and the web app
have the same UI. they both load an example scene at startup, and can load new
scenes if you have an internet connection.

## Building

The easiest way to build nb-sim is with [nix](https://nixos.org). From the
project root, you can:

### Build the native app

``` shell
nix build .#
```

### Build the web client

``` shell
nix build .#wasm-build
```

### Enter a development environment

``` shell
nix develop
cargo build --bin bevy
cargo build --bin bevy --target wasm32-unknown-unknown
```
