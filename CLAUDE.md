# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About nb-sim

NeuronBench is a web-first neural network simulator built with Rust and Bevy. This repository contains the main simulator client that compiles to both native applications and WASM web apps with identical UI.

## Development Commands

### Building

The project uses Nix for reproducible builds. All commands assume you're in a nix develop shell:

```shell
nix develop
```

**Native application:**
```shell
nix build .#
# or in dev shell:
cargo build --bin bevy
```

**Web client (WASM):**
```shell
nix build .#wasm-build
# or in dev shell:
cargo build --bin bevy --target wasm32-unknown-unknown
```

### Development

```shell
# Run native app
cargo run --bin bevy

# Build for web target
cargo build --bin bevy --target wasm32-unknown-unknown

# Use wasm-pack for web builds
wasm-pack build --mode no-install --release --target web
```

## Architecture Overview

### Core Structure

- **Entry point**: `src/bin/bevy.rs` - Calls `start()` with interpreter URL
- **Main app**: `src/start.rs` - Bevy app initialization with plugins and systems
- **Library root**: `src/lib.rs` - Module declarations

### Key Modules

**Neuron Simulation (`src/neuron/`)**
- `membrane.rs` - Membrane physics and materials
- `segment.rs` - Neuron segment modeling
- `channel.rs` - Ion channel simulation
- `solution.rs` - Chemical solution modeling
- `synapse.rs` - Synaptic connections
- `network.rs` - Neural network structures

**GUI (`src/gui/`)**
- `mod.rs` - Main GUI systems using egui
- `load.rs` - Scene loading and file handling
- `oscilloscope.rs` - Signal visualization
- `external_trigger.rs` - External trigger plugin

**Integrations (`src/integrations/`)**
- `grace.rs` - Grace scene format support
- `neuroml.rs` - NeuroML format integration (uses local neuroml-rs dependency)

**Core Systems**
- `plugin.rs` - Main NbSimPlugin for Bevy
- `selection.rs` - 3D selection and highlighting
- `stimulator.rs` - Electrical stimulation
- `serialize.rs` - Data serialization utilities

### Dependencies

- **Bevy 0.13.2** - Game engine and ECS framework
- **bevy_egui** - Immediate mode GUI integration
- **bevy_mod_picking** - 3D object picking/selection
- **bevy_panorbit_camera** - Camera controls
- **neuroml** - Local dependency at `../neuroml-rs` for NeuroML format support
- **wasm-bindgen** - WebAssembly bindings

### Key Patterns

- Uses Bevy's Entity Component System (ECS) architecture
- Web and native targets share same codebase via feature flags
- 3D visualization with egui overlay for controls
- External scene loading from remote interpreter service
- Sample data in `sample_data/` includes SWC neuron files and scene definitions

### Environment Variables

- `INTERPRETER_URL` - Defaults to "https://neuronbench.com/interpret"