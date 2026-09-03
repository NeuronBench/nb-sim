# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

- **Entry point**: `src/bin/bevy.rs` - Calls `start()`; scenes are Nickel files evaluated in-process
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
- `grace.rs` - Spawns entities from the `serialize::Scene` wire format
- `nickel.rs` - Nickel scenes: link/evaluate/decode into `serialize::Scene`
- `neuroml.rs` - NeuroML format integration (uses local neuroml-rs dependency)

**Nickel scene language** (two sibling repositories, expected at `../nb-nickel` and `../nb-lib`)
- `../nb-nickel` - Crate: import linker (no filesystem needed, works in WASM), evaluator, structured diagnostics, `params` slider discovery. Prepends the `nb` prelude to every program. Path dependency during development; switch to a git dependency once pushed.
- `../nb-nickel/prelude/schema.ncl` - GENERATED contracts from `src/serialize.rs`; regenerate with `cargo run --bin gen_nickel_schema` and commit in nb-nickel (a test in `tests/nickel_lib.rs` fails when the embedded copy is stale)
- `../nb-nickel/prelude/helpers.ncl` - `nb.Slider`, `nb.Nullable`, `nb.Tagged`
- `../nb-lib` - The model library (channels, membranes, neurons, scenes) in Nickel; `../nb-lib/scene.ncl` is the native default scene; `tests/nickel_lib.rs` needs it (or `NB_LIB_DIR`)
- `src/gui/load.rs` - Async source fetching, linking, evaluation, respawn; `src/gui/scene_panel.rs` - egui Scene window with sliders and diagnostics
- Nickel gotcha: record fields are recursively scoped, so `{ neuron = neuron }` is a self-reference. Bind with a different name.
- Tests: `cargo test -p nb-nickel` and `cargo test -p nb-sim --test nickel_lib` (other nb-sim test targets are broken since the Bevy 0.18 migration)

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
- Scenes load from a URL (`?scene=` in the browser, `NB_SCENE` env var natively) and evaluate in-process
- Sample data in `sample_data/` includes SWC neuron files and scene definitions

### Environment Variables

- `NB_SCENE` - Native: path or URL of the root `.ncl` scene (defaults to `../nb-lib/scene.ncl`)
- `NB_LIB_DIR` - Where tests find nb-lib (defaults to `../nb-lib`)
- `INTERPRETER_URL` - Accepted by `start()` for compatibility and ignored