# ONNX integration design

Status: design, backed by a working spike (2026-11-18). Nothing is merged.

## Goals

1. **Models as sim components** (primary). A trained network reads simulation
   state (membrane voltages, time, slider params) and writes back into the
   simulation, starting with injected current. Examples: closed-loop
   controllers, learned stimulators, decoders of population activity.
2. **Model visualization** (secondary). Show a running model's activations
   next to the biophysical scene.

Scenes stay the single source of truth: a Nickel scene names the model by URL
and declares its bindings, exactly as it declares neurons and stimulators.

## Key findings

### burn-onnx is codegen-only, so we interpret onnx-ir at runtime

`burn-onnx` (formerly `burn-import`) turns a `.onnx` file into Rust source in
`build.rs`. That can't serve models named by a scene URL, and certainly not in
the browser.

The parser underneath it, **`onnx-ir`**, is an ordinary runtime library:
`OnnxGraphBuilder::new().parse_bytes(&bytes)` returns an `OnnxGraph` of typed
nodes (`Node::Gemm(GemmNode { config, inputs, outputs })`, `Node::Tanh`,
`Node::Conv2d`, ...), with type inference and simplification already done, and
constant weights reachable through `Argument::value()`. We write a small
interpreter that maps each `Node` variant to burn tensor ops.

**Spike result:** a 4→8→2 Gemm/Tanh/Gemm MLP was parsed at runtime and
evaluated with burn-flex. The output `[3.973571, 3.3044875]` matches numpy's
reference exactly. The same crate builds for `wasm32-unknown-unknown`
(545 KB release). Spike source:
`/private/tmp/claude-501/-Users-greghale--claude-spares-spare-KGJdiw/1d574e5d-7378-410a-b7ca-96f4568eb19a/scratchpad/burn-spike/src/main.rs`.

Interpreter notes from the spike:
- ONNX initializers show up as `Node::Constant`. Skip them; consumers read the
  data via `Argument::value()`.
- Weight `TensorData` comes back as burn's own type, so loading is zero-glue.
- Gemm's `alpha`, `beta`, `trans_a` and `trans_b` are already decoded in `GemmConfig`.
- onnx-ir's simplifier rewrites the graph before we see it: MatMul+Add with
  a constant weight becomes `Linear`, and `Identity` is removed. Fixtures
  must be designed so each op actually reaches the interpreter. Check with
  `node.node_type()`.

### Versions

| | wgpu | rustc |
|---|---|---|
| Bevy 0.18 (current) | 27 | — |
| Bevy 0.19 | 29 | — |
| Bevy 0.20 | 30 | — |
| burn 0.21 (cubecl 0.10) | 29 | 1.92 |
| burn 0.22 (cubecl 0.11) | 30 | **1.95** |

- **Start CPU-only with burn 0.22 + the `flex` backend.** It pulls in no wgpu
  at all, so it can't conflict with Bevy's. A scratch copy of nb-sim with
  `burn = { version = "0.22.1", default-features = false, features = ["std", "flex"] }`
  and `onnx-ir = { version = "0.22.0", default-features = false }` passes
  `cargo check` both natively and for wasm.
- **burn 0.22 needs rustc 1.95.** The nix dev shell already provides 1.98.1,
  and nixpkgs' rustc (used by `nix build`) is also 1.98.1. A rustup `cargo`
  outside the shell may be too old, so build inside `nix develop`.
- **wasm needs** `getrandom 0.4` with `wasm_js` (target-gated dependency,
  renamed so it doesn't clash with the existing getrandom versions).
- burn 0.22 reworked its API. `Tensor<D, K>` is no longer generic over the
  backend: a runtime `Device` picks it (`Device::default()`, `Device::flex()`,
  `Device::wgpu(..)`). This makes CPU→GPU a device swap, not a type change.
  `burn-ndarray` is deprecated; use `flex`.
- **Sharing Bevy's GPU device** means matching wgpu versions, which requires
  Bevy 0.20 (wgpu 30) or Bevy 0.19 with burn 0.21 (wgpu 29). Deferred to
  phase 5.

## Architecture

```
scene.ncl ──► serialize::Scene.models[] ──► fetch .onnx (ehttp, cached by URL)
                                                │
                                         onnx-ir parse_bytes
                                                │
                                      onnx::Program (compiled op list)
                                                │
   MembraneVoltage, Timestamp, params ─► gather inputs ─► run (burn-flex) ─► scatter outputs ─► InputCurrent
                                                │
                                      activations ─► egui model panel
```

### `src/onnx/` module

- `program.rs`: `Program::compile(OnnxGraph) -> Result<Program, Unsupported>`.
  It walks the nodes once, rejects unsupported ops with a clear diagnostic
  (op name and node name, shown in the Scene panel), resolves constants to
  `Tensor`s once, and keeps the op list.
  `Program::run(&self, inputs) -> Outputs` evaluates it. Optionally, it also
  keeps every intermediate value for visualization.
- **Rank erasure.** burn's `Tensor<const D>` is rank-typed, but an interpreter
  sees ranks only at runtime. Store values as `enum Value { R1(Tensor<1>), ..., R4(Tensor<4>) }`
  with a `rank_dispatch!` macro. ONNX rank is static after onnx-ir's type
  inference, so this check happens once, at compile time.
- **First op set:** Constant, Gemm, MatMul, Add, Sub, Mul, Div, Relu, Tanh,
  Sigmoid, Softmax, Reshape, Flatten, Concat, Identity. That covers MLP
  controllers and decoders. Conv1d and recurrent ops (GRU/LSTM) come later.
- `plugin.rs`: `OnnxPlugin` (systems below); `load.rs`: async fetch and cache.

### Scene schema (`src/serialize.rs`)

```rust
pub struct Scene {
    pub neurons: Vec<SceneNeuron>,
    pub synapses: Vec<Synapse>,
    #[serde(default)]
    pub models: Vec<SceneModel>,
}

pub struct SceneModel {
    pub url: String,                 // relative to the scene root, like imports
    pub update_interval_sec: f32,    // control rate; >= one frame's sim time
    pub inputs: Vec<ModelInput>,     // one per graph input, in order
    pub outputs: Vec<ModelOutput>,   // one per graph output, in order
}

#[serde(tag = "type")]
pub enum ModelInput {
    // Concatenated into a [1, n] tensor in list order.
    Voltages { taps: Vec<SegmentRef>, offset_mv: f32, scale: f32 },
    Time,
    Constant { values: Vec<f32> },   // slider-able through nb.Slider
}

#[serde(tag = "type")]
pub enum ModelOutput {
    // Output element i drives taps[i].
    InjectCurrent { taps: Vec<SegmentRef>, scale_uamps_per_square_cm: f32 },
    Ignore,
}

pub struct SegmentRef { pub neuron: u32, pub segment: u32 }
```

`models` defaults to empty, so existing scenes keep working. Afterwards,
regenerate `prelude/schema.ncl` (`cargo run --bin gen_nickel_schema`) and
commit it in nb-nickel. Graph input and output shapes are checked against the
bindings when a scene loads, and mismatches are reported as scene diagnostics.

### Coupling to the simulation

- Models run **between frames, at a control rate**, not inside the 0.5 µs
  integration step. A frame is `StepsPerFrame × dt` = 100 × 0.5 µs = 50 µs of
  sim time, so the effective latency is at least one frame. That's fine for
  controllers, stimulators and decoders. It is wrong for things that must run
  every step, such as a learned channel surrogate (see phase 5).
- **Output seam: `InputCurrent`.** It's spawned on every segment
  (`integrations/grace.rs`) and read every frame by both backends
  (`step_biophysics_flat` → `extract::update_input_currents` on the GPU path,
  and the flat CPU path). Writing it needs no shader or buffer changes, and
  stimulators still add on top.
  - To keep a hand-set `InputCurrent` intact, add a separate
    `ModelCurrent` component and sum it in `update_input_currents` and the
    CPU extract. That's a 2-line change in each, and cleaner than overwriting.
- **Input seam: `MembraneVoltage`.** It's already kept current on both
  backends (GPU readback writes it back). A voltage readback lags the GPU
  by a frame; that's acceptable at control rates.
- Systems: `run_models` in `Update`, after `apply_voltage_readback` and
  before `step_biophysics_flat`. It gathers inputs, runs each model whose
  interval has elapsed, and scatters into `ModelCurrent`.
- Respawn: the scene loader already despawns and respawns everything on a
  reload. Models are re-bound to the new segment entities by `SegmentRef`;
  the compiled `Program` is cached by URL and content hash, so slider tweaks
  don't refetch or recompile.

### Visualization

`Program::run` can keep every intermediate value. A `ModelActivations`
resource holds the latest values per model, and an egui "Models" window
draws each layer as a heatmap: the old branch's red/blue
`tensor_to_2d_image` coloring, ported to egui textures. 3D placement of
layers in the scene (the 2024 branch's idea) can come later and reuse the
same data.

## Phases

1. **Runtime. Done.** burn-flex and onnx-ir deps; `src/onnx/program.rs`
   (`Program::from_bytes`, `run`) covering Constant, Identity, Add, Sub, Mul,
   Div, Relu, Tanh, Sigmoid, LeakyRelu, Clip (static bounds), Softmax,
   Flatten, Reshape (static shape), Concat, MatMul, Gemm and Linear. Values
   are stored as `Tensor<4>` plus a logical rank, rather than as a rank enum.
   Fixtures are in `sample_data/onnx/`, regenerated with
   `uv run --with onnx --with numpy python -I sample_data/onnx/make_fixtures.py`.
   Expected outputs come from onnx's `ReferenceEvaluator`. Tests:
   `cargo test --test onnx_runtime`.
2. **Scene coupling.** Schema plus nb-nickel schema regeneration, async
   loader, `ModelCurrent`, `run_models`, diagnostics in the Scene panel.
   Example in nb-lib: a closed-loop controller that holds a soma at a target
   voltage, with the target as an `nb.Slider`.
3. **Visualization.** `ModelActivations` and an egui Models window.
4. **Coverage.** Conv1d, recurrent ops with state carried between calls
   (ONNX GRU/LSTM `initial_h` fed from the previous `Y_h`), more output
   sinks (conductance scaling, synaptic gain).
5. **GPU.** Upgrade to Bevy 0.20 (wgpu 30) so burn's wgpu backend can share
   Bevy's device. Then decide on per-step models (channel surrogates) inside
   the compute pass, or a burn→WGSL path for small MLPs.

## Open questions

- Should models run on a fixed sim-time interval only, or also be
  event-driven (for example on a spike)?
- Do we want ONNX outputs to address segments by index (as above) or by a
  named tag on segments? Names survive morphology edits better.
- Should a model be a scene-level entity (as above), or attached to a neuron
  so it moves with neuron templates in nb-lib?
