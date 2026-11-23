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
    pub interval_sec: f32,           // control rate; runs at most once per frame
    pub inputs: Vec<ModelInput>,     // one per graph input, in order
    pub outputs: Vec<ModelOutput>,   // one per graph output, in order
}

#[serde(tag = "type")]
pub enum ModelInput {
    // Concatenated into a [1, n] tensor in list order.
    Voltages { taps: Vec<SegmentRef>, offset_mv: f32, scale: f32 },
    Time,
    Constant { values: Vec<f32> },   // slider-able through nb.Slider
    Feedback { output: String },     // previous run's output; zeros at first
}

#[serde(tag = "type")]
pub enum ModelOutput {
    // Output element i drives taps[i].
    InjectCurrent { taps: Vec<SegmentRef>, scale_uamps_per_square_cm: f32 },
    // Multiplier max(0, offset + scale * value) on one channel (by index in
    // the segment's membrane_channels) or, with channel null, all of them.
    ScaleConductance { taps: Vec<SegmentRef>, channel: Option<u32>, offset: f32, scale: f32 },
    // Same multiplier on every receptor current of synapses[i] (an index
    // into the scene's synapses).
    SynapticGain { synapses: Vec<u32>, offset: f32, scale: f32 },
    Ignore,
}

pub struct SegmentRef { pub neuron: u32, pub segment: u32 }
```

`models` defaults to empty, so existing scenes keep working. Afterwards,
regenerate `prelude/schema.ncl` (`cargo run --bin gen_nickel_schema`) and
commit it in nb-nickel. Graph input and output shapes are checked against the
bindings when a model finishes compiling (`ModelInstance::bind`). Mismatches
show in red under "Models" in the Scene panel.

A dynamic input dim (`None`) absorbs the bound values: `[?, 3]` with 6 values
becomes `[2, 3]` (`coupling::fit_shape`).

**Recurrent state.** `Feedback { output }` feeds a graph input from the
named graph output of the previous run. This is how PyTorch exports a
stateful RNN (`h0` in, `hn` out), and it works for any graph, not just
GRU/LSTM nodes. The input must have a fixed shape, which gives the state
its size (onnx-ir infers no static shape for GRU/LSTM outputs). The state
starts at zeros, resets when the scene is reloaded (the model is rebound),
and only advances on runs that succeed. A non-finite value fed back is an
error, so NaN can't latch.

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
- **Sinks reach the backends through one per-frame `controls` array**
  (`GpuSimState::controls`, filled by `extract::fill_controls`). Its
  layout, with n segments: `[0, n)` input currents, then `MAX_CHANNELS`
  conductance multipliers per segment, then one gain per synapse. The CPU
  kernels read it directly. The GPU uploads it into binding 5, which
  replaced the input-current buffer, so no new storage buffer was needed
  (WebGPU allows 8). `run_models` writes `ModelCurrent`,
  `ModelConductanceScale` (segments) and `ModelSynapseGain` (synapses,
  addressed by `SynapseAddress`). Currents from several models add;
  multipliers multiply. Sink targets are checked against the live scene
  on each run, and a missing segment, channel or synapse fails the model
  with an error in the Scene panel.
  - This also fixed a phase 2 bug: the GPU's per-frame upload rebuilt
    input currents from `InputCurrent` and the stimulator only, so
    `ModelCurrent` never reached the shader after the first frame.
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

`Program::run_traced` returns every graph input and every step's output as
`Activation { name, op, tensor }`. `run_models` keeps the latest trace on
each `ModelSlot` (`activations`), and the egui "Models" window
(`src/gui/models_panel.rs`) draws each layer as a heatmap with painter
rectangles, not textures. The last dim runs along a line and the other
dims stack, wrapping when a line is too long. The coloring is diverging:
blue for negative, red for positive, scaled per layer to its largest
|value|. Hovering a cell shows its index and value.

- Layer names: onnx-ir renames every value after its node (`tanh1_out1`).
  Graph outputs get their names back from the `ModelProto`
  (`Program::restore_output_names`); intermediate layers show the node
  name (`tanh1`).
- Every run is traced. That's cheap for controller-sized MLPs; for big
  models, trace only while the window is open.
- 3D placement of layers in the scene (the 2024 branch's idea) can come
  later and reuse the same data.

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
2. **Scene coupling. Done.**
   - Schema types in `serialize.rs`; `SegmentAddress` and
     `ModelCurrent::default()` on every spawned segment.
   - `src/onnx/coupling.rs`: pure binding, gather and scatter logic.
     NaN inputs (GPU voltages before the first readback) skip a run, and
     outputs are summed per segment.
   - `src/onnx/plugin.rs`: `SceneModels` caches compiled programs by
     resolved URL. Slider changes rebind without refetching, and "Load"
     refetches. Also `run_models` before `step_biophysics_flat`.
     `ModelCurrent` is summed in `extract_to_gpu_state` and
     `update_input_currents`, so it feeds both backends.
   - A status line `model <url> (<runs> runs): n0s1: -50.2 mV, +3.1 µA/cm²`
     is logged every 2 s.
   - Demo: nb-lib `demos/onnx_clamp.ncl` and `models/voltage_clamp.onnx`
     (from `models/make_models.py`).
   - Tests: `tests/onnx_coupling.rs` (includes a headless Bevy app from
     fetch to `ModelCurrent`), and `nickel_lib` for the demo.
3. **Visualization. Done.** `Program::run_traced`, `ModelSlot::activations`
   and the egui Models window. Tests: the traced run in `onnx_runtime`, the
   kept trace in `onnx_coupling`'s headless app, and heatmap layout and
   color in `models_panel`.
4. **Coverage. Done.**
   - Steps may have several outputs.
   - Conv1d: explicit padding, `SAME_UPPER`/`SAME_LOWER`, stride, dilation
     and groups.
   - Conv2d and MaxPool2d on `[N, C, H, W]`: explicit (also asymmetric)
     padding, `SAME_UPPER`/`SAME_LOWER`/`VALID`, stride and dilation;
     groups for Conv2d, `ceil_mode` for MaxPool2d. MaxPool's Indices output
     is rejected. This is enough for the ONNX model zoo's MNIST CNN
     (mnist-12), which matches the zoo's own test output to 2e-5.
   - GRU (`linear_before_reset` 0 and 1) and LSTM (`input_forget`). Both
     support forward, reverse and bidirectional, `clip`, optional bias and
     initial state, and sigmoid/tanh/relu activations. They are written as
     explicit per-timestep loops over burn ops. onnx-ir itself rejects
     `sequence_lens` and LSTM peepholes; we reject `layout=1` (batch first).
   - `ModelInput::Feedback` carries state between runs.
   - Tests: the `conv1d`, `conv2d`, `gru` and `lstm` fixtures, and the
     nb-lib `onnx_digits` (MLP) and `onnx_mnist` (zoo CNN) demos in
     `nickel_lib`. The LSTM's expected
     values come from onnxruntime, because the reference evaluator ignores
     `clip`. Feedback tests are in `onnx_coupling`.
   - Output sinks: `ScaleConductance` and `SynapticGain`, through the
     shared `controls` array (see Coupling). Tests: `onnx_coupling` (the
     multipliers, target checks, and the headless app writing all three
     components) and `sim_controls` (the CPU kernels' response to each
     control, plus naga validation of the WGSL and its layout constants).
     The GPU path has not been run on a real GPU.
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
