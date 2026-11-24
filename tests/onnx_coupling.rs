//! Scene models: binding checks and gather/run/scatter against a fake
//! simulation, then the ECS systems end to end in a headless Bevy app.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use bevy::prelude::*;
use burn::tensor::Device;
use nb_sim::dimension::{MilliVolts, Timestamp};
use nb_sim::neuron::membrane::MembraneVoltage;
use nb_sim::dimension::FaradsPerSquareCm;
use nb_sim::integrations::grace::{ModelSynapseGain, SynapseAddress};
use nb_sim::neuron::membrane::Membrane;
use nb_sim::neuron::segment::ecs::{ModelConductanceScale, ModelCurrent, SegmentAddress};
use nb_sim::onnx::coupling::{Effects, ModelInstance, SimState, Step};
use nb_sim::onnx::plugin::{run_models, ModelFetchSender, ModelState, SceneModels};
use nb_sim::onnx::{HostTensor, OnnxPlugin, Program};
use nb_sim::serialize::{ModelInput, ModelOutput, SceneModel, SegmentRef};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sample_data/onnx").join(name)
}

fn clamp_program() -> Arc<Program> {
    let bytes = std::fs::read(fixture("clamp.onnx")).unwrap();
    Arc::new(Program::from_bytes(&bytes, Device::default()).unwrap())
}

fn tap(neuron: u32, segment: u32) -> SegmentRef {
    SegmentRef { neuron, segment }
}

/// The clamp bound to two voltage taps, driving `out_taps`.
fn clamp_spec(out_taps: Vec<SegmentRef>) -> SceneModel {
    SceneModel {
        url: "clamp.onnx".into(),
        interval_sec: 0.0,
        inputs: vec![
            ModelInput::Voltages { taps: vec![tap(0, 1), tap(1, 1)], offset_mv: 0.0, scale: 1.0 },
            ModelInput::Constant { values: vec![-50.0] },
            ModelInput::Constant { values: vec![0.1] },
        ],
        outputs: vec![ModelOutput::InjectCurrent { taps: out_taps, scale_uamps_per_square_cm: 100.0 }],
        hidden: vec![],
    }
}

struct FakeSim(HashMap<SegmentRef, f32>);

impl SimState for FakeSim {
    fn voltage_mv(&self, segment: SegmentRef) -> Option<f32> {
        self.0.get(&segment).copied()
    }
    fn time_sec(&self) -> f32 {
        0.0
    }
    fn channel_count(&self, segment: SegmentRef) -> Option<usize> {
        self.0.get(&segment).map(|_| 3)
    }
    fn has_synapse(&self, index: u32) -> bool {
        index < 2
    }
}

/// Run the model once and return what it asked for.
fn effects(model: &mut ModelInstance, sim: &FakeSim) -> Effects {
    match model.step(sim).unwrap() {
        Step::Ran(effects) => effects,
        Step::NotReady => panic!("should run"),
    }
}

fn sim(v0: f32, v1: f32) -> FakeSim {
    FakeSim(HashMap::from([(tap(0, 1), v0), (tap(1, 1), v1)]))
}

fn expected(v: f32) -> f32 {
    100.0 * (0.1 * (-50.0 - v)).tanh()
}

fn assert_close(got: f32, want: f32) {
    assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
}

#[test]
fn clamp_injects_scaled_drive_per_tap() {
    let mut model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(1, 1)])).unwrap();
    let currents = effects(&mut model, &sim(-70.0, -40.0)).currents;
    assert_eq!(currents.len(), 2);
    assert_eq!(currents[0].0, tap(0, 1));
    assert_close(currents[0].1, expected(-70.0)); // below target: positive current
    assert_close(currents[1].1, expected(-40.0)); // above target: negative current
    assert!(currents[0].1 > 0.0 && currents[1].1 < 0.0);
}

#[test]
fn one_segment_driven_twice_gets_the_sum() {
    let mut model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(0, 1)])).unwrap();
    let currents = effects(&mut model, &sim(-70.0, -40.0)).currents;
    assert_eq!(currents.len(), 1);
    assert_close(currents[0].1, expected(-70.0) + expected(-40.0));
}

#[test]
fn nan_voltage_means_not_ready() {
    let mut model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(1, 1)])).unwrap();
    assert_eq!(model.step(&sim(f32::NAN, -40.0)).unwrap(), Step::NotReady);
}

#[test]
fn missing_segment_is_an_error() {
    let mut model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(1, 1)])).unwrap();
    let err = model.step(&FakeSim(HashMap::from([(tap(0, 1), -70.0)]))).unwrap_err();
    assert!(err.contains("no segment 1 in neuron 1"), "{err}");
}

#[test]
fn binding_errors_name_the_problem() {
    let bind = |spec| ModelInstance::bind(clamp_program(), spec).err().expect("should fail");

    let mut spec = clamp_spec(vec![tap(0, 1), tap(1, 1)]);
    spec.inputs.pop();
    let err = bind(spec);
    assert!(err.contains("3 inputs") && err.contains("binds 2"), "{err}");

    let mut spec = clamp_spec(vec![tap(0, 1), tap(1, 1)]);
    spec.inputs[0] = ModelInput::Voltages { taps: vec![tap(0, 1)], offset_mv: 0.0, scale: 1.0 };
    let err = bind(spec);
    assert!(err.contains("`v_mv`") && err.contains("[1, 2]"), "{err}");

    let err = bind(clamp_spec(vec![tap(0, 1)]));
    assert!(err.contains("2 values") && err.contains("InjectCurrent names 1 targets"), "{err}");
}

/// Load the clamp from disk through `SceneModels`, as a scene would, and
/// check that `run_models` writes `ModelCurrent` on the tapped segments only.
#[test]
fn systems_fetch_compile_and_inject() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(OnnxPlugin)
        .insert_resource(Timestamp(0.0))
        .add_systems(Update, run_models);

    let segment = |app: &mut App, neuron, v| {
        app.world_mut()
            .spawn((
                SegmentAddress { neuron, segment: 1 },
                MembraneVoltage(MilliVolts(v)),
                Membrane { membrane_channels: vec![], capacitance: FaradsPerSquareCm(1.0) },
                ModelCurrent::default(),
                ModelConductanceScale::default(),
            ))
            .id()
    };
    let s0 = segment(&mut app, 0, -70.0);
    let s1 = segment(&mut app, 1, -40.0);
    let bystander = segment(&mut app, 2, -70.0);
    let synapse = app.world_mut().spawn((SynapseAddress(0), ModelSynapseGain::default())).id();

    // Relative URL resolved against the scene root, as in a scene file.
    let root = fixture("scene.ncl").to_string_lossy().into_owned();
    let sender = app.world().resource::<ModelFetchSender>().clone();
    // The same clamp twice: once injecting current, once as a gain on all
    // of segment (1,1)'s channels and on synapse 0.
    let mut gains = clamp_spec(vec![]);
    gains.outputs = vec![ModelOutput::ScaleConductance { taps: vec![tap(1, 1), tap(1, 1)], channel: None, offset: 1.0, scale: 0.5 }];
    let mut synaptic = clamp_spec(vec![]);
    synaptic.outputs = vec![ModelOutput::SynapticGain { synapses: vec![0, 0], offset: 1.0, scale: 0.5 }];
    app.world_mut()
        .resource_mut::<SceneModels>()
        .load(&[clamp_spec(vec![tap(0, 1), tap(1, 1)]), gains, synaptic], &root, false, &sender);
    app.world_mut().resource_mut::<SceneModels>().scene_spawned();

    for _ in 0..3 {
        app.update();
    }

    let models = app.world().resource::<SceneModels>();
    for slot in &models.slots {
        match &slot.state {
            ModelState::Ready(_) => {}
            ModelState::Failed(e) => panic!("model failed: {e}"),
            ModelState::Loading(_) => panic!("model still loading"),
        }
    }
    assert!(models.slots[0].runs > 0);

    // The latest run's layers are kept for the Models window: the three
    // inputs, then Sub, Mul, Tanh, ending in the graph output `drive`.
    let layers: Vec<(&str, &str)> = models.slots[0].activations.iter().map(|a| (a.name.as_str(), a.op)).collect();
    assert_eq!(layers.len(), 6, "{layers:?}");
    assert_eq!(layers[0], ("v_mv", "Input"));
    assert_eq!(layers[5], ("drive", "Tanh"));
    assert_eq!(models.slots[0].activations[0].tensor.data, vec![-70.0, -40.0]);

    let current = |e| app.world().get::<ModelCurrent>(e).unwrap().0 .0;
    assert_close(current(s0), expected(-70.0));
    assert_close(current(s1), expected(-40.0));
    assert_eq!(current(bystander), 0.0);

    let factor = (1.0 + 0.5 * (0.1f32 * (-50.0 - -70.0)).tanh()) * (1.0 + 0.5 * (0.1f32 * (-50.0 - -40.0)).tanh());
    let scales = |app: &App, e| app.world().get::<ModelConductanceScale>(e).unwrap().0;
    assert!(scales(&app, s1).iter().all(|s| (s - factor).abs() < 1e-5), "{:?}", scales(&app, s1));
    assert_eq!(scales(&app, s0), [1.0; 8]);
    assert_close(app.world().get::<ModelSynapseGain>(synapse).unwrap().0, factor);

    // Reloading with no models resets every sink.
    app.world_mut().resource_mut::<SceneModels>().load(&[], &root, false, &sender);
    app.world_mut().resource_mut::<SceneModels>().scene_spawned();
    app.update();
    assert_eq!(app.world().get::<ModelCurrent>(s0).unwrap().0 .0, 0.0);
    assert_eq!(scales(&app, s1), [1.0; 8]);
    assert_eq!(app.world().get::<ModelSynapseGain>(synapse).unwrap().0, 1.0);

    // A slider change respawns the scene: the compiled model is bound at
    // once, but the old neurons are gone and the new ones spawn frames
    // later. The model waits for them instead of failing on missing taps.
    for e in [s0, s1, bystander] {
        app.world_mut().despawn(e);
    }
    app.world_mut().resource_mut::<SceneModels>().load(&[clamp_spec(vec![tap(0, 1), tap(1, 1)])], &root, false, &sender);
    for _ in 0..3 {
        app.update();
    }
    let slot = &app.world().resource::<SceneModels>().slots[0];
    assert!(matches!(slot.state, ModelState::Ready(_)) && slot.runs == 0);
    let s0 = segment(&mut app, 0, -70.0);
    segment(&mut app, 1, -40.0);
    app.world_mut().resource_mut::<SceneModels>().scene_spawned();
    for _ in 0..3 {
        app.update();
    }
    let slot = &app.world().resource::<SceneModels>().slots[0];
    if let ModelState::Failed(e) = &slot.state {
        panic!("model failed after respawn: {e}");
    }
    assert!(slot.runs > 0);
    assert_close(app.world().get::<ModelCurrent>(s0).unwrap().0 .0, expected(-70.0));
}

fn program(name: &str) -> Arc<Program> {
    let bytes = std::fs::read(fixture(name)).unwrap();
    Arc::new(Program::from_bytes(&bytes, Device::default()).unwrap())
}

/// The GRU fixture with its initial state `h0` fed back from `y_h`.
fn gru_feedback_spec() -> SceneModel {
    SceneModel {
        url: "gru.onnx".into(),
        interval_sec: 0.0,
        inputs: vec![
            ModelInput::Constant { values: (0..18).map(|i| (i as f32 * 0.37).sin()).collect() },
            ModelInput::Feedback { output: "y_h".into() },
        ],
        outputs: vec![ModelOutput::Ignore, ModelOutput::Ignore, ModelOutput::Ignore],
        hidden: vec![],
    }
}

#[test]
fn feedback_carries_state_between_runs() {
    let program = program("gru.onnx");
    let mut model = ModelInstance::bind(program.clone(), gru_feedback_spec()).unwrap();
    let x = HostTensor::new(vec![3, 2, 3], (0..18).map(|i| (i as f32 * 0.37).sin()).collect());

    // Replay by hand: h starts at zeros, then each run's y_h is the next h0.
    let mut h = HostTensor::new(vec![1, 2, 4], vec![0.0; 8]);
    for run in 0..3 {
        let (step, trace) = model.step_traced(&sim(-70.0, -40.0)).unwrap();
        assert_eq!(step, Step::Ran(Effects::default()));
        let fed = trace.iter().find(|a| a.name == "h0").unwrap();
        assert_eq!(fed.tensor, h, "run {run}: h0 should be the previous y_h");
        let outputs = program.run(&[x.clone(), h.clone()]).unwrap();
        h = outputs[1].clone();
        assert!(h.data.iter().any(|v| *v != 0.0));
    }

    // A fresh binding starts from zeros again.
    let mut model = ModelInstance::bind(program, gru_feedback_spec()).unwrap();
    let (_, trace) = model.step_traced(&sim(-70.0, -40.0)).unwrap();
    assert!(trace.iter().find(|a| a.name == "h0").unwrap().tensor.data.iter().all(|v| *v == 0.0));
}

#[test]
fn feedback_binding_errors_name_the_problem() {
    let mut spec = gru_feedback_spec();
    spec.inputs[1] = ModelInput::Feedback { output: "hn".into() };
    let err = ModelInstance::bind(program("gru.onnx"), spec).err().unwrap();
    assert!(err.contains("`hn`") && err.contains("`y_h`"), "{err}");

    // `y` is [3, 1, 2, 4]: 24 values for an 8-value input. Caught on the first run.
    let mut spec = gru_feedback_spec();
    spec.inputs[1] = ModelInput::Feedback { output: "y".into() };
    let mut model = ModelInstance::bind(program("gru.onnx"), spec).unwrap();
    let err = model.step(&sim(-70.0, -40.0)).unwrap_err();
    assert!(err.contains("24 values") && err.contains("`h0`"), "{err}");
}

/// The clamp's two outputs (tanh drive per tap) routed to the new sinks.
fn clamp_with_output(output: ModelOutput) -> ModelInstance {
    let mut spec = clamp_spec(vec![tap(0, 1), tap(1, 1)]);
    spec.outputs = vec![output];
    ModelInstance::bind(clamp_program(), spec).unwrap()
}

#[test]
fn scale_conductance_multiplies_per_channel() {
    let drive = |v: f32| (0.1 * (-50.0 - v)).tanh();
    // One output scales channel 2 of segment (0,1), the other all of (1,1).
    let mut model = clamp_with_output(ModelOutput::ScaleConductance {
        taps: vec![tap(0, 1), tap(0, 1)],
        channel: Some(2),
        offset: 1.0,
        scale: 0.5,
    });
    let e = effects(&mut model, &sim(-70.0, -40.0));
    assert!(e.currents.is_empty() && e.synapse_gains.is_empty());
    let [(t, channel, factor)] = e.conductance_scales.as_slice() else { panic!("{e:?}") };
    assert_eq!((*t, *channel), (tap(0, 1), Some(2)));
    // Two outputs on one channel multiply.
    assert_close(*factor, (1.0 + 0.5 * drive(-70.0)) * (1.0 + 0.5 * drive(-40.0)));

    // Multipliers never go negative.
    let mut model = clamp_with_output(ModelOutput::ScaleConductance {
        taps: vec![tap(0, 1), tap(1, 1)],
        channel: None,
        offset: 0.0,
        scale: 10.0,
    });
    let e = effects(&mut model, &sim(-70.0, -40.0));
    assert_eq!(e.conductance_scales[1], (tap(1, 1), None, 0.0));
    assert_close(e.conductance_scales[0].2, 10.0 * drive(-70.0));
}

#[test]
fn synaptic_gain_targets_scene_synapses() {
    let drive = |v: f32| (0.1 * (-50.0 - v)).tanh();
    let mut model = clamp_with_output(ModelOutput::SynapticGain { synapses: vec![1, 0], offset: 1.0, scale: 1.0 });
    let e = effects(&mut model, &sim(-70.0, -40.0));
    assert_eq!(e.synapse_gains.len(), 2);
    assert_eq!(e.synapse_gains[0].0, 1);
    assert_close(e.synapse_gains[0].1, 1.0 + drive(-70.0));
}

#[test]
fn sink_targets_are_checked_against_the_scene() {
    let run = |output| clamp_with_output(output).step(&sim(-70.0, -40.0)).unwrap_err();
    let err = run(ModelOutput::SynapticGain { synapses: vec![0, 7], offset: 1.0, scale: 1.0 });
    assert!(err.contains("no synapse 7"), "{err}");
    let err = run(ModelOutput::ScaleConductance { taps: vec![tap(0, 1), tap(1, 1)], channel: Some(3), offset: 1.0, scale: 1.0 });
    assert!(err.contains("has 3 channels") && err.contains("no channel 3"), "{err}");
    let err = run(ModelOutput::InjectCurrent { taps: vec![tap(0, 1), tap(4, 4)], scale_uamps_per_square_cm: 1.0 });
    assert!(err.contains("no segment 4 in neuron 4"), "{err}");
}

#[test]
fn hidden_taps_drive_segments_from_intermediate_values() {
    use nb_sim::serialize::HiddenTap;
    let names: Vec<String> = clamp_program().value_names().iter().map(|s| s.to_string()).collect();
    let hidden = names.iter().find(|n| n.starts_with("sub")).expect("the clamp subtracts").clone();
    let hidden_tap = |value: &str, elements: Vec<u32>, taps: Vec<SegmentRef>| HiddenTap {
        value: value.into(),
        elements,
        output: ModelOutput::InjectCurrent { taps, scale_uamps_per_square_cm: 2.0 },
    };

    // Unknown values and miscounted elements are binding errors.
    let mut spec = clamp_spec(vec![tap(0, 1), tap(1, 1)]);
    spec.hidden = vec![hidden_tap("nope", vec![0], vec![tap(2, 1)])];
    let err = ModelInstance::bind(clamp_program(), spec.clone()).err().unwrap();
    assert!(err.contains("`nope`") && err.contains(&hidden), "{err}");
    spec.hidden = vec![hidden_tap(&hidden, vec![0, 1], vec![tap(2, 1)])];
    let err = ModelInstance::bind(clamp_program(), spec.clone()).err().unwrap();
    assert!(err.contains("picks 2 elements") && err.contains("1 targets"), "{err}");

    // Element 1 of the hidden value drives segment (2, 1), element 0 adds to
    // segment (0, 1) on top of the graph output's current there.
    spec.hidden = vec![hidden_tap(&hidden, vec![1, 0], vec![tap(2, 1), tap(0, 1)])];
    let mut model = ModelInstance::bind(clamp_program(), spec).unwrap();
    let mut state = sim(-70.0, -40.0);
    state.0.insert(tap(2, 1), -65.0);
    let (Step::Ran(effects), trace) = model.step_traced(&state).unwrap() else { panic!("should run") };
    let values = &trace.iter().find(|a| a.name == hidden).unwrap().tensor.data;
    let current = |t| effects.currents.iter().find(|(s, _)| *s == t).unwrap().1;
    assert_close(current(tap(2, 1)), 2.0 * values[1]);
    assert_close(current(tap(0, 1)), expected(-70.0) + 2.0 * values[0]);
    assert!(model.taps().any(|t| t == tap(2, 1)), "hidden targets count as the model's taps");

    // A missing target segment is a runtime error naming the sink.
    let mut model = ModelInstance::bind(clamp_program(), {
        let mut s = clamp_spec(vec![tap(0, 1), tap(1, 1)]);
        s.hidden = vec![hidden_tap(&hidden, vec![0], vec![tap(9, 9)])];
        s
    })
    .unwrap();
    let err = model.step(&sim(-70.0, -40.0)).err().unwrap();
    assert!(err.contains("no segment 9 in neuron 9"), "{err}");
}
