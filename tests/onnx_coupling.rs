//! Scene models: binding checks and gather/run/scatter against a fake
//! simulation, then the ECS systems end to end in a headless Bevy app.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use bevy::prelude::*;
use burn::tensor::Device;
use nb_sim::dimension::{MilliVolts, Timestamp};
use nb_sim::neuron::membrane::MembraneVoltage;
use nb_sim::neuron::segment::ecs::{ModelCurrent, SegmentAddress};
use nb_sim::onnx::coupling::{ModelInstance, SimState, Step};
use nb_sim::onnx::plugin::{run_models, ModelFetchSender, ModelState, SceneModels};
use nb_sim::onnx::{OnnxPlugin, Program};
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
    let model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(1, 1)])).unwrap();
    let Step::Inject(currents) = model.step(&sim(-70.0, -40.0)).unwrap() else { panic!("should run") };
    assert_eq!(currents.len(), 2);
    assert_eq!(currents[0].0, tap(0, 1));
    assert_close(currents[0].1, expected(-70.0)); // below target: positive current
    assert_close(currents[1].1, expected(-40.0)); // above target: negative current
    assert!(currents[0].1 > 0.0 && currents[1].1 < 0.0);
}

#[test]
fn one_segment_driven_twice_gets_the_sum() {
    let model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(0, 1)])).unwrap();
    let Step::Inject(currents) = model.step(&sim(-70.0, -40.0)).unwrap() else { panic!("should run") };
    assert_eq!(currents.len(), 1);
    assert_close(currents[0].1, expected(-70.0) + expected(-40.0));
}

#[test]
fn nan_voltage_means_not_ready() {
    let model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(1, 1)])).unwrap();
    assert_eq!(model.step(&sim(f32::NAN, -40.0)).unwrap(), Step::NotReady);
}

#[test]
fn missing_segment_is_an_error() {
    let model = ModelInstance::bind(clamp_program(), clamp_spec(vec![tap(0, 1), tap(1, 1)])).unwrap();
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
    assert!(err.contains("2 values") && err.contains("1 taps"), "{err}");
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
                ModelCurrent::default(),
            ))
            .id()
    };
    let s0 = segment(&mut app, 0, -70.0);
    let s1 = segment(&mut app, 1, -40.0);
    let bystander = segment(&mut app, 2, -70.0);

    // Relative URL resolved against the scene root, as in a scene file.
    let root = fixture("scene.ncl").to_string_lossy().into_owned();
    let sender = app.world().resource::<ModelFetchSender>().clone();
    app.world_mut()
        .resource_mut::<SceneModels>()
        .load(&[clamp_spec(vec![tap(0, 1), tap(1, 1)])], &root, false, &sender);

    for _ in 0..3 {
        app.update();
    }

    let models = app.world().resource::<SceneModels>();
    match &models.slots[0].state {
        ModelState::Ready(_) => {}
        ModelState::Failed(e) => panic!("model failed: {e}"),
        ModelState::Loading(_) => panic!("model still loading"),
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

    // Reloading with no models clears the injected current.
    app.world_mut().resource_mut::<SceneModels>().load(&[], &root, false, &sender);
    app.update();
    assert_eq!(app.world().get::<ModelCurrent>(s0).unwrap().0 .0, 0.0);
}
