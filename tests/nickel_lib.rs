//! End-to-end checks that nb-lib evaluates through the Nickel pipeline into
//! the simulator's scene type.

use nb_nickel::{read_params, Override};
use nb_sim::integrations::nickel::{link_blocking, scene_from_linked};
use nb_sim::serialize::CurrentShape;

/// nb-lib lives in its own repository. Use `NB_LIB_DIR`, else the sibling
/// checkout `../nb-lib`.
fn lib_dir() -> String {
    if let Ok(dir) = std::env::var("NB_LIB_DIR") {
        return dir;
    }
    let sibling = format!("{}/../nb-lib", env!("CARGO_MANIFEST_DIR"));
    assert!(
        std::path::Path::new(&sibling).join("scene.ncl").exists(),
        "nb-lib not found at {sibling}; check out https://github.com/neuronbench/nb-lib next to nb-sim or set NB_LIB_DIR"
    );
    sibling
}

fn lib(path: &str) -> String {
    format!("{}/{path}", lib_dir())
}

fn on_current(scene: &nb_sim::serialize::Scene, neuron: usize, stim: usize) -> f32 {
    match &scene.neurons[neuron].stimulator_segments[stim].stimulator.current_shape {
        CurrentShape::SquareWave { on_current_uamps_per_square_cm, .. } => *on_current_uamps_per_square_cm,
        other => panic!("expected a square wave, got {other:?}"),
    }
}

#[test]
fn scene_evaluates_to_two_connected_cells() {
    let linked = link_blocking(&lib("scene.ncl")).unwrap();
    let scene = scene_from_linked(&linked, &[]).unwrap();
    assert_eq!(scene.neurons.len(), 2);
    assert_eq!(scene.neurons[0].neuron.segments.len(), 448);
    assert_eq!(scene.neurons[0].neuron.membranes.len(), 5);
    assert_eq!(scene.synapses.len(), 2);
    assert_eq!(scene.neurons[0].stimulator_segments[0].segment, 100);
    assert_eq!(on_current(&scene, 0, 0), 10.0);
    assert_eq!(scene.neurons[0].location.x_mm, 0.5);
    let soma = &scene.neurons[0].neuron.membranes[0];
    assert_eq!(soma.membrane_channels.len(), 3);
    assert_eq!(scene.synapses[0].synapse_membranes.presynaptic_pumps[0].transmitter, "Glutamate");
}

#[test]
fn scene_params_are_sliders_and_override_the_scene() {
    let linked = link_blocking(&lib("scene.ncl")).unwrap();
    let specs = read_params(&linked).unwrap();
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["stim_current", "stim_offset_sec", "spacing_mm"], "declaration order is preserved");
    let stim = specs.iter().find(|s| s.name == "stim_current").unwrap();
    assert_eq!((stim.default, stim.min), (10.0, Some(0.0)));
    let max = stim.max.expect("stim_current declares an nb.Slider range");
    assert!(max > stim.default);
    assert!(stim.doc.as_deref().unwrap().contains("current"));

    let scene = scene_from_linked(&linked, &[Override::number("params.stim_current", 25.0)]).unwrap();
    assert_eq!(on_current(&scene, 0, 0), 25.0);

    let err = scene_from_linked(&linked, &[Override::number("params.stim_current", max + 1.0)]).unwrap_err();
    assert!(err[0].message.contains("slider range"), "{}", err[0].render());
}

#[test]
fn refractory_demo_builds_a_row_of_neurons() {
    let linked = link_blocking(&lib("demos/refractory.ncl")).unwrap();
    let scene = scene_from_linked(&linked, &[]).unwrap();
    assert_eq!(scene.neurons.len(), 11);
    assert_eq!(scene.neurons[0].neuron.segments.len(), 14);
    assert_eq!(scene.neurons[3].stimulator_segments.len(), 2);
    let scene = scene_from_linked(&linked, &[Override::number("params.columns", 4.0)]).unwrap();
    assert_eq!(scene.neurons.len(), 4);
}

#[test]
fn onnx_clamp_demo_binds_a_model_to_the_soma() {
    use nb_sim::serialize::{ModelInput, ModelOutput};

    let linked = link_blocking(&lib("demos/onnx_clamp.ncl")).unwrap();
    let scene = scene_from_linked(&linked, &[Override::number("params.target_mv", -30.0)]).unwrap();
    assert_eq!(scene.neurons.len(), 2);
    let [model] = scene.models.as_slice() else { panic!("expected one model, got {:?}", scene.models) };
    assert_eq!(model.url, "../models/voltage_clamp.onnx");
    assert!(matches!(&model.inputs[1], ModelInput::Constant { values } if values == &vec![-30.0]));
    let ModelOutput::InjectCurrent { taps, .. } = &model.outputs[0] else { panic!("expected InjectCurrent") };
    assert_eq!((taps[0].neuron, taps[0].segment), (0, 1));

    // The model the demo names exists and binds to the demo's inputs/outputs.
    let path = nb_nickel::resolve_path(&lib("demos/onnx_clamp.ncl"), &model.url);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let program = nb_sim::onnx::Program::from_bytes(&bytes, Default::default()).unwrap();
    nb_sim::onnx::coupling::ModelInstance::bind(std::sync::Arc::new(program), model.clone()).unwrap();

    // Scenes without `models` still decode.
    let scene = scene_from_linked(&link_blocking(&lib("scene.ncl")).unwrap(), &[]).unwrap();
    assert!(scene.models.is_empty());
}

#[test]
fn errors_point_at_the_library_file_and_line() {
    let tmp = format!("{}/target/nickel-test-lib", env!("CARGO_MANIFEST_DIR"));
    let _ = std::fs::remove_dir_all(&tmp);
    for f in ["scene.ncl", "synapse.ncl", "membranes.ncl", "default_swc_neuron.ncl", "neurons/neuron_3.ncl"] {
        let dst = format!("{tmp}/{f}");
        std::fs::create_dir_all(std::path::Path::new(&dst).parent().unwrap()).unwrap();
        std::fs::copy(lib(f), dst).unwrap();
    }
    let channels = std::fs::read_to_string(lib("channels.ncl")).unwrap();
    // Nickel is lazy: a contract on a library entry the scene never uses is
    // never checked, so break a channel that the scene's membranes do use.
    let bad_line = channels.lines().position(|l| l.contains("sigma = 0.030")).unwrap() + 1;
    std::fs::write(format!("{tmp}/channels.ncl"), channels.replacen("sigma = 0.030", "sigmaa = 0.030", 1)).unwrap();

    let linked = link_blocking(&format!("{tmp}/scene.ncl")).unwrap();
    let err = scene_from_linked(&linked, &[]).unwrap_err();
    let rendered = err[0].render();
    assert!(rendered.contains("sigmaa"), "{rendered}");
    let (file, line, _) = err[0].primary_location().unwrap();
    assert!(file.ends_with("channels.ncl"), "{rendered}");
    assert!(line > 0 && line <= bad_line, "{rendered}");
}

#[test]
fn schema_file_is_current() {
    let embedded = nb_nickel::PRELUDE_SCHEMA;
    let generated = nb_sim::nickel_schema::generate();
    assert!(
        embedded == generated,
        "nb-nickel's prelude/schema.ncl is stale; run `cargo run --bin gen_nickel_schema` and commit it in nb-nickel"
    );
}

/// The digit models read only constants; every segment exists.
struct NoSim;
impl nb_sim::onnx::coupling::SimState for NoSim {
    fn voltage_mv(&self, _: nb_sim::serialize::SegmentRef) -> Option<f32> { Some(-70.0) }
    fn time_sec(&self) -> f32 { 0.0 }
    fn channel_count(&self, _: nb_sim::serialize::SegmentRef) -> Option<usize> { None }
    fn has_synapse(&self, _: u32) -> bool { false }
}

/// Evaluate a digit demo with `params.sample` set, run its model once, and
/// return the neuron given the most current, that current, and the trace.
fn run_digit_demo(path: &str, sample: usize) -> (usize, f32, Vec<nb_sim::onnx::Activation>) {
    use nb_sim::onnx::coupling::{ModelInstance, Step};

    let linked = link_blocking(path).unwrap();
    let scene = scene_from_linked(&linked, &[Override::number("params.sample", sample as f64)]).unwrap();
    assert_eq!(scene.neurons.len(), 10);
    let model = &scene.models[0];
    let bytes = std::fs::read(nb_nickel::resolve_path(path, &model.url)).unwrap();
    let program = nb_sim::onnx::Program::from_bytes(&bytes, Default::default()).unwrap();
    let mut instance = ModelInstance::bind(std::sync::Arc::new(program), model.clone()).unwrap();
    let (Step::Ran(effects), trace) = instance.step_traced(&NoSim).unwrap() else { panic!("not ready") };
    let (winner, current) = effects
        .currents
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(seg, i)| (seg.neuron as usize, *i))
        .unwrap();
    (winner, current, trace)
}

#[test]
fn onnx_digits_demo_fires_the_neuron_for_each_sample_digit() {
    for sample in 0..20 {
        let (winner, current, trace) = run_digit_demo(&lib("demos/onnx_digits.ncl"), sample);
        assert_eq!(winner, sample / 2, "sample {sample}");
        assert!(current > 50.0, "sample {sample}: winner gets only {current}");
        let image = trace.iter().find(|a| a.name == "reshape1").expect("the image layer");
        assert_eq!(image.tensor.shape, vec![8, 8]);
    }
}

#[test]
fn onnx_mnist_demo_runs_the_zoo_cnn_on_ascii_digits() {
    let path = lib("demos/onnx_mnist.ncl");
    // Sample 10, a 5, reads as a 6 once its pixels are rounded to 4 levels;
    // models/mnist_samples.ncl records that as its `predicted`.
    let mut correct = 0;
    for sample in 0..20 {
        let (winner, current, _) = run_digit_demo(&path, sample);
        let expected = if sample == 10 { 6 } else { sample / 2 };
        assert_eq!(winner, expected, "sample {sample}");
        assert!(current > 50.0, "sample {sample}: winner gets only {current}");
        correct += usize::from(winner == sample / 2);
    }
    assert_eq!(correct, 19);

    // Sample 20 is the hand-drawn 2 in the demo file.
    let (winner, _, trace) = run_digit_demo(&path, 20);
    assert_eq!(winner, 2);
    let shapes: Vec<_> = trace.iter().map(|a| (a.op, a.tensor.shape.clone())).collect();
    assert!(shapes.contains(&("Conv2d", vec![1, 16, 14, 14])), "{shapes:?}");
    assert!(shapes.contains(&("MaxPool2d", vec![1, 16, 4, 4])), "{shapes:?}");
}

/// Voltages read from a map; segments not in it don't exist.
struct MapSim(std::collections::HashMap<nb_sim::serialize::SegmentRef, f32>);

impl nb_sim::onnx::coupling::SimState for MapSim {
    fn voltage_mv(&self, segment: nb_sim::serialize::SegmentRef) -> Option<f32> {
        self.0.get(&segment).copied()
    }
    fn time_sec(&self) -> f32 {
        0.0
    }
    fn channel_count(&self, segment: nb_sim::serialize::SegmentRef) -> Option<usize> {
        self.0.get(&segment).map(|_| 3)
    }
    fn has_synapse(&self, _: u32) -> bool {
        false
    }
}

#[test]
fn onnx_mnist_neurons_demo_reads_the_painted_retina_and_drives_allen_cells() {
    use nb_sim::onnx::coupling::{ModelInstance, Step};
    use nb_sim::serialize::SegmentRef;

    let path = lib("demos/onnx_mnist_neurons.ncl");
    let linked = link_blocking(&path).unwrap();
    let scene = scene_from_linked(&linked, &[Override::number("params.sample", 6.0)]).unwrap();
    // 784 pixel cells, 4 hidden and 10 digit cells, the Allen ones decimated
    // to about a tenth of the original's 1497 points.
    assert_eq!(scene.neurons.len(), 784 + 14);
    let allen = &scene.neurons[784].neuron;
    assert!((100..200).contains(&allen.segments.len()), "{}", allen.segments.len());

    // The canvas starts as the sample image: a 3 reaching well down the grid.
    let canvas = &scene.canvases[0];
    assert_eq!((canvas.width, canvas.height, canvas.taps.len(), canvas.initial.len()), (28, 28, 784, 784));
    let rows: Vec<usize> = (0..28).filter(|r| canvas.initial[r * 28..(r + 1) * 28].iter().any(|v| *v > 0.0)).collect();
    assert!(rows.len() > 12 && *rows.last().unwrap() > 18, "painted rows {rows:?}");

    // Pixel cells sit at rest (the black level the model's input expects)
    // plus their painted depolarization (1 mV per µA/cm² on the pixel
    // membrane); every other segment rests.
    let nb_sim::serialize::ModelInput::Voltages { offset_mv: rest, .. } = scene.models[0].inputs[0] else {
        panic!("the model reads the retina's voltages")
    };
    let mut volts = std::collections::HashMap::new();
    for n in 784..scene.neurons.len() {
        for s in &scene.neurons[n].neuron.segments {
            volts.insert(SegmentRef { neuron: n as u32, segment: s.id as u32 }, rest);
        }
    }
    for (tap, v) in canvas.taps.iter().zip(&canvas.initial) {
        volts.insert(*tap, rest + v * canvas.max_uamps_per_square_cm);
    }

    let model = &scene.models[0];
    let bytes = std::fs::read(nb_nickel::resolve_path(&path, &model.url)).unwrap();
    let program = nb_sim::onnx::Program::from_bytes(&bytes, Default::default()).unwrap();
    let mut instance = ModelInstance::bind(std::sync::Arc::new(program), model.clone()).unwrap();
    let (Step::Ran(effects), _) = instance.step_traced(&MapSim(volts)).unwrap() else { panic!("not ready") };

    let soma = |n: u32| effects.currents.iter().find(|(t, _)| *t == SegmentRef { neuron: n, segment: 1 }).map_or(0.0, |c| c.1);
    let winner = (0..10).max_by(|a, b| soma(788 + a).total_cmp(&soma(788 + b))).unwrap();
    assert_eq!(winner, 3);
    assert!(soma(788 + 3) > 100.0);
    // Each hidden cell gets current on its dendrite tips, not its soma.
    for h in 0..4u32 {
        let on_tips: Vec<_> = effects.currents.iter().filter(|(t, _)| t.neuron == 784 + h).collect();
        assert!(!on_tips.is_empty() && on_tips.iter().all(|(t, _)| t.segment != 1), "hidden cell {h}: {on_tips:?}");
    }
    assert!(effects.currents.iter().filter(|(t, _)| (784..788).contains(&t.neuron)).any(|(_, c)| *c > 0.0));
}
