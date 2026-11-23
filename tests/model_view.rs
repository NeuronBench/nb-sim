//! The 3D model view: element connectivity in traces, the layout, picking.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use bevy::math::{Dir3, Ray3d, Vec3};
use burn::tensor::Device;
use nb_sim::gui::model_view::pick;
use nb_sim::onnx::layout::{layout, Grid, LIFT};
use nb_sim::onnx::links::{consumers, sources};
use nb_sim::onnx::{Activation, HostTensor, Program};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Vec<Named>,
}

#[derive(Deserialize)]
struct Named {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn fixture(name: &str) -> (Program, Vec<HostTensor>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sample_data/onnx");
    let program = Program::from_bytes(&std::fs::read(dir.join(format!("{name}.onnx"))).unwrap(), Device::default()).unwrap();
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.json"))).unwrap()).unwrap();
    let inputs = fixture.inputs.into_iter().map(|n| HostTensor::new(n.shape, n.data)).collect();
    (program, inputs)
}

/// Every graph-input element that element `index` of `trace[layer]` is
/// computed from, following `sources` back through the trace.
fn closure(trace: &[Activation], layer: usize, index: usize) -> HashSet<(usize, usize)> {
    let mut seen = HashSet::new();
    let mut stack = vec![(layer, index)];
    let mut inputs = HashSet::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if trace[node.0].op == "Input" {
            inputs.insert(node);
        }
        stack.extend(sources(trace, node.0, node.1));
    }
    inputs
}

/// Perturb each input element in turn: every output element that changes
/// must have that input element among its traced sources.
fn sources_cover_perturbations(name: &str) {
    let (program, inputs) = fixture(name);
    let (base, trace) = program.run_traced(&inputs).unwrap();
    let outputs: Vec<usize> = program
        .outputs()
        .iter()
        .map(|p| trace.iter().rposition(|a| a.name == p.name).unwrap())
        .collect();
    let input_layers: Vec<usize> = (0..trace.len()).filter(|i| trace[*i].op == "Input").collect();
    let mut closures: HashMap<(usize, usize), HashSet<(usize, usize)>> = HashMap::new();
    let mut checked = 0;
    for (k, layer) in input_layers.iter().enumerate() {
        for e in 0..inputs[k].data.len() {
            let mut moved = inputs.clone();
            moved[k].data[e] += 3.0;
            let out = program.run(&moved).unwrap();
            for (o, (before, after)) in base.iter().zip(&out).enumerate() {
                for (i, (b, a)) in before.data.iter().zip(&after.data).enumerate() {
                    if (a - b).abs() > 1e-5 {
                        let deps = closures.entry((outputs[o], i)).or_insert_with(|| closure(&trace, outputs[o], i));
                        assert!(
                            deps.contains(&(*layer, e)),
                            "{name}: output {o}[{i}] moved with input {k}[{e}], which its sources miss"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0, "{name}: no output moved");
}

#[test]
fn traced_sources_cover_every_dependency() {
    for name in ["conv2d", "mlp", "matmul", "ops", "clamp"] {
        sources_cover_perturbations(name);
    }
}

#[test]
fn conv_sources_are_the_receptive_field() {
    let (program, inputs) = fixture("conv2d");
    let (_, trace) = program.run_traced(&inputs).unwrap();
    let y = trace.iter().position(|a| a.name == "y").unwrap();
    // y[0, 0, 0, 0] depends on batch 0 only, and on a bounded window.
    let deps = closure(&trace, y, 0);
    let per_batch = 3 * 7 * 9;
    assert!(!deps.is_empty() && deps.iter().all(|(_, e)| *e < per_batch), "{deps:?}");
    assert!(deps.len() < per_batch, "{} of {per_batch}", deps.len());
    // The first conv reads all 3 input channels through its kernel.
    let conv = trace.iter().position(|a| a.op == "Conv2d").unwrap();
    let first = sources(&trace, conv, 0);
    let channels: HashSet<usize> = first.iter().map(|(_, e)| e / (7 * 9)).collect();
    assert_eq!(channels, HashSet::from([0, 1, 2]));
}

#[test]
fn consumers_invert_sources() {
    let (program, inputs) = fixture("conv2d");
    let (_, trace) = program.run_traced(&inputs).unwrap();
    for layer in 0..trace.len() {
        for index in [0, trace[layer].tensor.data.len() / 2] {
            let mut expected = Vec::new();
            for (t, act) in trace.iter().enumerate().skip(layer + 1) {
                for i in 0..act.tensor.data.len() {
                    if sources(&trace, t, i).contains(&(layer, index)) {
                        expected.push((t, i));
                    }
                }
            }
            assert_eq!(consumers(&trace, layer, index), expected, "layer {layer} element {index}");
        }
    }
}

#[test]
fn grids_tile_channels_and_wrap_vectors() {
    assert_eq!(Grid::of(&[1, 8, 28, 28]), Grid { tiles: 8, h: 28, w: 28, gap: 1 });
    assert_eq!(Grid::of(&[8, 8]), Grid { tiles: 1, h: 8, w: 8, gap: 1 });
    assert_eq!(Grid::of(&[1, 256]), Grid { tiles: 256, h: 1, w: 1, gap: 0 });
    // Eight 28x28 channels in a 2700 x 400 region: one row is widest.
    let (per_row, cell) = Grid::of(&[1, 8, 28, 28]).fit(2700.0, 400.0);
    assert_eq!(per_row, 8);
    assert!(cell > 10.0 && cell * 28.0 <= 400.0, "{cell}");
    // Row-major: element 28 is the second row of the first tile.
    let g = Grid::of(&[1, 2, 28, 28]);
    assert_eq!(g.cell_of(28, 2), (0, 1));
    assert_eq!(g.cell_of(28 * 28, 2), (29, 0));
}

fn mnist() -> Option<Program> {
    let dir = std::env::var("NB_LIB_DIR").unwrap_or_else(|_| format!("{}/../nb-lib", env!("CARGO_MANIFEST_DIR")));
    let bytes = std::fs::read(format!("{dir}/models/mnist_softmax.onnx")).ok()?;
    Some(Program::from_bytes(&bytes, Device::default()).unwrap())
}

#[test]
fn mnist_outputs_sit_above_their_neurons_and_the_rest_behind() {
    let Some(program) = mnist() else {
        eprintln!("skipping: nb-lib's mnist_softmax.onnx not found");
        return;
    };
    let input = HostTensor::new(vec![1, 1, 28, 28], (0..784).map(|i| (i % 255) as f32).collect());
    let (_, trace) = program.run_traced(&[input]).unwrap();
    let probs = trace.iter().rposition(|a| a.name == "probs").unwrap();
    let somas: Vec<Vec3> = (0..10).map(|d| Vec3::new(d as f32 * 300.0, 0.0, 0.0)).collect();
    let anchors = HashMap::from([(probs, somas.clone())]);
    let placed = layout(&trace, &anchors);
    assert_eq!(placed, layout(&trace, &anchors), "deterministic");

    for (d, soma) in somas.iter().enumerate() {
        assert_eq!(placed[probs].positions[d], *soma + Vec3::Y * LIFT);
    }
    // The logits feeding the softmax line up behind the probabilities.
    let logits = trace[probs].inputs[0].unwrap();
    for d in 0..10 {
        let p = placed[logits].positions[d];
        assert_eq!((p.x, p.y), (somas[d].x, LIFT));
        assert!(p.z < 0.0);
    }
    for (l, p) in placed.iter().enumerate() {
        assert_eq!(p.positions.len(), trace[l].tensor.data.len(), "{}", trace[l].name);
        if l != probs {
            assert!(p.positions.iter().all(|q| q.z < 0.0), "{} should stand behind the neurons", trace[l].name);
        }
        // Cubes in a layer don't overlap.
        if let [a, b, ..] = p.positions[..] {
            assert!(a.distance(b) >= p.cube, "{}", trace[l].name);
        }
    }
    // Deeper layers are nearer the neurons.
    let input_z = placed[0].positions[0].z;
    assert!(input_z < placed[logits].positions[0].z);
}

#[test]
fn picking_finds_the_nearest_cube_on_the_ray() {
    use nb_sim::onnx::layout::Placement;
    let near = Placement { positions: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)], cube: 4.0 };
    let far = Placement { positions: vec![Vec3::new(10.0, 0.0, -50.0)], cube: 4.0 };
    let layers = [near, far];
    let ray = |x: f32| Ray3d::new(Vec3::new(x, 0.0, 100.0), Dir3::NEG_Z);
    assert_eq!(pick(&[&layers[..]], ray(10.0)), Some((0, 0, 1)));
    assert_eq!(pick(&[&layers[..]], ray(0.5)), Some((0, 0, 0)));
    assert_eq!(pick(&[&layers[..]], ray(5.0)), None);
    assert_eq!(pick(&[&layers[1..]], ray(10.0)), Some((0, 0, 0)));
}
