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
