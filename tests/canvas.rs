//! Painted canvases: brushing, keeping the painting across reloads, and the
//! currents they inject.

use nb_sim::canvas::SceneCanvases;
use nb_sim::serialize::{Canvas, SegmentRef};

fn canvas(name: &str, width: u32, height: u32, initial: Vec<f32>) -> Canvas {
    Canvas {
        name: name.into(),
        width,
        height,
        taps: (0..width * height).map(|i| SegmentRef { neuron: i, segment: 1 }).collect(),
        max_uamps_per_square_cm: 10.0,
        initial,
        frames: vec![],
        frame_sec: 0.0,
        labels: vec![],
    }
}

#[test]
fn initial_values_are_clamped_and_padded() {
    let mut c = SceneCanvases::default();
    c.load(&[canvas("a", 2, 2, vec![0.5, 3.0, -2.0])]);
    assert_eq!(c.canvases[0].values, vec![0.5, 1.0, -1.0, 0.0]);
}

#[test]
fn brushing_raises_the_center_most_and_stays_in_range() {
    let mut c = SceneCanvases::default();
    c.load(&[canvas("a", 5, 5, vec![])]);
    let state = &mut c.canvases[0];
    state.brush(2.5, 2.5, 1.5, 0.6);
    let v = |x: usize, y: usize| state.values[y * 5 + x];
    assert!(v(2, 2) > v(3, 2) && v(3, 2) > 0.0, "{:?}", state.values);
    assert_eq!(v(0, 0), 0.0, "outside the brush");
    for _ in 0..10 {
        state.brush(2.5, 2.5, 1.5, 0.6);
    }
    assert_eq!(state.values[12], 1.0);
    for _ in 0..30 {
        state.brush(2.5, 2.5, 1.5, -0.6);
    }
    assert_eq!(state.values[12], -1.0);
    // Brushing off the edge only touches cells inside the grid.
    state.brush(-0.2, 4.9, 1.0, 0.5);
    assert!(state.values[20] > 0.0);
}

#[test]
fn reloading_keeps_the_painting_of_the_same_canvas() {
    let mut c = SceneCanvases::default();
    c.load(&[canvas("retina", 2, 2, vec![]), canvas("other", 1, 1, vec![])]);
    c.canvases[0].values[3] = 0.75;
    c.canvases[1].values[0] = -0.5;
    // Same name and size: kept. A resized canvas starts again.
    c.load(&[canvas("other", 2, 1, vec![0.1, 0.2]), canvas("retina", 2, 2, vec![1.0; 4])]);
    assert_eq!(c.canvases[0].values, vec![0.1, 0.2]);
    assert_eq!(c.canvases[1].values, vec![0.0, 0.0, 0.0, 0.75]);
}

#[test]
fn cells_inject_their_value_times_the_maximum() {
    let mut c = SceneCanvases::default();
    c.load(&[canvas("a", 2, 1, vec![0.5, -1.0])]);
    let currents: Vec<_> = c.currents().collect();
    assert_eq!(
        currents,
        vec![(SegmentRef { neuron: 0, segment: 1 }, 5.0), (SegmentRef { neuron: 1, segment: 1 }, -10.0)]
    );
}

#[test]
fn frames_stream_in_order_and_cycle() {
    let mut spec = canvas("digits", 2, 1, vec![]);
    spec.frames = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5]];
    spec.frame_sec = 0.1;
    spec.labels = vec![7, 3, 9];
    let mut c = SceneCanvases::default();
    c.load(&[spec]);
    assert!(c.advance(0.05));
    assert_eq!(c.canvases[0].values, vec![1.0, 0.0]);
    assert_eq!(c.label("digits"), Some((0, 7)));
    assert!(!c.advance(0.07), "same frame: nothing to do");
    c.advance(0.15);
    assert_eq!(c.label("digits"), Some((1, 3)));
    c.advance(0.25);
    assert_eq!(c.canvases[0].values, vec![0.5, 0.0], "short frames are padded");
    c.advance(0.31);
    assert_eq!(c.label("digits"), Some((0, 7)), "cycles");
    assert_eq!(c.label_of("digits", 2), Some(9));
}
