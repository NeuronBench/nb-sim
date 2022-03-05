use axum::{routing::post, Router};
use reuron::dimension::Timestamp;
use std::sync::{Arc, Mutex};

use reuron::neuron;
use reuron::neuron::segment::examples::giant_squid_axon;
use reuron_commands::*;

#[derive(Debug)]
struct State {
    time: Timestamp,
    time_coefficient: f32,
    runtime_simulation_frame_rate: f32,
    segment: reuron::neuron::segment::Segment,
}

fn initial_state() -> State {
    State {
        time: Timestamp(0.0),
        time_coefficient: 0.1,
        runtime_simulation_frame_rate: 1000.0,
        segment: giant_squid_axon(),
    }
}

fn main() {
    println!("Hello, world!!");
}

async fn run(state: Arc<Mutex<State>>) {
    let state = state.write().unwrap();
}
