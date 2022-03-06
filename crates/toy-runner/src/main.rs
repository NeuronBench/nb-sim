use axum::{routing::post, Router};
use reuron::dimension::{Interval, Timestamp};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use crate::neuron::solution::INTERSTICIAL_FLUID;
use reuron::constants::BODY_TEMPERATURE;
use reuron::neuron;
use reuron::neuron::segment::examples::giant_squid_axon;
// use reuron_commands::*;

#[derive(Debug)]
struct State {
    time: Timestamp,
    time_coefficient: f32,
    simulation_interval: Interval,
    display_rate: f32,
    segment: reuron::neuron::segment::Segment,
    simulation_batch_size: usize,
    steps: u64,
    batches: u64,
}

fn initial_state() -> State {
    State {
        time: Timestamp(0.0),
        steps: 0,
        batches: 0,
        time_coefficient: 0.001,
        simulation_interval: Interval(10e-6),
        segment: giant_squid_axon(),
        simulation_batch_size: 100,
        display_rate: 10.0,
    }
}

#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(initial_state()));
    let watcher_state = state.clone();
    let _watcher = tokio::task::spawn(watch(watcher_state));
    // let runner = tokio::task::spawn(async move { run(state) });
    run(state).await;

    // let (res1, res2) = tokio::join!(runner, watcher);
    // res1.expect("should join");
    // res2.expect("should join");
}

async fn watch(state: Arc<Mutex<State>>) {
    let mut now = SystemTime::now();
    let mut most_recent_display = now.clone();
    loop {
        let wait_interval = {
            let state = state.lock().unwrap();
            println!(
                "step {:.10}, batch {:.5}, {:.2} ms: {:.1} mV",
                state.steps,
                state.batches,
                state.time.0 * 1e3,
                state.segment.membrane_potential.0
            );

            let inter_display_interval = Duration::from_micros((1e6 / state.display_rate) as u64);
            let next_target_display_time = most_recent_display + inter_display_interval;
            most_recent_display = next_target_display_time;

            now = SystemTime::now();
            next_target_display_time.duration_since(now)
        };

        match wait_interval {
            Ok(interval) if interval > Duration::ZERO => {
                tokio::time::sleep(interval).await;
            }
            Ok(i) => {
                println!("warning: clock is behind by {:?}", i);
            }
            Err(e) => {
                println!("warning: clock monotonicity error: {:?}", e);
            }
        };
    }
}

async fn run(state: Arc<Mutex<State>>) {
    let mut now = SystemTime::now();
    let mut most_recent_simulation_wall_clock_time = now.clone();

    loop {
        let wait_interval = {
            let mut state = state.lock().unwrap();

            state.batches += 1;
            let inter_batch_wall_clock_interval = Duration::from_micros(
                (1e6 * state.simulation_batch_size as f32 * state.simulation_interval.0 as f32
                    / state.time_coefficient) as u64,
            );
            let next_target_simulation_time =
                most_recent_simulation_wall_clock_time + inter_batch_wall_clock_interval;

            let batch_start_time = SystemTime::now();

            let interval = state.simulation_interval.clone();
            for _ in 0..state.simulation_batch_size {
                state.steps += 1;
                state.time = Timestamp(state.time.0 + state.simulation_interval.0);
                state
                    .segment
                    .step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            most_recent_simulation_wall_clock_time = next_target_simulation_time;

            now = SystemTime::now();

            next_target_simulation_time.duration_since(now)
        };
        dbg!(&wait_interval);

        match wait_interval {
            Ok(interval) if interval > Duration::ZERO => {
                tokio::time::sleep(interval).await;
            }
            Ok(i) => {
                println!("warning: clock is behind by {:?}", i);
            }
            Err(e) => {
                println!("warning: clock monotonicity error: {:?}", e);
            }
        };
    }
}
