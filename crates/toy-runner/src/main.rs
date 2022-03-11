use axum::{routing::post, Router, extract::Extension};
use reuron::dimension::{Interval, Timestamp, MilliVolts, MicroAmpsPerSquareCm};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use serde::Deserialize;
use serde_dhall;

use crate::neuron::solution::INTERSTICIAL_FLUID;
use reuron::constants::BODY_TEMPERATURE;
use reuron::neuron::{self, Neuron};
use reuron::neuron::segment::examples::giant_squid_axon;
use reuron_commands::*;

use toy_runner::ring_buffer::RingBuffer;

#[derive(Debug)]
struct State {
    time: Timestamp,
    time_coefficient: f32,
    simulation_interval: Interval,
    display_rate: f32,
    neuron: reuron::neuron::Neuron,
    simulation_batch_size: usize,
    steps: u64,
    batches: u64,
    waiting_fraction: RingBuffer<f32>,
}

fn initial_state() -> State {
    let mut s =
    State {
        time: Timestamp(0.0),
        steps: 0,
        batches: 0,
        time_coefficient: 0.01,
        simulation_interval: Interval(10e-6),
        neuron: neuron::examples::squid_with_passive_attachment(),
        simulation_batch_size: 10,
        display_rate: 20.0,
        waiting_fraction: RingBuffer::new(10, 0.0),
    };
    s.neuron.segments[0].input_current = MicroAmpsPerSquareCm(10.0);
    s
}

async fn handle_dhall_command(Extension(state): Extension<Arc<Mutex<State>>>, body: String) {
    let command : Command = serde_dhall::from_str(&body).parse().unwrap();
    println!("Parsing command {:?}", command);
    {
        let mut state = state.lock().unwrap();

        match command {
            Command::SetTimeCoefficient(c) => {
                state.time_coefficient = c;
            },
            Command::SetInterval(i) => {
                state.simulation_interval = Interval(i);
            },
            _ => {}
        };
    }
}


#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(initial_state()));
    let watcher_state = state.clone();
    let _watcher = tokio::task::spawn(watch(watcher_state));
    let _runner = tokio::task::spawn(run(state.clone()));

    let app = Router::new().route("/", post(handle_dhall_command)).layer(Extension(state));

    axum::Server::bind(&"0.0.0.0:8000".parse().unwrap()).serve(app.into_make_service()).await.unwrap();

    // run(state).await;

    // let (res1, res2) = tokio::join!(runner, watcher);
    // res1.expect("should join");
    // res2.expect("should join");
}

fn quick_plot_v(v: &MilliVolts) -> String {
    let mut s : Vec<u8> = Vec::from("     .       ".as_bytes());
    let ind = ((v.0 + 150.0) / 20.0) as usize;
    if ind > 0 && ind < s.len() {
        s[ind] = '|' as u8;
    }
    String::from_utf8(s).unwrap()
}

async fn watch(state: Arc<Mutex<State>>) {
    let mut now = SystemTime::now();
    let mut most_recent_display = now.clone();
    loop {
        let wait_interval = {
            let state = state.lock().unwrap();
            println!(
                "step {:.10}, batch {:.5}, avg_wait: {:.1} {}, {}, {}, {}, {}, {:.2}, {:.2} mV",
                // state.steps,
                // state.batches,
                0,
                0,
                state.waiting_fraction.contents().into_iter().sum::<f32>() / state.waiting_fraction.len() as f32,
                quick_plot_v(&state.neuron.segments[0].membrane_potential),
                quick_plot_v(&state.neuron.segments[1].membrane_potential),
                quick_plot_v(&state.neuron.segments[2].membrane_potential),
                quick_plot_v(&state.neuron.segments[3].membrane_potential),
                quick_plot_v(&state.neuron.segments[4].membrane_potential),
                state.time.0 * 1e3,
                state.neuron.segments[1].membrane_potential.0,
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

            // In the next batch, Interval * BatchSize simulation seconds will pass,
            // We want (Interval * BatchSize) / time_coefficient wall clock seconds to pass.
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
                    .neuron
                    .step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            most_recent_simulation_wall_clock_time = next_target_simulation_time;

            now = SystemTime::now();
            let wait_interval = next_target_simulation_time.duration_since(now);

            let this_waiting_fraction = wait_interval.clone().map(|i| i.as_micros() as f32 / inter_batch_wall_clock_interval.as_micros() as f32).unwrap_or(0.0);
            state.waiting_fraction.write(this_waiting_fraction);

            wait_interval

        };

        match wait_interval {
            Ok(interval) if interval > Duration::ZERO => {
                tokio::time::sleep(interval).await;
            }
            Ok(i) => {
                println!("warning: clock is behind by {:?}", i);
            }
            Err(e) => {
            }
        };
    }
}
