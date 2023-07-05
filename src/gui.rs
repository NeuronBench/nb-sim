pub mod load;

use bevy::prelude::*;
use bevy::diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin};
use bevy_egui::{egui, EguiContexts};
use bevy_egui::egui::Ui;

use crate::neuron::Junction;
use crate::dimension::{Timestamp, SimulationStepSeconds, Hz, MicroAmpsPerSquareCm, Interval};
use crate::constants::SIMULATION_STEPS_PER_FRAME;
use crate::stimulator::{Stimulator, Stimulation, Envelope, CurrentShape};
use crate::integrations::grace::{GraceSceneSender};
use crate::neuron::ecs::Neuron;
use crate::neuron::segment::ecs::Segment;
use crate::selection::Selection;


pub fn run_gui(
    commands: Commands,
    mut contexts: EguiContexts,
    diagnostics: ResMut<Diagnostics>,
    timestamp: Res<Timestamp>,
    simulation_step: ResMut<SimulationStepSeconds>,
    mut new_stimulators: ResMut<Stimulator>,
    is_loading: ResMut<load::IsLoading>,
    source: ResMut<load::GraceSceneSource>,
    neurons: Query<(Entity, &Neuron)>,
    segments: Query<(Entity, &Segment)>,
    junctions: Query<(Entity, &Junction)>,
    stimulations: Query<(Entity, &Stimulation)>,
    mut selected_stimulators: Query<&mut Stimulator, With<Selection>>,
    grace_scene_sender: Res<GraceSceneSender>,
) {
    egui::Window::new("Reuron").show(contexts.ctx_mut(), |ui| {
        runtime_stats_header(ui, diagnostics, timestamp, simulation_step);

        let id = ui.make_persistent_id("grace_source_header");
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(), id, false
        ).show_header(ui, |ui| {
            ui.label("Source neuron")
        })
        .body(|ui| {
            load::run_grace_load_widget(commands, ui, is_loading, source, neurons, segments, junctions, stimulations, grace_scene_sender);
        });

        let id = ui.make_persistent_id("stimulator_header");
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(), id, false
        ).show_header(ui, |ui| {
            ui.label("Stimulation")
        })
        .body(|ui| {
            match selected_stimulators.get_single_mut() {
                Ok(mut s) => {
                    s.widget(ui);
                },
                Err(_) => {
                    new_stimulators.widget(ui);
                }
            }
        });

        let id = ui.make_persistent_id("build_header");
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(), id, false
        ).show_header(ui, |ui| {
            ui.label("Build")
        })
            .body( |ui| { build_info(ui); } )

    });
}

pub fn build_info(ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label("Version");
        ui.label(format!("{}", env!("VERGEN_GIT_SHA")));
    });
}

pub fn runtime_stats_header(
    ui: &mut Ui,
    diagnostics: ResMut<Diagnostics>,
    timestamp: Res<Timestamp>,
    mut simulation_step: ResMut<SimulationStepSeconds>,
) {

        let id = ui.make_persistent_id("runtime_stats_header");
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            id,
            false
        ).show_header(ui, |ui| {
            ui.label("Runtime Stats");
        })
        .body(|ui| {

            let fps_avg = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS).and_then(|d| {
                d.average()
            });

            ui.horizontal(|ui| {
                ui.label("Simulation time");
                ui.label(format!("{:.2} ms", timestamp.0 * 1000.0))
            });

            let fps_str = fps_avg.map_or("unknown".to_string(),|v| format!("{:.1}", v));
            ui.horizontal(|ui| {
              ui.label("FPS");
              ui.label(fps_str);
            });

            let realtime_frac_str = fps_avg
                .map_or(
                    "unknown".to_string(),
                    |v| format!(
                        "{:.4}",
                        v as f32 * simulation_step.0 * SIMULATION_STEPS_PER_FRAME as f32
                    )
                );
            ui.horizontal(|ui| {
              ui.label("Realtime ratio");
              ui.label(realtime_frac_str);
            });

            // ui.add(egui::Slider::new(&mut simulation_step.0, 0.00001..=0.00010).logarithmic().text("Simulation step"));
            ui.add(egui::Slider::from_get_set(
                1.0..=10.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        simulation_step.0 = v as f32 * 0.0000001;
                    }
                    (simulation_step.0 * 10000000.0) as f64
                }).logarithmic(false).text("Simulation step (microseconds)"));



        });


}

pub fn test_stimulator(
    ui: &mut Ui
) {
    let stim = Stimulator {
        envelope: Envelope {
            period: Interval(2.0),
            onset: Interval(0.1),
            offset: Interval(1.9),
        },
        // current_shape: CurrentShape::SquareWave {
        //     on_current: MicroAmpsPerSquareCm(2.10),
        //     off_current: MicroAmpsPerSquareCm(-0.2),
        // }
        // current_shape: CurrentShape::LinearRamp {
        //     start_current: MicroAmpsPerSquareCm(0.1),
        //     end_current: MicroAmpsPerSquareCm(0.2),
        //     off_current: MicroAmpsPerSquareCm(-0.1),
        // }
        current_shape: CurrentShape::FrequencyRamp {
            on_amplitude: MicroAmpsPerSquareCm(2.0),
            offset_current: MicroAmpsPerSquareCm(-1.0),
            start_frequency: Hz(1.0),
            end_frequency: Hz(100.0),
        }
    };
    stim.plot(ui);
}
