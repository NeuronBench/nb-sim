pub mod external_trigger;
pub mod load;
pub mod oscilloscope;

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy_egui::{egui, EguiContexts};
use bevy_egui::egui::Ui;

use crate::dimension::{
    Timestamp,
    StepsPerFrame,
    SimulationStepSeconds,
    Hz,
    MicroAmpsPerSquareCm,
    Interval
};
// use crate::gui::load::InterpreterUrl;
use crate::gui::oscilloscope::Oscilloscope;
use crate::stimulator::{Stimulator, Envelope, CurrentShape};
// use crate::integrations::grace::GraceSceneSender;
use crate::selection::Selection;


pub fn run_gui(
    // commands: Commands,
    // interpreter_url: Res<InterpreterUrl>,
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    timestamp: Res<Timestamp>,
    simulation_step: ResMut<SimulationStepSeconds>,
    steps_per_frame: ResMut<StepsPerFrame>,
    mut next_click: ResMut<NextClickAction>,
    mut new_stimulators: ResMut<Stimulator>,
    // is_loading: ResMut<load::IsLoading>, // TODO: surface is_loading to user with a spinner.
    // source: ResMut<load::GraceSceneSource>,
    oscilloscope: ResMut<Oscilloscope>,
    // neurons: Query<(Entity, &Neuron)>,
    // segments: Query<(Entity, &Segment)>,
    // junctions: Query<(Entity, &Junction)>,
    // stimulations: Query<(Entity, &Stimulation)>,
    mut selected_stimulators: Query<&mut Stimulator, With<Selection>>,
    // grace_scene_sender: Res<GraceSceneSender>,
) {
    egui::Window::new("NeuronBench").show(contexts.ctx_mut(), |ui| {
        runtime_stats_header(ui, diagnostics, timestamp, steps_per_frame, simulation_step);

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

        let id = ui.make_persistent_id("oscilloscope_header");
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(), id, false
        ).show_header(ui, |ui| {
            ui.label("Oscilloscope")
        })
            .body( |ui| {
                ui.horizontal( |h| {
                   if h.add(egui::Button::new("1")).clicked() {
                       *next_click = NextClickAction::SetVoltageSource(0);
                   }
                   if h.add(egui::Button::new("2")).clicked() {
                       *next_click = NextClickAction::SetVoltageSource(1);
                   }
                   if h.add(egui::Button::new("3")).clicked() {
                       *next_click = NextClickAction::SetVoltageSource(2);
                   }
                   if h.add(egui::Button::new("4")).clicked() {
                       *next_click = NextClickAction::SetVoltageSource(3);
                   }

                } );
                oscilloscope.plot(ui);
            } );

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
    diagnostics: Res<DiagnosticsStore>,
    timestamp: Res<Timestamp>,
    mut steps_per_frame: ResMut<StepsPerFrame>,
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

            let fps_avg = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(|d| {
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

            let spf = steps_per_frame.0.clone();
            let realtime_frac_str = fps_avg
                .map_or(
                    "unknown".to_string(),
                    |v| format!(
                        "{:.4}",
                        v as f32 * simulation_step.0 * spf as f32
                    )
                );
            ui.horizontal(|ui| {
              ui.label("Realtime ratio");
              ui.label(realtime_frac_str);
            });

            ui.add(egui::Slider::from_get_set(
                1.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        simulation_step.0 = v as f32 * 0.0000001;
                    }
                    (simulation_step.0 * 10000000.0) as f64
                }).logarithmic(false).text("Simulation step (microseconds)"));

            ui.add(egui::Slider::from_get_set(
                1.0..=500.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        steps_per_frame.0 = v as usize;
                    }
                    (steps_per_frame.0) as f64
                }).logarithmic(false).text("Steps per frame"));


        });


}

#[derive(Resource)]
pub enum NextClickAction {
    ModifyStimulator,
    SetVoltageSource(usize),
}

impl Default for NextClickAction {
    fn default() -> Self {
        NextClickAction::ModifyStimulator
    }
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
