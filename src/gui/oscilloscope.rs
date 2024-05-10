use bevy::prelude::*;
use bevy_egui::egui::Ui;
use bevy_egui::egui::Color32;
use egui_plot::{Plot, Line, PlotPoints};

use crate::gui::{NextClickAction, SimulationStepSeconds};
use crate::dimension::StepsPerFrame;

use crate::neuron::segment::{ecs::Segment};
use crate::neuron::membrane::MembraneVoltage;

const N_SOURCES: usize = 4;
const N_SAMPLES: usize = 2000;

#[derive(Debug, Resource)]
pub struct Oscilloscope {
    pub buffers: [ [f32; N_SAMPLES]; N_SOURCES],
    pub sources: [Option<Entity>; N_SOURCES],
    pub times: [ f32; N_SAMPLES ],
    pub write_offset: usize,
    pub trigger_setting: Option<TriggerSetting>,
    pub trigger_sample: Option<usize>,
    pub last_known_simulation_step_seconds: SimulationStepSeconds,
}

#[derive(Debug)]
pub struct TriggerSetting {
    pub source_index: usize,
    pub threshold: f32,
}

impl Oscilloscope {
    pub fn init() -> Self {
        Oscilloscope {
            buffers: [ [ 0.0; N_SAMPLES ]; N_SOURCES ],
            sources: [ None; N_SOURCES ],
            times: [ 0.0; N_SAMPLES ],
            write_offset: 0,
            trigger_setting: None,
            trigger_sample: None,
            last_known_simulation_step_seconds: SimulationStepSeconds(0.0),
        }
    }

    pub fn accept_source(
        &mut self,
        ind: usize,
        new_source: Entity) {
            self.sources[ind] = Some(new_source);
    }

    pub fn accept_source_if_available_slot(
        &mut self,
        mut next_click: ResMut<NextClickAction>,
        new_source: Entity
    ) {
        for source in self.sources.iter_mut() {
            if source.is_none() {
                *source = Some(new_source);
                break;
            }
        }
    }

    pub fn plot(&self, ui: &mut Ui) {
        Plot::new("oscilloscope")
            .view_aspect(2.0)
            .auto_bounds_x()
            .auto_bounds_y()
            .show(ui, |plot_ui| {
                for i in 0..4 {
                    let name = (i+1).to_string();
                    let color = [Color32::YELLOW, Color32::LIGHT_GREEN, Color32::LIGHT_RED, Color32::LIGHT_BLUE][i];
                    let line_before_break = self.buffers[i].iter().enumerate().take(self.write_offset - 1).map(|(x,y)| [self.times[x] as f64, *y as f64]).collect::<Vec<_>>();
                    let line_after_break = self.buffers[i].iter().enumerate().skip(self.write_offset).map(|(x,y)| [self.times[x] as f64, *y as f64]).collect::<Vec<_>>();
                    plot_ui.line( Line::new(line_before_break).name(i.to_string()).color(color) );
                    plot_ui.line( Line::new(line_after_break).name(i.to_string()).color(color) );
                }
            });
    }
}

impl Default for Oscilloscope {
    fn default() -> Self {
        Self::init()
    }
}

pub fn step_oscilloscope_system(
    simulation_step_seconds: Res<SimulationStepSeconds>,
    mut oscilloscope: ResMut<Oscilloscope>,
    steps_per_frame: Res<StepsPerFrame>,
    membrane_voltages: Query<&MembraneVoltage>
) {
    if simulation_step_seconds.0 != oscilloscope.last_known_simulation_step_seconds.0 {
        oscilloscope.last_known_simulation_step_seconds.0 = simulation_step_seconds.0;
        oscilloscope.write_offset = 0;
        oscilloscope.buffers = [ [0.0; N_SAMPLES]; N_SOURCES ];
        for i in 0..N_SAMPLES {
            oscilloscope.times[i] = (i as f32) * simulation_step_seconds.0 * steps_per_frame.0 as f32;
        }
    }
    let sources = oscilloscope.sources.clone();
    for (source_index, source) in sources.iter().enumerate() {
        if let Some(entity) = source {
            if let Ok(voltage) = membrane_voltages.get(*entity) {
                let write_offset = oscilloscope.write_offset;
                oscilloscope.buffers[source_index][write_offset] = voltage.0.0;
            }
        }
    }
    oscilloscope.write_offset += 1;
    if oscilloscope.write_offset >= N_SAMPLES {
        oscilloscope.write_offset = 0;
    }
}

pub fn print_oscilloscope_system(
    oscilloscope: Res<Oscilloscope>
) {
    eprintln!("{:?}", oscilloscope);
}
