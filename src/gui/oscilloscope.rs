use bevy::prelude::*;
use bevy_egui::egui::Ui;
use bevy_egui::egui::widgets::plot::{Plot, Line, PlotPoints};

use crate::gui::NextClickAction;

use crate::neuron::segment::{ecs::Segment};
use crate::neuron::membrane::MembraneVoltage;

const N_SOURCES: usize = 4;
const N_SAMPLES: usize = 10000;

#[derive(Debug, Resource)]
pub struct Oscilloscope {
    pub buffers: [ [f32; N_SAMPLES]; N_SOURCES],
    pub sources: [Option<Entity>; N_SOURCES],
    pub write_offset: usize,
    pub trigger_setting: Option<TriggerSetting>,
    pub trigger_sample: Option<usize>,
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
            write_offset: 0,
            trigger_setting: None,
            trigger_sample: None,
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
        let line_0 : PlotPoints = self.buffers[0].iter().enumerate().map(|(x,y)| [x as f64,*y as f64]).clone().collect();
        let line_1 : PlotPoints = self.buffers[1].iter().enumerate().map(|(x,y)| [x as f64,*y as f64]).clone().collect();
        let line_2 : PlotPoints = self.buffers[2].iter().enumerate().map(|(x,y)| [x as f64,*y as f64]).clone().collect();
        let line_3 : PlotPoints = self.buffers[3].iter().enumerate().map(|(x,y)| [x as f64,*y as f64]).clone().collect();
        Plot::new("oscilloscope")
            .view_aspect(2.0)
            .auto_bounds_x()
            .auto_bounds_y()
            .show(ui, |plot_ui| {
                plot_ui.line( Line::new(line_0) );
                plot_ui.line( Line::new(line_1) );
                plot_ui.line( Line::new(line_2) );
                plot_ui.line( Line::new(line_3) );
            });
    }
}

impl Default for Oscilloscope {
    fn default() -> Self {
        Self::init()
    }
}

pub fn step_oscilloscope_system(
    mut oscilloscope: ResMut<Oscilloscope>,
    membrane_voltages: Query<&MembraneVoltage>
) {
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
