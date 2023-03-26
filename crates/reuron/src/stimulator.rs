use bevy::prelude::{Assets, Color, Component, FromWorld, Handle, Resource, StandardMaterial, World};
use bevy_egui::egui::widgets::plot::{Plot, Line, PlotPoints};
use bevy_egui::egui::{self, Ui};
use std::default::Default;

use crate::dimension::{Interval, Hz, MicroAmpsPerSquareCm, Timestamp};

#[derive(Debug, Clone, Component, Resource)]
pub struct Stimulator {
    pub envelope: Envelope,
    pub current_shape: CurrentShape,
}

impl Default for Stimulator {
    fn default() -> Self {
        Stimulator {
            envelope: Envelope {
                period: Interval(0.1),
                onset: Interval(0.0),
                offset: Interval(0.050),
            },
            current_shape: CurrentShape::SquareWave {
                on_current: MicroAmpsPerSquareCm(50.0),
                off_current:MicroAmpsPerSquareCm(-10.0)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Envelope {
    pub period: Interval,
    pub onset: Interval,
    pub offset: Interval,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CurrentShape {
    SquareWave {
        on_current: MicroAmpsPerSquareCm,
        off_current: MicroAmpsPerSquareCm
    },
    LinearRamp {
        start_current: MicroAmpsPerSquareCm,
        end_current: MicroAmpsPerSquareCm,
        off_current: MicroAmpsPerSquareCm,
    },
    FrequencyRamp {
        on_amplitude: MicroAmpsPerSquareCm,
        offset_current: MicroAmpsPerSquareCm,
        start_frequency: Hz,
        end_frequency: Hz,
    }

}

impl Stimulator {
    pub fn current(&self, t: Timestamp) -> MicroAmpsPerSquareCm {

        let cycle_start = Timestamp(t.0.div_euclid(self.envelope.period.0));
        let cycle_time = Interval(t.0.rem_euclid(self.envelope.period.0));
        let envelope_time = Interval(cycle_time.0 - self.envelope.onset.0);
        let envelope_length = Interval(self.envelope.offset.0 - self.envelope.onset.0);
        let window_completion = envelope_time.0 / envelope_length.0;
        let in_envelope = window_completion >= 0.0 && window_completion <= 1.0;

        match &self.current_shape {
            CurrentShape::SquareWave { on_current, off_current } =>
                if in_envelope { on_current.clone() } else { off_current.clone() },
            CurrentShape::LinearRamp { start_current, end_current, off_current } =>
                if in_envelope {
                    let i = window_completion * (end_current.0 - start_current.0) + start_current.0;
                    MicroAmpsPerSquareCm(i)
                } else {
                    off_current.clone()
                },
            CurrentShape::FrequencyRamp { on_amplitude, offset_current, start_frequency, end_frequency } => {
                if in_envelope {
                    let freq = window_completion * (end_frequency.0 - start_frequency.0) + start_frequency.0;
                    let phase = freq * 2.0 * std::f32::consts::PI * envelope_time.0;
                    let i = on_amplitude.0 * phase.sin() + offset_current.0.clone();
                    MicroAmpsPerSquareCm(i)
                } else {
                    offset_current.clone()
                }

            }
        }
    }

    pub fn plot(&self, ui: &mut Ui) {
        let currents : PlotPoints = (0..2000).map(|t| {
            let timestamp = Timestamp(t.clone() as f32 * 0.0005);
            let current = self.current(timestamp.clone());
            // let current = timestamp.clone();
            [timestamp.0 as f64, current.0 as f64]
        }).collect();
        let line = Line::new(currents);
        Plot::new("stimulator_plot")
            .view_aspect(2.0)
            .auto_bounds_x()
            .auto_bounds_y()
            .show(ui, |plot_ui| plot_ui.line(line));
    }

    pub fn widget(&mut self, ui: &mut Ui) {
        let Envelope { ref mut period, ref mut onset, ref mut offset } = &mut self.envelope;
        let mut current_shape = &mut self.current_shape;
        // let current_shape_copy = current_shape.clone();

        ui.add(egui::Slider::from_get_set(1.0..=10000.0, move |v: Option<f64>| {
            if let Some(v) = v {
                period.0 = v as f32 * 0.001;
            }
            period.0 as f64 * 1000.0
        }).logarithmic(true).text("Period (ms)"));

        ui.add(egui::Slider::from_get_set(1.0..=10000.0, move |v: Option<f64>| {
            if let Some(v) = v {
                onset.0 = v as f32 * 0.001;
            }
            onset.0 as f64 * 1000.0
        }).logarithmic(true).text("Onset Time (ms)"));

        ui.add(egui::Slider::from_get_set(1.0..=10000.0, move |v: Option<f64>| {
            if let Some(v) = v {
                offset.0 = v as f32 * 0.001;
            }
            offset.0 as f64 * 1000.0
        }).logarithmic(true).text("Offsete Time (ms)"));

        let default_square_wave = match &mut current_shape {
            c@CurrentShape::SquareWave {..} => c.clone(),
            _ => CurrentShape::SquareWave {
                on_current: MicroAmpsPerSquareCm(50.0),
                off_current: MicroAmpsPerSquareCm(-10.0)
            }
        };


        let default_linear_ramp = match &mut current_shape {
            c@CurrentShape::LinearRamp {..} => c.clone(),
            _ => CurrentShape::LinearRamp {
                start_current: MicroAmpsPerSquareCm(10.0),
                end_current: MicroAmpsPerSquareCm(50.0),
                off_current: MicroAmpsPerSquareCm(-10.0),
            }
        };

        let default_frequency_ramp = match &mut current_shape {
            c@CurrentShape::FrequencyRamp {..} => c.clone(),
            _ => CurrentShape::FrequencyRamp {
                on_amplitude: MicroAmpsPerSquareCm(50.0),
                offset_current: MicroAmpsPerSquareCm(-10.0),
                start_frequency: Hz(10.0),
                end_frequency: Hz(100.0),
            }
        };


        ui.horizontal(|ui| {
            ui.selectable_value(current_shape, default_square_wave, "Square");
            ui.selectable_value(current_shape, default_linear_ramp, "Linear Ramp");
            ui.selectable_value(current_shape, default_frequency_ramp, "Frequency Ramp");
        });

        match &mut current_shape {
            CurrentShape::SquareWave {ref mut on_current, ref mut off_current} => {

                ui.add(egui::Slider::from_get_set(-100.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        on_current.0 = v as f32;
                    }
                    on_current.0 as f64
                }).logarithmic(false).text("Onset Current (uAmps)"));

                ui.add(egui::Slider::from_get_set(-100.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        off_current.0 = v as f32;
                    }
                    off_current.0 as f64
                }).logarithmic(false).text("Offset Current (uAmps)"));

            },

            CurrentShape::LinearRamp { ref mut start_current, ref mut end_current, ref mut off_current
            } => {

                ui.add(egui::Slider::from_get_set(-100.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        start_current.0 = v as f32;
                    }
                    start_current.0 as f64
                }).logarithmic(false).text("Start Current (uAmps)"));

                ui.add(egui::Slider::from_get_set(-100.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        end_current.0 = v as f32;
                    }
                    end_current.0 as f64
                }).logarithmic(false).text("End Current (uAmps)"));

                ui.add(egui::Slider::from_get_set(-100.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        off_current.0 = v as f32;
                    }
                    off_current.0 as f64
                }).logarithmic(false).text("Off Current (uAmps)"));

            },

            CurrentShape::FrequencyRamp {
                ref mut on_amplitude, ref mut offset_current, ref mut start_frequency, ref mut end_frequency
            } => {

                ui.add(egui::Slider::from_get_set(0.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        on_amplitude.0 = v as f32;
                    }
                    on_amplitude.0 as f64
                }).logarithmic(false).text("Amplitude (uAmps)"));

                ui.add(egui::Slider::from_get_set(-100.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        offset_current.0 = v as f32;
                    }
                    offset_current.0 as f64
                }).logarithmic(false).text("Offset Current (uAmps)"));

                ui.add(egui::Slider::from_get_set(1.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        start_frequency.0 = v as f32;
                    }
                    start_frequency.0 as f64
                }).logarithmic(false).text("Start Frequency (Hz)"));

                ui.add(egui::Slider::from_get_set(1.0..=100.0, move |v: Option<f64>| {
                    if let Some(v) = v {
                        end_frequency.0 = v as f32;
                    }
                    end_frequency.0 as f64
                }).logarithmic(false).text("End Frequency (Hz)"));

            },
        }

        self.plot(ui);
        dbg!(&self);

    }
}

#[derive(Resource)]
pub struct StimulatorMaterials {
    pub handles: Vec<Handle<StandardMaterial>>,
    pub current_range: (MicroAmpsPerSquareCm, MicroAmpsPerSquareCm),
    pub len: usize,
}

impl FromWorld for StimulatorMaterials {
    fn from_world(world: &mut World) -> Self {
        let mut material_assets = world.get_resource_mut::<Assets<StandardMaterial>>()
            .expect("Can get material assets");
        let len = 200;
        let current_range = (MicroAmpsPerSquareCm(-10.0),MicroAmpsPerSquareCm(10.0));
        let mut handles = Vec::new();
        let unselected_handles: Vec<_> = (0..100).map(|i| {
          let intensity_range = 1.0;
          let intensity = (i as f32) / len as f32 * intensity_range;
          let color = Color::rgba(intensity, 0.0, 1.0 - intensity, 0.9);
          let mut material : StandardMaterial = color.clone().into();
          material.emissive = Color::rgb_linear(
              1.0 * intensity,
              1.0 * intensity * intensity,
              1.0 * intensity * intensity
          );
          let handle = material_assets.add(material);
          handle
        }).collect();
        handles.extend(unselected_handles);
        let selected_handles: Vec<_> = (0..100).map(|i| {
          let intensity_range = 1.0;
          let intensity = (i as f32 - 50.0) / len as f32 * intensity_range;
          let color = Color::rgba(intensity, 0.0, 1.0 - intensity,0.95);
          let mut material : StandardMaterial = color.clone().into();
          material.emissive = Color::rgb_linear(
              30.0 * intensity,
              30.0 * intensity * intensity,
              30.0 * intensity * intensity
          );
          let handle = material_assets.add(material);
          handle
        }).collect();
        handles.extend(selected_handles);
        StimulatorMaterials { handles, current_range, len }
    }
}

impl StimulatorMaterials {
    pub fn from_selected_and_current(
        &self,
        selected: bool,
        current: &MicroAmpsPerSquareCm
    ) -> Handle<StandardMaterial> {
        let i_min = self.current_range.0.0;
        let i_max = self.current_range.1.0;
        let selected_offset = if selected { 100 } else { 0 };
        let index = (((current.0 - i_min) / (i_max - i_min)).min(0.9) * self.len as f32 * 0.5).floor() as usize + selected_offset;
        self.handles[index].clone()
    }
}
