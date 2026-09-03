//! The "Scene" window: scene source, live parameter sliders, and diagnostics.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::gui::load::{IsLoading, SceneDiagnostics, SceneLoader, SceneParams, SceneSource};

const ERROR_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 96, 96);

pub fn run_scene_panel(
    mut contexts: EguiContexts,
    mut source: ResMut<SceneSource>,
    mut loader: ResMut<SceneLoader>,
    mut params: ResMut<SceneParams>,
    diagnostics: Res<SceneDiagnostics>,
    is_loading: Res<IsLoading>,
    time: Res<Time>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let now = time.elapsed_secs_f64();
    egui::Window::new("Scene")
        .default_pos([10.0, 460.0])
        .default_width(360.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut source.0).hint_text("scene.ncl URL or path").desired_width(280.0));
                if ui.button("Load").clicked() {
                    loader.request(true);
                }
            });
            if is_loading.0 {
                ui.label("Loading…");
            }

            if !params.specs.is_empty() {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.heading("Parameters");
                    if ui.small_button("Reset").clicked() {
                        params.reset(now);
                    }
                });
                let specs = params.specs.clone();
                for spec in &specs {
                    let current = *params.values.get(&spec.name).unwrap_or(&spec.default);
                    let mut value = current;
                    let response = match (spec.min, spec.max) {
                        (Some(lo), Some(hi)) => ui.add(egui::Slider::new(&mut value, lo..=hi).text(&spec.name)),
                        _ => ui
                            .horizontal(|ui| {
                                let speed = (spec.default.abs().max(1.0)) * 0.01;
                                let r = ui.add(egui::DragValue::new(&mut value).speed(speed));
                                ui.label(&spec.name);
                                r
                            })
                            .inner,
                    };
                    if let Some(doc) = &spec.doc {
                        response.on_hover_text(doc);
                    }
                    if value != current {
                        params.set(&spec.name, value, now);
                    }
                }
            }

            if !diagnostics.0.is_empty() {
                ui.separator();
                egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    for d in &diagnostics.0 {
                        ui.colored_label(ERROR_COLOR, d.render());
                    }
                });
            }
        });
}
