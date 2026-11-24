//! The Canvas window: paint a scene canvas with the mouse. Left-drag raises
//! cells (more depolarizing current), right-drag lowers them.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::canvas::SceneCanvases;
use crate::gui::models_panel::color;

/// Seconds of dragging to take a cell at the brush center from 0 to 1.
const FILL_SECS: f32 = 0.25;
/// Largest side of the grid on screen, in points.
const GRID_SIZE: f32 = 280.0;

pub fn run_canvas_panel(
    mut contexts: EguiContexts,
    mut canvases: ResMut<SceneCanvases>,
    time: Res<Time>,
    mut brush: Local<Option<f32>>,
) {
    if canvases.canvases.is_empty() {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let radius = brush.get_or_insert(1.5);
    let dt = time.delta_secs();
    for (i, canvas) in canvases.canvases.iter_mut().enumerate() {
        let (w, h) = (canvas.spec.width as usize, canvas.spec.height as usize);
        if w == 0 || h == 0 {
            continue;
        }
        egui::Window::new(&canvas.spec.name)
            .id(egui::Id::new(("canvas", i)))
            .default_pos([20.0, 240.0 + 40.0 * i as f32])
            .resizable(false)
            .show(ctx, |ui| {
                let cell = (GRID_SIZE / w.max(h) as f32).floor().max(2.0);
                let size = egui::vec2(cell * w as f32, cell * h as f32);
                let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                if let Some(pos) = response.interact_pointer_pos().or_else(|| response.hover_pos()) {
                    let (x, y) = ((pos.x - rect.left()) / cell, (pos.y - rect.top()) / cell);
                    let (left, right) = ui.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down()));
                    let amount = dt / FILL_SECS;
                    if response.is_pointer_button_down_on() || response.dragged() {
                        if left {
                            canvas.brush(x, y, *radius, amount);
                        } else if right {
                            canvas.brush(x, y, *radius, -amount);
                        }
                    }
                }
                let painter = ui.painter_at(rect);
                for (k, v) in canvas.values.iter().enumerate() {
                    let (col, row) = ((k % w) as f32, (k / w) as f32);
                    let min = rect.min + egui::vec2(col * cell, row * cell);
                    let [r, g, b] = color(*v, 1.0);
                    painter.rect_filled(egui::Rect::from_min_size(min, egui::vec2(cell, cell)), 0.0, egui::Color32::from_rgb(r, g, b));
                }
                if let Some(pos) = response.hover_pos() {
                    painter.circle_stroke(pos, *radius * cell, egui::Stroke::new(1.0, egui::Color32::from_white_alpha(120)));
                }
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(radius, 0.5..=6.0).text("brush"));
                    if ui.button("Clear").clicked() {
                        canvas.values.iter_mut().for_each(|v| *v = 0.0);
                    }
                    if ui.button("Reset").clicked() {
                        let mut initial: Vec<f32> = canvas.spec.initial.iter().map(|x| x.clamp(-1.0, 1.0)).collect();
                        initial.resize(w * h, 0.0);
                        canvas.values = initial;
                    }
                });
                ui.label(
                    egui::RichText::new(format!(
                        "Left-drag raises, right-drag lowers; 1 injects {} µA/cm²",
                        canvas.spec.max_uamps_per_square_cm
                    ))
                    .small()
                    .weak(),
                );
            });
    }
}
