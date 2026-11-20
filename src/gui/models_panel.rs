//! The "Models" window: every layer of each running scene model, as a
//! heatmap of its latest values.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::onnx::plugin::SceneModels;
use crate::onnx::Activation;

/// Cell edge in points: never smaller than `MIN_CELL` (values wrap onto more
/// lines instead), never larger than `MAX_CELL`.
pub const MIN_CELL: f32 = 4.0;
pub const MAX_CELL: f32 = 18.0;
/// Lines drawn per layer before the rest is elided.
const MAX_LINES: usize = 48;

const NEUTRAL: [f32; 3] = [52.0, 52.0, 58.0];
const POSITIVE: [f32; 3] = [235.0, 95.0, 70.0];
const NEGATIVE: [f32; 3] = [75.0, 140.0, 235.0];

pub fn run_models_panel(mut contexts: EguiContexts, models: Res<SceneModels>) {
    if models.slots.iter().all(|s| s.activations.is_empty()) {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Window::new("Models")
        .default_pos([380.0, 460.0])
        .default_width(340.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().max_height(520.0).show(ui, |ui| {
                for slot in models.slots.iter().filter(|s| !s.activations.is_empty()) {
                    let name = slot.url.rsplit('/').next().unwrap_or(&slot.url);
                    egui::CollapsingHeader::new(format!("{name} ({} runs)", slot.runs))
                        .id_salt(&slot.url)
                        .default_open(true)
                        .show(ui, |ui| {
                            for activation in &slot.activations {
                                layer(ui, activation);
                            }
                        });
                }
            });
            ui.label(
                egui::RichText::new("Blue is negative, red positive; each layer is scaled to its own largest |value|.")
                    .small()
                    .weak(),
            );
        });
}

fn layer(ui: &mut egui::Ui, activation: &Activation) {
    let data = &activation.tensor.data;
    let shape = &activation.tensor.shape;
    let max_abs = data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let summary = match data.as_slice() {
        [v] => format!("= {v:.4}"),
        _ => {
            let (lo, hi) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            format!("{lo:.3} … {hi:.3}")
        }
    };
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(&activation.name).strong());
        ui.label(egui::RichText::new(format!("{} {:?}", activation.op, shape)).weak());
        ui.label(egui::RichText::new(summary).monospace().small());
    });

    let grid = Grid::fit(shape, ui.available_width());
    let shown_lines = grid.lines.min(MAX_LINES);
    let size = egui::vec2(grid.per_line as f32 * grid.cell, shown_lines as f32 * grid.cell);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    for line in 0..shown_lines {
        for col in 0..grid.per_line {
            let Some(&v) = data.get(line * grid.per_line + col) else { break };
            let min = rect.min + egui::vec2(col as f32 * grid.cell, line as f32 * grid.cell);
            let cell = egui::Rect::from_min_size(min, egui::vec2(grid.cell, grid.cell)).shrink(0.5);
            let [r, g, b] = color(v, max_abs);
            painter.rect_filled(cell, 0.0, egui::Color32::from_rgb(r, g, b));
        }
    }
    if let Some(pos) = response.hover_pos() {
        let rel = pos - rect.min;
        let index = (rel.y / grid.cell) as usize * grid.per_line + (rel.x / grid.cell) as usize;
        if let Some(v) = data.get(index) {
            let at = unravel(index, shape);
            response.on_hover_text_at_pointer(format!("{}{at:?} = {v:.5}", activation.name));
        }
    }
    if grid.lines > MAX_LINES {
        let hidden = data.len().saturating_sub(MAX_LINES * grid.per_line);
        ui.label(egui::RichText::new(format!("… {hidden} more values")).small().weak());
    }
    ui.add_space(4.0);
}

/// How a layer's values are laid out: `per_line` cells per line, `lines`
/// lines, each cell `cell` points wide. The last dim runs along a line and
/// the other dims stack lines, unless the last dim is too long for the width,
/// in which case all values wrap row-major.
#[derive(Debug, PartialEq)]
pub struct Grid {
    pub per_line: usize,
    pub lines: usize,
    pub cell: f32,
}

impl Grid {
    pub fn fit(shape: &[usize], width: f32) -> Grid {
        let total: usize = shape.iter().product();
        if total == 0 {
            return Grid { per_line: 0, lines: 0, cell: MIN_CELL };
        }
        let cols = shape.last().copied().unwrap_or(1);
        let fits = ((width / MIN_CELL).floor() as usize).max(1);
        let per_line = if cols <= fits { cols } else { fits };
        let lines = total.div_ceil(per_line);
        let cell = (width / per_line as f32).clamp(MIN_CELL, MAX_CELL);
        Grid { per_line, lines, cell }
    }
}

/// Diverging color: blue for negative, red for positive, dark gray at 0,
/// scaled by `max_abs`.
pub fn color(v: f32, max_abs: f32) -> [u8; 3] {
    let t = if max_abs > 0.0 && v.is_finite() { (v / max_abs).clamp(-1.0, 1.0) } else { 0.0 };
    let end = if t < 0.0 { NEGATIVE } else { POSITIVE };
    let t = t.abs();
    std::array::from_fn(|i| (NEUTRAL[i] + (end[i] - NEUTRAL[i]) * t).round() as u8)
}

/// The multi-index of flat row-major `index` in `shape`.
pub fn unravel(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut at = vec![0; shape.len()];
    for (i, &d) in shape.iter().enumerate().rev() {
        if d > 0 {
            at[i] = index % d;
            index /= d;
        }
    }
    at
}
