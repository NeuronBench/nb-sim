//! The "Models" window: every layer of each running scene model, as a
//! heatmap of its latest values. `[N, C, H, W]` layers (conv and pooling
//! outputs) draw as one small image per channel.

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
/// Image tiles drawn per layer before the rest are elided.
pub const MAX_TILES: usize = 64;
/// Target pixel size when choosing how many tiles share a row.
const TILE_CELL: f32 = 3.0;
/// Largest pixel size in a tile, so one big image stays a reasonable size.
pub const MAX_TILE_CELL: f32 = 8.0;
/// Space between tiles, in points.
pub const TILE_GAP: f32 = 3.0;

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

    if let Some(tiles) = Tiles::fit(shape, ui.available_width()) {
        image_tiles(ui, activation, &tiles, max_abs);
    } else {
        lines(ui, activation, max_abs);
    }
    ui.add_space(4.0);
}

/// Values as wrapped rows of cells.
fn lines(ui: &mut egui::Ui, activation: &Activation, max_abs: f32) {
    let data = &activation.tensor.data;
    let shape = &activation.tensor.shape;
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
}

/// An `[N, C, H, W]` layer as one H x W image per (n, c), wrapped into rows.
fn image_tiles(ui: &mut egui::Ui, activation: &Activation, tiles: &Tiles, max_abs: f32) {
    let data = &activation.tensor.data;
    let (rect, response) = ui.allocate_exact_size(tiles.size(), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let plane = tiles.h * tiles.w;
    for t in 0..tiles.shown() {
        let origin = rect.min + tiles.origin(t);
        for y in 0..tiles.h {
            for x in 0..tiles.w {
                let Some(&v) = data.get(t * plane + y * tiles.w + x) else { break };
                let min = origin + egui::vec2(x as f32 * tiles.cell, y as f32 * tiles.cell);
                let [r, g, b] = color(v, max_abs);
                painter.rect_filled(egui::Rect::from_min_size(min, egui::vec2(tiles.cell, tiles.cell)), 0.0, egui::Color32::from_rgb(r, g, b));
            }
        }
    }
    if let Some(pos) = response.hover_pos() {
        if let Some(index) = tiles.locate(pos - rect.min) {
            if let Some(v) = data.get(index) {
                let at = unravel(index, &activation.tensor.shape);
                response.on_hover_text_at_pointer(format!("{}{at:?} = {v:.5}", activation.name));
            }
        }
    }
    if tiles.count > tiles.shown() {
        ui.label(egui::RichText::new(format!("… {} more images", tiles.count - tiles.shown())).small().weak());
    }
}

/// How an `[N, C, H, W]` layer is drawn: `count` = N * C images of `h` x `w`
/// cells, `per_row` to a row, with `TILE_GAP` between them.
#[derive(Debug, PartialEq)]
pub struct Tiles {
    pub count: usize,
    pub h: usize,
    pub w: usize,
    pub per_row: usize,
    pub cell: f32,
}

impl Tiles {
    /// `None` unless the layer is rank 4 with images at least 3 x 3.
    pub fn fit(shape: &[usize], width: f32) -> Option<Tiles> {
        let &[n, c, h, w] = shape else { return None };
        if h < 3 || w < 3 || n * c == 0 {
            return None;
        }
        let count = n * c;
        let fits = ((width + TILE_GAP) / (w as f32 * TILE_CELL + TILE_GAP)).floor() as usize;
        let per_row = fits.clamp(1, count.min(MAX_TILES));
        let gaps = (per_row - 1) as f32 * TILE_GAP;
        let cell = ((width - gaps) / (per_row * w) as f32).clamp(1.0, MAX_TILE_CELL);
        Some(Tiles { count, h, w, per_row, cell })
    }

    pub fn shown(&self) -> usize {
        self.count.min(MAX_TILES)
    }

    fn rows(&self) -> usize {
        self.shown().div_ceil(self.per_row)
    }

    fn tile_size(&self) -> egui::Vec2 {
        egui::vec2(self.w as f32 * self.cell, self.h as f32 * self.cell)
    }

    pub fn size(&self) -> egui::Vec2 {
        let tile = self.tile_size();
        let (cols, rows) = (self.per_row.min(self.shown()) as f32, self.rows() as f32);
        egui::vec2(cols * tile.x + (cols - 1.0) * TILE_GAP, rows * tile.y + (rows - 1.0) * TILE_GAP)
    }

    /// Top-left of tile `t`, relative to the layout's top-left.
    pub fn origin(&self, t: usize) -> egui::Vec2 {
        let tile = self.tile_size();
        let (row, col) = (t / self.per_row, t % self.per_row);
        egui::vec2(col as f32 * (tile.x + TILE_GAP), row as f32 * (tile.y + TILE_GAP))
    }

    /// The flat index of the value at `pos` (relative to the top-left), or
    /// `None` in a gap or past the last tile.
    pub fn locate(&self, pos: egui::Vec2) -> Option<usize> {
        if pos.x < 0.0 || pos.y < 0.0 {
            return None;
        }
        let tile = self.tile_size();
        let (col, row) = ((pos.x / (tile.x + TILE_GAP)) as usize, (pos.y / (tile.y + TILE_GAP)) as usize);
        let t = row * self.per_row + col;
        if col >= self.per_row || t >= self.shown() {
            return None;
        }
        let inner = pos - self.origin(t);
        let (x, y) = ((inner.x / self.cell) as usize, (inner.y / self.cell) as usize);
        (x < self.w && y < self.h).then_some((t * self.h + y) * self.w + x)
    }
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
