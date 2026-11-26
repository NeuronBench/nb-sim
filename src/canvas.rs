//! Canvases: grids of injected current that the user paints (see
//! [`crate::serialize::Canvas`] and the Canvas window in
//! `gui::canvas_panel`). `run_models` adds their currents to the segments'
//! `ModelCurrent` along with the models' own.

use bevy::prelude::*;

use crate::serialize::{Canvas, SegmentRef};

/// The scene's canvases and what has been painted on them.
#[derive(Resource, Default)]
pub struct SceneCanvases {
    pub canvases: Vec<CanvasState>,
}

pub struct CanvasState {
    pub spec: Canvas,
    /// Cell values in -1..1, row-major.
    pub values: Vec<f32>,
    /// The frame on show, for a canvas with `frames`.
    pub frame: Option<usize>,
}

impl SceneCanvases {
    /// Replace the scene's canvases, keeping the painting of any canvas
    /// whose name and size are unchanged (so moving a slider doesn't wipe
    /// it).
    pub fn load(&mut self, specs: &[Canvas]) {
        let mut old = std::mem::take(&mut self.canvases);
        self.canvases = specs
            .iter()
            .map(|spec| {
                let cells = (spec.width * spec.height) as usize;
                let kept = old
                    .iter()
                    .position(|c| c.spec.name == spec.name && c.spec.width == spec.width && c.spec.height == spec.height)
                    .map(|i| old.swap_remove(i).values);
                let values = kept.unwrap_or_else(|| {
                    let mut v: Vec<f32> = spec.initial.iter().map(|x| x.clamp(-1.0, 1.0)).collect();
                    v.resize(cells, 0.0);
                    v
                });
                CanvasState { spec: spec.clone(), values, frame: None }
            })
            .collect();
    }

    /// Show each streaming canvas's frame for simulated time `t`. Returns
    /// whether any canvas changed frame.
    pub fn advance(&mut self, t: f32) -> bool {
        let mut changed = false;
        for c in &mut self.canvases {
            let n = c.spec.frames.len();
            if n == 0 || !(c.spec.frame_sec > 0.0) {
                continue;
            }
            let k = ((t / c.spec.frame_sec).max(0.0) as usize) % n;
            if c.frame != Some(k) {
                let cells = (c.spec.width * c.spec.height) as usize;
                c.values = c.spec.frames[k].iter().map(|x| x.clamp(-1.0, 1.0)).collect();
                c.values.resize(cells, 0.0);
                c.frame = Some(k);
                changed = true;
            }
        }
        changed
    }

    /// The label of the frame on show in the named canvas, if it has one.
    pub fn label(&self, name: &str) -> Option<(usize, u32)> {
        let c = self.canvases.iter().find(|c| c.spec.name == name)?;
        let k = c.frame?;
        Some((k, *c.spec.labels.get(k)?))
    }

    /// The label of frame `frame` of the named canvas.
    pub fn label_of(&self, name: &str, frame: usize) -> Option<u32> {
        self.canvases.iter().find(|c| c.spec.name == name)?.spec.labels.get(frame).copied()
    }

    /// Current per tapped segment, in µA/cm². A segment under several cells
    /// gets the sum.
    pub fn currents(&self) -> impl Iterator<Item = (SegmentRef, f32)> + '_ {
        self.canvases.iter().flat_map(|c| {
            c.spec.taps.iter().zip(&c.values).map(|(tap, v)| (*tap, v * c.spec.max_uamps_per_square_cm))
        })
    }
}

impl CanvasState {
    /// Raise (positive `amount`) or lower every cell within `radius` cells of
    /// (`x`, `y`), by `amount` at the center falling off to 0 at the edge.
    pub fn brush(&mut self, x: f32, y: f32, radius: f32, amount: f32) {
        let (w, h) = (self.spec.width as i32, self.spec.height as i32);
        let r = radius.max(0.5);
        let reach = r.ceil() as i32;
        let (cx, cy) = (x.floor() as i32, y.floor() as i32);
        for row in (cy - reach).max(0)..=(cy + reach).min(h - 1) {
            for col in (cx - reach).max(0)..=(cx + reach).min(w - 1) {
                let d = ((col as f32 + 0.5 - x).powi(2) + (row as f32 + 0.5 - y).powi(2)).sqrt();
                let falloff = (1.0 - d / r).max(0.0);
                if falloff > 0.0 {
                    let cell = &mut self.values[(row * w + col) as usize];
                    *cell = (*cell + amount * falloff).clamp(-1.0, 1.0);
                }
            }
        }
    }
}

/// Step streaming canvases to the current frame (before models read them).
pub fn advance_canvas_frames(timestamp: Res<crate::dimension::Timestamp>, canvases: Option<ResMut<SceneCanvases>>) {
    let Some(mut canvases) = canvases else { return };
    let t = timestamp.0;
    if canvases.canvases.iter().any(|c| !c.spec.frames.is_empty()) {
        canvases.advance(t);
    }
}
