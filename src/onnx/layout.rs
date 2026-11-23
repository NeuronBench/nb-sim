//! Where each element of a traced model goes in the 3D scene.
//!
//! The layout is a pure function of the trace's shapes and edges and of the
//! positions of the segments the model's outputs drive, so the same model
//! and scene always land in the same place:
//!
//! - A layer's depth is its longest path from a graph input. Layers stand as
//!   vertical slabs (in x-y) one `DEPTH_SPACING` apart along -z, the deepest
//!   in front.
//! - An output that drives segments (InjectCurrent, ScaleConductance) is
//!   anchored: element i floats `LIFT` above `taps[i]`. Layers feeding it
//!   one-to-one (its Softmax input, a bias Add, ...) line up behind it in
//!   the same columns.
//! - Every other layer is a slab: `[N, C, H, W]` as one H x W tile per
//!   channel, vectors as rows of cells, tiled to fit the anchors' width.
//!   Layers at the same depth stand side by side.

use std::collections::HashMap;

use bevy::math::Vec3;

use crate::onnx::links::Link;
use crate::onnx::Activation;

/// Distance between successive depths (µm, the scene's unit).
pub const DEPTH_SPACING: f32 = 250.0;
/// Height of the slab region and of anchored cells above their segments.
pub const LIFT: f32 = 200.0;
pub const MAX_SLAB_HEIGHT: f32 = 400.0;
pub const MIN_WIDTH: f32 = 600.0;
pub const MAX_CELL: f32 = 40.0;
/// Edge of a cube in an anchored (or anchor-aligned) layer.
pub const ANCHORED_CUBE: f32 = 30.0;
/// Fraction of a cell a cube fills.
pub const CUBE_FILL: f32 = 0.8;
/// Gap between slabs at the same depth.
const SLAB_GAP: f32 = 100.0;
/// Layers with more elements are not drawn.
pub const MAX_ELEMENTS: usize = 50_000;

/// Where one layer's elements are drawn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
    /// One position per element, in flat (row-major) order; empty when the
    /// layer isn't drawn.
    pub positions: Vec<Vec3>,
    /// Cube edge length.
    pub cube: f32,
}

/// A slab's grid: `tiles` tiles of `h` x `w` cells, `gap` cells apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub tiles: usize,
    pub h: usize,
    pub w: usize,
    pub gap: usize,
}

impl Grid {
    /// How a tensor of `shape` is tiled.
    pub fn of(shape: &[usize]) -> Grid {
        let image = |tiles, h, w| Grid { tiles, h, w, gap: 1 };
        let cells = |n| Grid { tiles: n, h: 1, w: 1, gap: 0 };
        match shape {
            [n, c, h, w] => image(n * c, *h, *w),
            [a, b, c] if *b > 1 && *c > 1 => image(*a, *b, *c),
            [a, b] if *a > 1 && *b > 1 => image(1, *a, *b),
            _ => cells(shape.iter().product()),
        }
    }

    /// Tiles per row and cell size that make the slab as large as possible
    /// within `width` x `height`.
    pub fn fit(&self, width: f32, height: f32) -> (usize, f32) {
        let mut best = (1, 0.0);
        for per_row in 1..=self.tiles.max(1) {
            let rows = self.tiles.div_ceil(per_row).max(1);
            let across = (per_row * self.w + (per_row - 1) * self.gap) as f32;
            let down = (rows * self.h + (rows - 1) * self.gap) as f32;
            let cell = (width / across).min(height / down).min(MAX_CELL);
            if cell > best.1 {
                best = (per_row, cell);
            }
        }
        best
    }

    /// Cell (column, row) of element `index` with `per_row` tiles per row,
    /// row 0 at the top.
    pub fn cell_of(&self, index: usize, per_row: usize) -> (usize, usize) {
        let per_tile = (self.h * self.w).max(1);
        let (tile, within) = (index / per_tile, index % per_tile);
        let (tile_row, tile_col) = (tile / per_row, tile % per_row);
        let col = tile_col * (self.w + self.gap) + within % self.w.max(1);
        let row = tile_row * (self.h + self.gap) + within / self.w.max(1);
        (col, row)
    }

    /// Slab size in cells with `per_row` tiles per row.
    pub fn cells(&self, per_row: usize) -> (usize, usize) {
        let rows = self.tiles.div_ceil(per_row).max(1);
        (per_row * self.w + (per_row - 1) * self.gap, rows * self.h + (rows - 1) * self.gap)
    }
}

/// Lay out a trace. `anchors` maps a trace index to the world positions of
/// the segments its elements drive.
pub fn layout(trace: &[Activation], anchors: &HashMap<usize, Vec<Vec3>>) -> Vec<Placement> {
    let n = trace.len();
    let mut depth = vec![0usize; n];
    for (i, act) in trace.iter().enumerate() {
        depth[i] = act.inputs.iter().flatten().filter(|&&t| t < i).map(|&t| depth[t] + 1).max().unwrap_or(0);
    }

    // Anchors, and layers that line up with them through one-to-one links.
    let mut columns: HashMap<usize, Vec<Vec3>> = anchors
        .iter()
        .filter(|(i, a)| **i < n && !a.is_empty())
        .map(|(i, a)| (*i, a.clone()))
        .collect();
    for i in (0..n).rev() {
        let Some(cols) = columns.get(&i).cloned() else { continue };
        let act = &trace[i];
        if !matches!(act.link, Link::Elementwise | Link::SameIndex | Link::Axis { .. }) {
            continue;
        }
        for t in act.inputs.iter().flatten() {
            if trace[*t].tensor.data.len() == act.tensor.data.len() && !columns.contains_key(t) {
                columns.insert(*t, cols.clone());
            }
        }
    }

    let anchor_points: Vec<Vec3> = anchors.values().flatten().copied().collect();
    let (center, width) = if anchor_points.is_empty() {
        (Vec3::new(0.0, 0.0, -DEPTH_SPACING), MIN_WIDTH)
    } else {
        let sum: Vec3 = anchor_points.iter().copied().sum();
        let (lo, hi) = anchor_points.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p.x), hi.max(p.x)));
        (sum / anchor_points.len() as f32, (hi - lo).max(MIN_WIDTH))
    };
    let front = anchors.keys().filter(|i| **i < n).map(|i| depth[*i]).max().or(depth.iter().copied().max()).unwrap_or(0);
    let z_of = |d: usize| center.z - (front as f32 - d as f32) * DEPTH_SPACING;
    let bottom = center.y + LIFT;

    let mut placements = vec![Placement::default(); n];
    // Slabs grouped by depth, in trace order within a depth.
    let mut by_depth: Vec<Vec<(usize, Grid, usize, f32)>> = vec![Vec::new(); depth.iter().max().map_or(0, |d| d + 1)];
    for (i, act) in trace.iter().enumerate() {
        let count = act.tensor.data.len();
        if count == 0 || count > MAX_ELEMENTS {
            continue;
        }
        if let Some(cols) = columns.get(&i) {
            let z = z_of(depth[i]);
            placements[i] = Placement {
                positions: (0..count.min(cols.len()))
                    .map(|k| Vec3::new(cols[k].x, cols[k].y + LIFT, if anchors.contains_key(&i) { cols[k].z } else { z }))
                    .collect(),
                cube: ANCHORED_CUBE,
            };
            continue;
        }
        let grid = Grid::of(&act.tensor.shape);
        let (per_row, cell) = grid.fit(width, MAX_SLAB_HEIGHT);
        by_depth[depth[i]].push((i, grid, per_row, cell));
    }
    for (d, slabs) in by_depth.iter().enumerate() {
        let widths: Vec<f32> = slabs.iter().map(|(_, g, per_row, cell)| g.cells(*per_row).0 as f32 * cell).collect();
        let total = widths.iter().sum::<f32>() + SLAB_GAP * slabs.len().saturating_sub(1) as f32;
        let mut left = center.x - total / 2.0;
        for ((i, grid, per_row, cell), w) in slabs.iter().zip(&widths) {
            let (_, rows) = grid.cells(*per_row);
            let top = bottom + rows as f32 * cell;
            placements[*i] = Placement {
                positions: (0..trace[*i].tensor.data.len())
                    .map(|k| {
                        let (col, row) = grid.cell_of(k, *per_row);
                        Vec3::new(left + (col as f32 + 0.5) * cell, top - (row as f32 + 0.5) * cell, z_of(d))
                    })
                    .collect(),
                cube: cell * CUBE_FILL,
            };
            left += w + SLAB_GAP;
        }
    }
    placements
}
