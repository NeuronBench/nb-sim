//! Layout and color of the Models window's heatmaps.

use bevy_egui::egui::vec2;
use nb_sim::gui::models_panel::{color, unravel, Grid, Tiles, MAX_CELL, MAX_TILES, MAX_TILE_CELL, MIN_CELL, TILE_GAP};

#[test]
fn last_dim_runs_along_a_line() {
    let g = Grid::fit(&[2, 3, 5], 300.0);
    assert_eq!((g.per_line, g.lines, g.cell), (5, 6, MAX_CELL));
}

#[test]
fn long_rows_wrap() {
    let g = Grid::fit(&[1, 1000], 200.0);
    assert_eq!((g.per_line, g.lines, g.cell), (50, 20, MIN_CELL));
}

#[test]
fn scalars_and_empties() {
    assert_eq!(Grid::fit(&[], 100.0).per_line, 1);
    assert_eq!(Grid::fit(&[0, 4], 100.0).lines, 0);
}

#[test]
fn colors_diverge_from_neutral() {
    assert_eq!(color(0.0, 1.0), [52, 52, 58]);
    assert_eq!(color(2.0, 2.0), [235, 95, 70]);
    assert_eq!(color(-5.0, 2.0), [75, 140, 235]);
    assert_eq!(color(f32::NAN, 1.0), [52, 52, 58]);
}

#[test]
fn unravel_is_row_major() {
    assert_eq!(unravel(7, &[2, 3, 4]), vec![0, 1, 3]);
    assert_eq!(unravel(23, &[2, 3, 4]), vec![1, 2, 3]);
}

#[test]
fn only_rank_4_images_tile() {
    assert!(Tiles::fit(&[1, 10], 300.0).is_none());
    assert!(Tiles::fit(&[8, 8], 300.0).is_none());
    assert!(Tiles::fit(&[3, 1, 2, 4], 300.0).is_none(), "a GRU output is not an image");
    assert!(Tiles::fit(&[1, 0, 28, 28], 300.0).is_none());
}

#[test]
fn conv_channels_wrap_into_rows() {
    // 8 channels of 28x28 at 340 points: 3 per row, at about 4 points a pixel.
    let t = Tiles::fit(&[1, 8, 28, 28], 340.0).unwrap();
    assert_eq!((t.count, t.per_row), (8, 3));
    assert!(t.cell > 3.5, "{}", t.cell);
    assert!(t.size().x <= 340.0 + 1e-3, "{:?}", t.size());
    // A single image grows, up to the cap.
    let t = Tiles::fit(&[1, 1, 28, 28], 340.0).unwrap();
    assert_eq!((t.per_row, t.cell), (1, MAX_TILE_CELL));
    // Few small tiles share one row.
    let t = Tiles::fit(&[1, 16, 4, 4], 340.0).unwrap();
    assert_eq!(t.per_row, 16);
    // Huge channel counts are elided.
    assert_eq!(Tiles::fit(&[1, 500, 4, 4], 340.0).unwrap().shown(), MAX_TILES);
}

#[test]
fn hovering_finds_the_value_under_the_pointer() {
    let t = Tiles { count: 3, h: 2, w: 3, per_row: 2, cell: 10.0 };
    assert_eq!(t.locate(vec2(5.0, 5.0)), Some(0));
    assert_eq!(t.locate(vec2(25.0, 15.0)), Some(5));
    // Second tile starts after a 30-point tile and the gap.
    assert_eq!(t.locate(vec2(30.0 + TILE_GAP + 1.0, 1.0)), Some(6));
    assert_eq!(t.locate(vec2(30.0 + TILE_GAP / 2.0, 1.0)), None, "in the gap");
    // Third tile wraps to the second row; the slot beside it is empty.
    assert_eq!(t.locate(vec2(1.0, 20.0 + TILE_GAP + 11.0)), Some(2 * 6 + 3));
    assert_eq!(t.locate(vec2(30.0 + TILE_GAP + 1.0, 20.0 + TILE_GAP + 1.0)), None);
    assert_eq!(t.size(), vec2(60.0 + TILE_GAP, 40.0 + TILE_GAP));
}
