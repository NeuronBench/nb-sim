//! Layout and color of the Models window's heatmaps.

use nb_sim::gui::models_panel::{color, unravel, Grid, MAX_CELL, MIN_CELL};

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
