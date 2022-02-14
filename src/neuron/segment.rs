use crate::dimension::{Diameter, P3};

#[derive(Clone, Debug)]
pub struct Segment {
    diameter_start: Diameter,
    diameter_end: Diameter,
    pos_start: P3,
    pos_end: P3,
}
