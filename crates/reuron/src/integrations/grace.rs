use bevy::prelude::*;
use serde_json;

use crate::dimension::{FaradsPerSquareCm, MilliVolts, Diameter, MicroAmpsPerSquareCm};
use crate::serialize;



pub mod sample {
    use std::include_str;
    pub fn neuron() -> &'static str {
        include_str!("../../sample_data/swc_neuron.json")
    }
}

#[cfg(test)]
pub mod tests {

    use super::*;
    use crate::serialize;
    use crate::dimension::{Molar};

    #[test]
    pub fn parse_solution() {

        // Parsing warmup.
        let solution : serialize::Solution = serde_json::from_str(
            "{\"k\": 1, \"na\": 1, \"ca\": 0, \"cl\": 0}"
        ).expect("should parse");
        assert!(solution.k - 1.0 < 1e-7);

        let neuron : serialize::Neuron = serde_json::from_str(sample::neuron()).expect("should parse");
    }

}
