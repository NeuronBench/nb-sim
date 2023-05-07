use bevy::prelude::Component;

use crate::dimension::Molar;
use crate::serialize;

#[derive(Clone, Component, Debug, PartialEq)]
pub struct Solution {
    pub ca_concentration: Molar,
    pub k_concentration: Molar,
    pub na_concentration: Molar,
    pub cl_concentration: Molar,
}

impl Solution {
    pub fn serialize(&self) -> serialize::Solution {
        let Solution {na_concentration, ca_concentration, cl_concentration, k_concentration} = self.clone();
        serialize::Solution {
            na: na_concentration.0,
            ca: ca_concentration.0,
            cl: cl_concentration.0,
            k: k_concentration.0,
        }
    }
}

pub const INTERSTICIAL_FLUID: Solution = Solution {
    na_concentration: Molar(145e-3),
    k_concentration: Molar(5e-3),
    cl_concentration: Molar(110e-3),
    ca_concentration: Molar(2.5e-3),
};

pub const EXAMPLE_CYTOPLASM: Solution = Solution {
    na_concentration: Molar(5e-3),
    k_concentration: Molar(140e-3),
    cl_concentration: Molar(4e-3),
    ca_concentration: Molar(0.1e-6),
};
