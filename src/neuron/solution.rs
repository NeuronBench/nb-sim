use crate::dimension::Molar;

#[derive(Clone, Debug)]
pub struct Solution {
    pub ca_concentration: Molar,
    pub k_concentration: Molar,
    pub na_concentration: Molar,
    pub cl_concentration: Molar,
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

#[cfg(test)]
pub mod tests {}
