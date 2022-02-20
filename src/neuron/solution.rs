use crate::dimension::Molar;

#[derive(Clone, Debug)]
pub struct Solution {
    pub ca_concentration: Molar,
    pub k_concentration: Molar,
    pub na_concentration: Molar,
}

pub const intersticial_fluid: Solution = Solution {
    ca_concentration: Molar(1e-9),
    k_concentration: Molar(1e-9),
    na_concentration: Molar(1e-8),
};

pub const example_neuron: Solution = Solution {
    ca_concentration: Molar(1e-9),
    k_concentration: Molar(1e-10),
    na_concentration: Molar(1e-7),
};

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dimension::Molar;
}
