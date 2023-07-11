pub mod channel;
pub mod membrane;
pub mod segment;
pub mod solution;
pub mod synapse;
pub mod network;

use crate::dimension::Diameter;
use crate::neuron::solution::Solution;

use bevy::prelude::{Component, Entity};

pub mod ecs {
    use bevy::prelude::Component;
    #[derive(Component)]
    pub struct Neuron;
}

#[derive(Component)]
pub struct Junction {
    pub first_segment: Entity,
    pub second_segment: Entity,
    pub pore_diameter: Diameter,
}

