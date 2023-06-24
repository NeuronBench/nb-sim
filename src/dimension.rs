use bevy::prelude::{Component, Resource};

// TODO: What are the units?
#[derive(Component, Debug, Clone)]
pub struct Diameter(pub f32);

#[derive(Component, Debug, Clone)]
pub struct AreaSquareMillimeters(pub f32);

#[derive(Component, Debug, Clone)]
pub struct P3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl P3 {
    pub fn squared_distance(&self, other: &P3) -> f32 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z + other.z).powi(2)
    }

    pub fn distance(&self, other: &P3) -> f32 {
        self.squared_distance(other).sqrt()
    }
}

/// Seconds since UNIX epoch.
#[derive(Debug, Clone, Resource)]
pub struct Timestamp(pub f32);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Interval(pub f32);

#[derive(Resource, Debug, Clone)]
pub struct SimulationStepSeconds(pub f32);

#[derive(Debug, Clone)]
pub struct Siemens(pub f32);

#[derive(Debug, Clone)]
pub struct Volts(pub f32);

#[derive(Debug, Clone)]
pub struct MilliVolts(pub f32);

#[derive(Debug, Clone)]
pub struct Kelvin(pub f32);

#[derive(Debug, Clone, PartialEq)]
pub struct Molar(pub f32);

#[derive(Debug, Clone)]
pub struct FaradsPerSquareCm(pub f32);

#[derive(Debug, Clone)]
pub struct Farads(pub f32);

#[derive(Debug, Clone, PartialEq)]
pub struct MicroAmpsPerSquareCm(pub f32);

#[derive(Debug, Clone)]
pub struct MicroAmps(pub f32);

#[derive(Debug, Clone, PartialEq)]
pub struct Hz(pub f32);

#[derive(Debug, Clone)]
pub struct Phase(pub f32);
