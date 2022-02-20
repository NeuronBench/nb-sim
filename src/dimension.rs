#[derive(Debug, Clone)]
pub struct Diameter(pub f32);

impl Diameter {
    pub fn mm(d: f32) -> Diameter {
        Diameter(d / 1000.0)
    }
}

#[derive(Debug, Clone)]
pub struct P3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl P3 {
    fn squared_distance(&self, other: &P3) -> f32 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z + other.z).powi(2)
    }

    fn distance(&self, other: &P3) -> f32 {
        self.squared_distance(other).sqrt()
    }
}

/// Seconds since UNIX epoch.
#[derive(Debug, Clone)]
pub struct Timestamp(pub f32);

#[derive(Debug, Clone)]
pub struct Interval(pub f32);

#[derive(Debug, Clone)]
pub struct Siemens(pub f32);

#[derive(Debug, Clone)]
pub struct Volts(pub f32);

#[derive(Debug, Clone)]
pub struct MilliVolts(pub f32);

#[derive(Debug, Clone)]
pub struct Celcius(pub f32);

#[derive(Debug, Clone)]
pub struct Molar(pub f32);

#[derive(Debug, Clone)]
pub struct FaradsPerArea(pub f32);

#[derive(Debug, Clone)]
pub struct Farads(pub f32);
