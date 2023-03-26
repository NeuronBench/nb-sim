use serde::{Deserialize, Serialize};
use serde_dhall::{StaticType};

#[derive(Debug, Deserialize, Serialize, StaticType)]
pub enum Command {
    AddNeuron(AddNeuron),
    SetTimeCoefficient(f32),
    SetInterval(f32),
}

#[derive(Debug, Deserialize, Serialize, StaticType)]
pub struct AddNeuron {}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_dhall;
    #[test]
    fn command_serialization() {
        let cmd = Command::SetTimeCoefficient(0.001);
        let serialized = serde_dhall::serialize(&cmd).static_type_annotation().to_string().unwrap();
        assert_eq!(serialized, "< AddNeuron: {} | SetTimeCoefficient: Double >.SetTimeCoefficient 0.0010000000474974513".to_string());
    }
}
