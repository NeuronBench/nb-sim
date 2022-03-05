use serde::Deserialize;
use serde_dhall;

#[derive(Deserialize)]
pub enum Command {
    AddNeuron(AddNeuron),
    SetTimeCoefficient(f32),
}

#[derive(Deserialize)]
pub struct AddNeuron {}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
