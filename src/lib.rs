use wasm_bindgen::prelude::*;

pub mod constants;
pub mod dimension;
pub mod neuron;

#[wasm_bindgen]
pub fn stub() {
    println!("Nice neuron.");
}

#[wasm_bindgen]
pub fn gcd(a: i64, b: i64) -> u64 {
    if b == 0 {
        a.unsigned_abs()
    } else {
        gcd(b, a % b)
    }
}
