use bevy::prelude::*;

use crate::onnx::{Onnx, example};

pub struct OnnxPlugin;

impl Plugin for OnnxPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(example());
        // app.add_systems(Update, print_onnx);
    }
}

pub fn print_onnx(onnx: Res<Onnx>) {
    println!("{:?}", onnx.model);
    println!("{:?}", onnx.node_positions);
}

pub fn spawn_onnx(
    commands: &mut Commands,
    onnx: Res<Onnx>,
) {

}
