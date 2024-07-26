pub mod plugin;

use bevy::prelude::*;
use tract_onnx::prelude::*;
use tract_onnx::pb::{GraphProto, ModelProto, NodeProto};
use tract_hir::infer::InferenceOp;

pub use crate::onnx::plugin::OnnxPlugin;

#[derive(Default, Resource)]
pub struct Onnx {
    proto: ModelProto,
    parsed: Graph<InferenceFact, Box<dyn InferenceOp>>,
}

pub fn example() -> Onnx {
    let example_path = format!("{}/sample_data/mnist-12-int8.onnx", env!("CARGO_MANIFEST_DIR"));
    let proto = tract_onnx::onnx()
        .proto_model_for_path(&example_path)
        .expect("Should find onnx example file");
    let parsed = tract_onnx::onnx()
        .model_for_path(&example_path)
        .expect("Should find onnx example file");
    Onnx { proto, parsed }
}

impl Onnx {
    pub fn set_default_positions(&mut self) {

    }
}

// pub fn set_node_position(node: )
