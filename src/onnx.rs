pub mod plugin;

use bevy::prelude::*;
use bevy::render::mesh::PrimitiveTopology;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use bevy::render::render_asset::RenderAssetUsages;
use std::collections::HashMap;
use tract_onnx::prelude::*;
use tract_onnx::pb::NodeProto;
use tract_hir::internal::GenericFactoid;
use tract_hir::infer::InferenceOp;

pub use crate::onnx::plugin::OnnxPlugin;

// The Onnx model resource.
#[derive(Default, Resource)]
pub struct Onnx {
    /// The parsed Onnx model.
    model: Graph<InferenceFact, Box<dyn InferenceOp>>,
    /// A mapping from node names to their spatial positions.
    node_positions: HashMap<String, Vec<f32>>,
}

impl Onnx {
    // Overwrite node_positions with a new node_positions, where
    // nodes are stacked one on top of the other according to their order.
    pub fn set_default_positions(&mut self) {
        let positions = self.model.nodes.iter().enumerate().map(|(i, node)| {
            let x = 0.0;
            let y = 0.0;
            let z = i as f32 * 20.0;
            (node.name.clone(), vec![x, y, z])
        }).collect();
        self.node_positions = positions;
    }
}

pub fn spawn_onnx_model(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    onnx: Res<Onnx>,
) {
    for node in onnx.model.nodes.iter() {
        // Spawn a node:
        //  - a 2d rectangle textured according to its values.
        //  - a Transform based on node_positions
        if node.outputs.len() != 1 {
            eprintln!("Node {} has {} outputs, expected 1", node.name, node.outputs.len());
            continue;
        }
        let values = &node.outputs[0].fact;
        if let GenericFactoid::Only(tensor_ref) = &values.value {
            let position = onnx.node_positions.get(&node.name).expect("Node position not found");
            match tensor_to_2d_image(tensor_ref) {
                None => {},
                Some(((h,w), image)) => {
                    let image_handle = asset_server.add(image);
                    let mesh_handle = meshes.add(Rectangle::new(w as f32 * 10.0,h as f32 * 10.0));
                    let transform = Transform::from_translation(Vec3::new(position[0], position[1], position[2]));
                    let material_handle = materials.add(StandardMaterial {
                        base_color_texture: Some(image_handle),
                        alpha_mode: AlphaMode::Blend,
                        unlit: true,
                        ..default()
                    });
                    eprintln!("ABOUT TO SPAWN NODE");
                    commands.spawn(PbrBundle {
                        mesh: mesh_handle.clone(),
                        material: material_handle,
                        transform,
                        ..default()
                    });
                }
            }
        }
    }
}


/// Get the id and spatial position of a node from the proto format.
fn node_position(node: &NodeProto) -> Option<(String, Vec<f32>)> {
    let position = node.attribute.iter().find(|attr| attr.name == "position");
    if let Some(position) = position {
        let position = &position.floats;
        if position.len() <= 3 {
            return None;
        }
        Some((node.name.clone(), position.clone()))
    } else {
        None
    }
}


/// Get the (h,w) and Image from a tensor.
fn tensor_to_2d_image(tensor: &Tensor) -> Option<((u32, u32), Image)> {
    let data_f32 = tensor.cast_to::<f32>().expect("should be able to cast to f32");
    let data = data_f32.to_array_view::<f32>().expect("should be f32 tensor");
    let extent = match tensor.shape() {
        [1, h, w] => Some((h,w)),
        [h, w] => Some((h,w)),
        _ => {
            eprintln!("Tensor has unexpected shape {:?}", tensor.shape());
            None
        },
    };
    extent.map(|(height,width)| {
        eprintln!("CONSTRUCTING IMAGEDATE FOR TENSOR WITH SHAPE {:?}", tensor.shape());
        let mut image_data : Vec<f32> = vec![0.0; height * width * 4];
        eprintln!("ABOUT TO CONSTRUCT IMAGEDATA");
        for y in 0..(*height as u64) {
            for x in 0..(*width as u64) {
                let value = data[[y as usize, x as usize]] / 25.0;
                let i = ((y * *width as u64 + x) * 4) as usize;
                dbg!(&value);
                image_data[i] = value.clamp(0.0, 1.0);
                image_data[i + 1] = 0.0;
                image_data[i + 2] = (value * -1.0).clamp(0.0, 1.0);
                image_data[i + 3] = value.abs();
            }
        }
        eprintln!("SUCCESS CONSTRUCTED IMAGE DATA");
        let image = Image::new(
            Extent3d { width: *width as u32, height: *height as u32, depth_or_array_layers: 1 },
            TextureDimension::D2,
            image_data.iter().flat_map(|&f| f.to_le_bytes()).collect(),
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        ((*height as u32, *width as u32), image)
    })
}

// Generate an example Onnx model from the mnist-12-int8.onnx file.
pub fn example() -> Onnx {
    let example_path = format!("{}/sample_data/mnist-12-int8.onnx", env!("CARGO_MANIFEST_DIR"));
    let proto = tract_onnx::onnx()
        .proto_model_for_path(&example_path)
        .expect("Should find onnx example file")
        .graph
        .expect("Should have a graph");
    let node_positions = proto.node.iter().filter_map(node_position).collect();
    let model = tract_onnx::onnx()
        .model_for_path(&example_path)
        .expect("Should find onnx example file");
    let mut onnx = Onnx { model, node_positions };
    onnx.set_default_positions();
    onnx
}
