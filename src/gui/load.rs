// use bevy::prelude::*;
// use bevy_egui::{egui::{self, Ui}, EguiContexts};
// use bevy::tasks::{IoTaskPool, Task};
// use ehttp::{Request, Response, fetch};
// use crossbeam::channel::unbounded;
//
// use crate::neuron::ecs::Neuron;
// use crate::neuron::Junction;
// use crate::neuron::segment::ecs::Segment;
// use crate::stimulator::{Stimulation};
// use crate::integrations::grace::{GraceScene, GraceSceneSender, GraceSceneReceiver};
// use crate::serialize;
// use crate::neuron::membrane::{MembraneMaterials};
//
// #[derive(Resource)]
// pub struct IsLoading(pub bool);
//
// #[derive(Resource)]
// pub struct GraceSceneSource(pub String);
//
//
// pub fn setup(app: &mut App) {
//   app.insert_resource(IsLoading(false));
//   app.insert_resource(GraceSceneSource("https://raw.githubusercontent.com/reuron/reuron-lib/main/scene.ffg".to_string()));
//   let (tx, rx) = unbounded();
//   app.insert_resource(GraceSceneSender(tx));
//   app.insert_resource(GraceSceneReceiver(rx));
// }
//
// pub fn run_grace_load_widget(
//     mut commands: &mut Commands,
//     mut ui: &mut Ui,
//     mut is_loading: ResMut<IsLoading>,
//     mut source: ResMut<GraceSceneSource>,
//     mut neurons: Query<(Entity, &Neuron)>,
//     mut segments: Query<(Entity, &Segment)>,
//     mut junctions: Query<(Entity, &Junction)>,
//     mut stimulations: Query<(Entity, &Stimulation)>,
//     grace_scene_sender: Res<GraceSceneSender>,
// ) {
//     let response = ui.add(egui::TextEdit::singleline(&mut source.0));
//     if ui.button("Load").clicked() {
//         for (entity, stimulation) in &mut stimulations {
//             commands.entity(entity).despawn();
//         }
//         for (entity, junction) in &mut junctions {
//             commands.entity(entity).despawn();
//         }
//         for (entity, segment) in &mut segments {
//             commands.entity(entity).despawn();
//         }
//         for (entity, neuron) in &mut neurons {
//             commands.entity(entity).despawn();
//         }
//         println!("Requesting from reuron.io: {}", source.0);
//         let request = Request::post("https://reuron.io/interpret", source.0.clone().into_bytes());
//         let sender = (*grace_scene_sender).clone();
//         fetch(request, move |response| {
//             match response {
//                 Err(_) => {
//                     eprintln!("fetch error");
//                 },
//                 Ok(r) => {
//                     println!("response: {:?}", r);
//                     match r.text().ok_or_else(|| {
//                         panic!("No response text!")
//                     }).and_then(|n| serde_json::from_str::<serialize::Scene>(n)) {
//                         Ok(grace_scene) => {
//                             // TODO: Simplify all neurons.
//                             sender.0.send(GraceScene(grace_scene)).expect("Send should succeed");
//
//                         },
//                         Err(e) => {
//                             panic!("{:?}",e)
//                         },
//                     }
//                 },
//             }
//         })
//     }
// }
//
// pub fn handle_loaded_neuron(
//     mut commands: Commands,
//     grace_scene_receiver: Res<GraceSceneReceiver>,
//     mut meshes: ResMut<Assets<Mesh>>,
//     membrane_materials: Res<MembraneMaterials>,
//     mut materials: ResMut<Assets<StandardMaterial>>,
// ) {
//     match grace_scene_receiver.0.try_recv() {
//         Err(_) => {},
//         Ok(n) => {
//             n.spawn(Vec3::new(0.0, 0.0, 0.0), &mut commands, &mut meshes, membrane_materials, &mut materials);
//         }
//     }
// }
