use bevy::prelude::*;
use bevy_egui::{egui::{self, Ui}, EguiContexts};
use bevy::tasks::{IoTaskPool, Task};
use ehttp::{Request, Response, fetch};
use crossbeam::channel::unbounded;

use crate::neuron::ecs::Neuron;
use crate::neuron::Junction;
use crate::neuron::segment::ecs::Segment;
use crate::integrations::grace::{GraceNeuron, GraceNeuronSender, GraceNeuronReceiver};
use crate::serialize;
use crate::neuron::membrane::{MembraneMaterials};

#[derive(Resource)]
pub struct IsLoading(pub bool);

#[derive(Resource)]
pub struct GraceNeuronSource(pub String);


pub fn setup(app: &mut App) {
  app.insert_resource(IsLoading(false));
  app.insert_resource(GraceNeuronSource("https://raw.githubusercontent.com/imalsogreg/reuron/greg/load/data/swc_neuron.json".to_string()));
  let (tx, rx) = unbounded();
  app.insert_resource(GraceNeuronSender(tx));
  app.insert_resource(GraceNeuronReceiver(rx));
}

pub fn run_grace_load_widget(
    mut commands: &mut Commands,
    mut ui: &mut Ui,
    mut is_loading: ResMut<IsLoading>,
    mut source: ResMut<GraceNeuronSource>,
    mut neurons: Query<(Entity, &Neuron)>,
    mut segments: Query<(Entity, &Segment)>,
    mut junctions: Query<(Entity, &Junction)>,
    grace_neuron_sender: Res<GraceNeuronSender>,
) {
    let response = ui.add(egui::TextEdit::singleline(&mut source.0));
    if ui.button("Load").clicked() {
        for (entity, neuron) in &mut neurons {
            commands.entity(entity).despawn();
        }
        for (entity, segment) in &mut segments {
            commands.entity(entity).despawn();
        }
        for (entity, junction) in &mut junctions {
            commands.entity(entity).despawn();
        }
        let request = Request::get(&source.0);
        let sender = (*grace_neuron_sender).clone();
        fetch(request, move |response| {
            match response {
                Err(_) => {
                    eprintln!("fetch error");
                },
                Ok(r) => {
                    match r.text().ok_or_else(|| {
                        panic!("No response text!")
                    }).and_then(|n| serde_json::from_str::<serialize::Neuron>(n)) {
                        Ok(grace_neuron) => {
                            sender.0.send(GraceNeuron(grace_neuron).simplify()).expect("Send should succeed");

                        },
                        Err(e) => {
                            panic!("{:?}",e)
                        },
                    }
                },
            }
        })
    }
}

pub fn handle_loaded_neuron(
    mut commands: Commands,
    grace_neuron_receiver: Res<GraceNeuronReceiver>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<MembraneMaterials>,
) {
    match grace_neuron_receiver.0.try_recv() {
        Err(_) => {},
        Ok(n) => {
            n.spawn(Vec3::new(0.0, 0.0, 0.0), &mut commands, &mut meshes, materials);
        }
    }
}
