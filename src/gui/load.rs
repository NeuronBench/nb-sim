use bevy::prelude::*;
use bevy_egui::{egui::{self, Ui}, EguiContexts};
use bevy::tasks::{IoTaskPool, Task};
use ehttp::{Request, Response, fetch};
use crossbeam::channel::unbounded;
use web_sys::{Location, window};

use crate::neuron::ecs::Neuron;
use crate::neuron::Junction;
use crate::neuron::segment::ecs::Segment;
use crate::stimulator::{Stimulation};
use crate::integrations::grace::{
    GraceScene,
    GraceSceneSender,
    GraceSceneReceiver
};
use crate::serialize;
use crate::neuron::membrane::{MembraneMaterials};

#[derive(Resource)]
pub struct IsLoading(pub bool);

#[derive(Resource)]
pub struct GraceSceneSource(pub String);

impl FromWorld for GraceSceneSource {

    #[cfg(target_arch = "wasm32")]
    fn from_world(world: &mut World) -> Self {
        GraceSceneSource(window_location_scene())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn from_world(world: &mut World) -> Self {
        GraceSceneSource("".to_string())
    }

}

#[cfg(target_arch = "wasm32")]
/// Parse an ffg expression from the browser's window.location.
pub fn window_location_scene() -> String {
    let search = window().expect("should have window").location().search();
    match search {
        Ok(s) => {
            let s = s.clone().to_string();
            if s.len() > 0 {
                let params = querystring::querify(&s[1..]);
                match params.iter().find(|(k,v)| k.clone() == "scene") {
                    Some((_,v)) => { v.to_string() },
                    None => { "".to_string() },
                }
            } else {
                "".to_string()
            }
        },
        Err(_) => {
            "".to_string()
        }
    }
}


pub fn setup(app: &mut App) {
  app.insert_resource(IsLoading(false));
  // app.insert_resource(GraceSceneSource("https://raw.githubusercontent.com/reuron/reuron-lib/main/scene.ffg".to_string()));
  app.init_resource::<GraceSceneSource>();
  let (tx, rx) = unbounded();
  app.insert_resource(GraceSceneSender(tx));
  app.insert_resource(GraceSceneReceiver(rx));
  app.add_startup_system(startup_load_ffg_scene);
}

pub fn startup_load_ffg_scene(
    mut commands: Commands,
    mut is_loading: ResMut<IsLoading>,
    mut source: ResMut<GraceSceneSource>,
    mut neurons: Query<(Entity, &Neuron)>,
    mut segments: Query<(Entity, &Segment)>,
    mut junctions: Query<(Entity, &Junction)>,
    mut stimulations: Query<(Entity, &Stimulation)>,
    grace_scene_sender: Res<GraceSceneSender>,
) {
    if source.0.len() > 0 {
        eprintln!("Doing startup scene load with {}", source.0);
        load_ffg_scene(commands, is_loading, source, neurons, segments, junctions, stimulations, grace_scene_sender);
    } else {
        eprintln!("Skipping startup scene load");
    }
}

pub fn load_ffg_scene(
    mut commands: Commands,
    mut is_loading: ResMut<IsLoading>,
    mut source: ResMut<GraceSceneSource>,
    mut neurons: Query<(Entity, &Neuron)>,
    mut segments: Query<(Entity, &Segment)>,
    mut junctions: Query<(Entity, &Junction)>,
    mut stimulations: Query<(Entity, &Stimulation)>,
    grace_scene_sender: Res<GraceSceneSender>,

) {

    for (entity, stimulation) in &mut stimulations {
        commands.entity(entity).despawn();
    }
    for (entity, junction) in &mut junctions {
        commands.entity(entity).despawn();
    }
    for (entity, segment) in &mut segments {
        commands.entity(entity).despawn();
    }
    for (entity, neuron) in &mut neurons {
        commands.entity(entity).despawn();
    }
    eprintln!("Requesting from reuron.io: {}", source.0);
    let request = Request::post("https://reuron.io/interpret", source.0.clone().into_bytes());
    let sender = (*grace_scene_sender).clone();
    fetch(request, move |response| {
        match response {
            Err(_) => {
                eprintln!("fetch error");
            },
            Ok(r) => {
                eprintln!("response: {:?}", r);
                match r.text().ok_or_else(|| {
                    panic!("No response text!")
                }).and_then(|n| serde_json::from_str::<serialize::Scene>(n)) {
                    Ok(grace_scene) => {
                        // TODO: Simplify all neurons.
                        sender.0.send(GraceScene(grace_scene)).expect("Send should succeed");

                    },
                    Err(e) => {
                        eprintln!("Failed to interpret: {:?}", e);
                    },
                }
            },
        }
    })

}

pub fn run_grace_load_widget(
    mut commands: Commands,
    mut ui: &mut Ui,
    mut is_loading: ResMut<IsLoading>,
    mut source: ResMut<GraceSceneSource>,
    mut neurons: Query<(Entity, &Neuron)>,
    mut segments: Query<(Entity, &Segment)>,
    mut junctions: Query<(Entity, &Junction)>,
    mut stimulations: Query<(Entity, &Stimulation)>,
    grace_scene_sender: Res<GraceSceneSender>,
) {
    let response = ui.add(egui::TextEdit::singleline(&mut source.0));
    if ui.button("Load").clicked() {
        load_ffg_scene(commands, is_loading, source, neurons, segments, junctions, stimulations, grace_scene_sender);
    }
}

pub fn handle_loaded_neuron(
    mut commands: Commands,
    grace_scene_receiver: Res<GraceSceneReceiver>,
    mut meshes: ResMut<Assets<Mesh>>,
    membrane_materials: Res<MembraneMaterials>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    match grace_scene_receiver.0.try_recv() {
        Err(_) => {},
        Ok(n) => {
            n.spawn(Vec3::new(0.0, 0.0, 0.0), &mut commands, &mut meshes, membrane_materials, &mut materials);
        }
    }
}
