use once_cell::sync::OnceCell; // TODO: Bump rustc and use std::cell::OnceCell when stable.
use crossbeam::channel::{Receiver, Sender};
use bevy::prelude::*;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::stimulator::Stimulation;
use crate::neuron::Junction;
use crate::neuron::ecs::Neuron;
use crate::neuron::segment::ecs::Segment;
use crate::gui::load::{load_ffg_scene, GraceSceneSource, IsLoading};
use crate::integrations::grace::GraceSceneSender;


static EXTERNAL_TRIGGER_SENDER: OnceCell<Sender<String>> = OnceCell::new();

pub struct ExternalTriggerPlugin;

#[derive(Resource)]
struct ExternalTriggerReceiver (Receiver<String>);

impl Plugin for ExternalTriggerPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = crossbeam::channel::unbounded();
        EXTERNAL_TRIGGER_SENDER.set(tx).expect("Should be able to set trigger.");
        app.insert_resource(ExternalTriggerReceiver(rx));
        app.add_systems(Update, respond_to_triggers);
    }
}

fn respond_to_triggers(
    trigger_receiver: Res<ExternalTriggerReceiver>,
    mut commands: Commands,
    is_loading: ResMut<IsLoading>,
    mut source: ResMut<GraceSceneSource>,
    mut neurons: Query<(Entity, &Neuron)>,
    mut segments: Query<(Entity, &Segment)>,
    mut junctions: Query<(Entity, &Junction)>,
    mut stimulations: Query<(Entity, &Stimulation)>,
    grace_scene_sender: Res<GraceSceneSender>,
) {
    match trigger_receiver.0.try_recv() {
        Err(_) => {},
        Ok(new_source) => {
            source.0 = new_source;
            load_ffg_scene(commands, is_loading, source, neurons, segments, junctions, stimulations, grace_scene_sender);
        },
    }
}

#[wasm_bindgen]
pub fn set_scene_source(str: String) {
    let sender = EXTERNAL_TRIGGER_SENDER.get().expect("Trigger should be initialized by start()");
    sender.send(str).expect("Should be able to send source to channel.");
}
