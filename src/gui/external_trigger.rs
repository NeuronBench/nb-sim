//! This module allows an external function call to trigger the loading
//! of a new scene.
//!
//! The difficulty comes from the fact that a naive scene loading function
//! natuarally needs access to mutable queries for neuronal components,
//! and a function that we expose via via FFI would not be able to take
//! such complex parameters (nor could the calling Javascript context possibly
//! provide them as arguments).
//!
//! Instead, we want to provide a simple function that takes a string (the
//! expression for the scene to load - usually a nb-lang source URL), and
//! returns ().
//!
//! To integrate this FFI function with the rest of the ECS framework, we
//! use a crossbeam channel. Its Writer end is stored in a global variable,
//! so that the FFI function can access it. The reader end will be polled
//! by ECS, and the polling function therfore has access to the necessary
//! queries for doing a full load and spawn.
use once_cell::sync::OnceCell; // TODO: Bump rustc and use std::cell::OnceCell when stable.
use crossbeam::channel::{Receiver, Sender};
use bevy::prelude::*;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::gui::load::{SceneLoader, SceneSource};

/// The primary interface interface to this module, from nb-sim's perspective.
/// nb-sim only needs to install this plugin, after the Neuron and Gui plugins
/// have been installed.
pub struct ExternalTriggerPlugin;


/// The global variable holding a Sender for new
static EXTERNAL_TRIGGER_SENDER: OnceCell<Sender<String>> = OnceCell::new();

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
    mut source: ResMut<SceneSource>,
    mut loader: ResMut<SceneLoader>,
) {
    if let Ok(new_source) = trigger_receiver.0.try_recv() {
        source.0 = new_source;
        loader.request(true);
    }
}

/// This function is exported via `wasm_bindgen`. It is exported to Javascript clients,
/// so that they can trigger the loading of new scenes by calling it.
#[wasm_bindgen]
pub fn set_scene_source(str: String) {
    let sender = EXTERNAL_TRIGGER_SENDER.get().expect("Trigger should be initialized by start()");
    sender.send(str).expect("Should be able to send source to channel.");
}
