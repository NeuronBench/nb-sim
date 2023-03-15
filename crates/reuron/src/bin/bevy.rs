use bevy::prelude::*;
use bevy::diagnostic::{LogDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};

use reuron::plugin::ReuronPlugin;

pub fn main() {
  App::new()
        .add_plugins(DefaultPlugins)
//         .add_plugin(LogDiagnosticsPlugin::default())
//         .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(ReuronPlugin)
        .run();
}
