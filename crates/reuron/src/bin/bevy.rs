use bevy::prelude::*;

use reuron::plugin::ReuronPlugin;

pub fn main() {
  App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(ReuronPlugin)
        .run();
}
