use bevy::prelude::*;
use bevy_egui::{egui::{self, Ui}, EguiContexts};

use crate::neuron::ecs::Neuron;
use crate::neuron::Junction;
use crate::neuron::segment::ecs::Segment;

#[derive(Resource)]
pub struct IsLoading(pub bool);

#[derive(Resource)]
pub struct GraceNeuronSource(pub String);


pub fn setup(app: &mut App) {
  app.insert_resource(IsLoading(false));
  app.insert_resource(GraceNeuronSource("https://raw.githubusercontent.com/imalsogreg/reuron/main/crates/reuron/sample_data/swc_neuron.json".to_string()));
}

pub fn run_grace_load_widget(
    mut commands: &mut Commands,
    mut ui: &mut Ui,
    mut is_loading: ResMut<IsLoading>,
    mut source: ResMut<GraceNeuronSource>,
    mut neurons: Query<(Entity, &Neuron)>,
    mut segments: Query<(Entity, &Segment)>,
    mut junctions: Query<(Entity, &Junction)>,
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
    }
}
