//! Models drawn in the 3D scene: every traced value as a slab of glowing
//! cubes (see [`crate::onnx::layout`]), brighter where the activation is
//! larger, warm for positive and cool for negative. Outputs that drive
//! neurons float above them, joined by a line to the segment they drive.
//! Hovering a cube draws its connections: the elements it reads (cyan) and
//! the ones that read it (orange).

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts};

use crate::neuron::segment::ecs::SegmentAddress;
use crate::onnx::layout::{layout, Placement};
use crate::onnx::links::{consumers, sources, unravel};
use crate::onnx::plugin::{ModelState, SceneModels};
use crate::onnx::Activation;
use crate::serialize::{ModelOutput, SegmentRef};

/// Lines drawn per direction for a hovered element; the strongest are kept.
const MAX_LINES: usize = 400;
const WARM: [f32; 3] = [1.0, 0.45, 0.15];
const COOL: [f32; 3] = [0.2, 0.55, 1.0];
/// Brightness of a zero activation, so every layer's shape stays visible.
const FLOOR: f32 = 0.035;
const GAIN: f32 = 1.4;

pub struct ModelViewPlugin;

impl Plugin for ModelViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ModelView>()
            .add_systems(Update, (sync_model_view, color_model_view, hover_model_view, draw_model_view).chain());
    }
}

#[derive(Resource)]
pub struct ModelView {
    pub enabled: bool,
    /// What the current meshes were built for.
    key: Option<String>,
    slots: Vec<ViewSlot>,
    pub hovered: Option<Hovered>,
}

impl Default for ModelView {
    fn default() -> Self {
        Self { enabled: true, key: None, slots: Vec::new(), hovered: None }
    }
}

struct ViewSlot {
    slot: usize,
    placements: Vec<Placement>,
    meshes: Vec<Option<Handle<Mesh>>>,
    colored_runs: u64,
    /// (trace index, element) → the segment it drives.
    taps: Vec<(usize, usize, SegmentRef)>,
}

/// The cube under the pointer and its connections.
pub struct Hovered {
    pub slot: usize,
    pub layer: usize,
    pub index: usize,
    sources: Vec<(usize, usize)>,
    consumers: Vec<(usize, usize)>,
}

#[derive(Component)]
struct ModelLayer;

/// For each output of a model that drives segments: its trace index and taps.
fn output_taps(activations: &[Activation], instance: &crate::onnx::coupling::ModelInstance) -> Vec<(usize, Vec<SegmentRef>)> {
    let outputs = instance.program.outputs();
    instance
        .spec
        .outputs
        .iter()
        .zip(outputs)
        .filter_map(|(out, port)| {
            let taps = match out {
                ModelOutput::InjectCurrent { taps, .. } | ModelOutput::ScaleConductance { taps, .. } => taps.clone(),
                _ => return None,
            };
            let layer = activations.iter().rposition(|a| a.name == port.name)?;
            Some((layer, taps))
        })
        .collect()
}

fn sync_model_view(
    mut commands: Commands,
    mut view: ResMut<ModelView>,
    models: Res<SceneModels>,
    segments: Query<(&SegmentAddress, &GlobalTransform)>,
    layers: Query<Entity, With<ModelLayer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut material: Local<Option<Handle<StandardMaterial>>>,
) {
    let positions: HashMap<SegmentRef, Vec3> = segments
        .iter()
        .map(|(a, t)| (SegmentRef { neuron: a.neuron, segment: a.segment }, t.translation()))
        .collect();

    // Rebuild when the models, their shapes or the tapped segments change.
    let mut key = String::new();
    let mut wanted = Vec::new();
    if view.enabled {
        for (s, slot) in models.slots.iter().enumerate() {
            let ModelState::Ready(instance) = &slot.state else { continue };
            if slot.activations.is_empty() {
                continue;
            }
            let taps = output_taps(&slot.activations, instance);
            let mut anchors = HashMap::new();
            for (layer, refs) in &taps {
                let points: Option<Vec<Vec3>> = refs.iter().map(|r| positions.get(r).copied()).collect();
                if let Some(points) = points {
                    anchors.insert(*layer, points);
                }
            }
            key.push_str(&format!("{s}:{};", slot.url));
            for a in &slot.activations {
                key.push_str(&format!("{:?}", a.tensor.shape));
            }
            let mut anchored: Vec<_> = anchors.iter().collect();
            anchored.sort_by_key(|(l, _)| **l);
            for (l, points) in anchored {
                key.push_str(&format!("@{l}"));
                for p in points {
                    key.push_str(&format!("({:.0},{:.0},{:.0})", p.x, p.y, p.z));
                }
            }
            wanted.push((s, anchors, taps));
        }
    }
    if view.key.as_deref() == Some(key.as_str()) {
        return;
    }
    for e in &layers {
        commands.entity(e).despawn();
    }
    view.slots.clear();
    view.hovered = None;
    view.key = Some(key);

    let material = material
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                alpha_mode: AlphaMode::Add,
                ..default()
            })
        })
        .clone();
    for (s, anchors, taps) in wanted {
        let activations = &models.slots[s].activations;
        let placements = layout(activations, &anchors);
        let handles = placements
            .iter()
            .map(|p| {
                if p.positions.is_empty() {
                    return None;
                }
                let handle = meshes.add(cubes(p));
                commands.spawn((
                    ModelLayer,
                    Mesh3d(handle.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::IDENTITY,
                    NotShadowCaster,
                    Pickable::IGNORE,
                ));
                Some(handle)
            })
            .collect();
        let taps = taps
            .into_iter()
            .flat_map(|(layer, refs)| refs.into_iter().enumerate().map(move |(i, r)| (layer, i, r)))
            .collect();
        view.slots.push(ViewSlot { slot: s, placements, meshes: handles, colored_runs: u64::MAX, taps });
    }
}

/// One mesh of `p.positions.len()` cubes, eight shared corners each.
fn cubes(p: &Placement) -> Mesh {
    let h = p.cube / 2.0;
    let corners: [Vec3; 8] = std::array::from_fn(|i| {
        Vec3::new(if i & 1 == 0 { -1.0 } else { 1.0 }, if i & 2 == 0 { -1.0 } else { 1.0 }, if i & 4 == 0 { -1.0 } else { 1.0 })
    });
    const FACES: [[u32; 4]; 6] = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
    let mut positions = Vec::with_capacity(p.positions.len() * 8);
    let mut normals = Vec::with_capacity(p.positions.len() * 8);
    let mut indices = Vec::with_capacity(p.positions.len() * 36);
    for (k, center) in p.positions.iter().enumerate() {
        let base = (k * 8) as u32;
        for c in corners {
            positions.push((*center + c * h).to_array());
            normals.push(c.normalize().to_array());
        }
        for [a, b, c, d] in FACES {
            indices.extend([a, b, c, a, c, d].map(|v| base + v));
        }
    }
    let colors = vec![[FLOOR, FLOOR, FLOOR, 1.0]; positions.len()];
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

/// Linear RGB for `v` in a layer whose largest |value| is `scale`.
pub fn glow(v: f32, scale: f32) -> [f32; 3] {
    let t = if scale > 0.0 && v.is_finite() { (v.abs() / scale).min(1.0) } else { 0.0 };
    let hue = if v < 0.0 { COOL } else { WARM };
    hue.map(|c| FLOOR + GAIN * c * t)
}

fn max_abs(data: &[f32]) -> f32 {
    data.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()))
}

fn color_model_view(mut view: ResMut<ModelView>, models: Res<SceneModels>, mut meshes: ResMut<Assets<Mesh>>) {
    for vs in view.slots.iter_mut() {
        let Some(slot) = models.slots.get(vs.slot) else { continue };
        if slot.runs == vs.colored_runs {
            continue;
        }
        vs.colored_runs = slot.runs;
        for (act, handle) in slot.activations.iter().zip(&vs.meshes) {
            let Some(mesh) = handle.as_ref().and_then(|h| meshes.get_mut(h)) else { continue };
            let scale = max_abs(&act.tensor.data);
            let colors: Vec<[f32; 4]> = act
                .tensor
                .data
                .iter()
                .flat_map(|v| {
                    let [r, g, b] = glow(*v, scale);
                    [[r, g, b, 1.0]; 8]
                })
                .collect();
            if let Some(bevy::mesh::VertexAttributeValues::Float32x4(old)) = mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
                if old.len() == colors.len() {
                    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
                }
            }
        }
    }
}

/// The nearest cube along `ray`, as (view slot, layer, element).
pub fn pick(placements: &[&[Placement]], ray: Ray3d) -> Option<(usize, usize, usize)> {
    let dir = *ray.direction;
    let mut best: Option<(f32, (usize, usize, usize))> = None;
    for (s, layers) in placements.iter().enumerate() {
        for (l, p) in layers.iter().enumerate() {
            let r = p.cube * 0.6;
            for (i, pos) in p.positions.iter().enumerate() {
                let to = *pos - ray.origin;
                let t = to.dot(dir);
                if t <= 0.0 || best.is_some_and(|(bt, _)| t >= bt) {
                    continue;
                }
                if (to - dir * t).length_squared() <= r * r {
                    best = Some((t, (s, l, i)));
                }
            }
        }
    }
    best.map(|(_, hit)| hit)
}

fn hover_model_view(
    mut view: ResMut<ModelView>,
    models: Res<SceneModels>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut contexts: EguiContexts,
) {
    let over_egui = contexts.ctx_mut().is_ok_and(|ctx| ctx.is_pointer_over_area() || ctx.wants_pointer_input());
    let cursor = windows.single().ok().and_then(|w| w.cursor_position());
    let hit = match (cursor, cameras.iter().next()) {
        (Some(cursor), Some((camera, transform))) if !over_egui => camera
            .viewport_to_world(transform, cursor)
            .ok()
            .and_then(|ray| pick(&view.slots.iter().map(|v| v.placements.as_slice()).collect::<Vec<_>>(), ray)),
        _ => None,
    };
    let Some((vs, layer, index)) = hit else {
        view.hovered = None;
        return;
    };
    let slot = view.slots[vs].slot;
    if view.hovered.as_ref().is_some_and(|h| (h.slot, h.layer, h.index) == (slot, layer, index)) {
        return;
    }
    let trace = &models.slots[slot].activations;
    let strongest = |mut found: Vec<(usize, usize)>| {
        found.sort_by(|a, b| {
            let v = |(l, i): &(usize, usize)| trace[*l].tensor.data.get(*i).map_or(0.0, |v| v.abs());
            v(b).total_cmp(&v(a))
        });
        found.truncate(MAX_LINES);
        found
    };
    view.hovered = Some(Hovered {
        slot,
        layer,
        index,
        sources: strongest(sources(trace, layer, index)),
        consumers: strongest(consumers(trace, layer, index)),
    });
}

fn draw_model_view(
    view: Res<ModelView>,
    models: Res<SceneModels>,
    segments: Query<(&SegmentAddress, &GlobalTransform)>,
    mut gizmos: Gizmos,
) {
    if view.slots.is_empty() {
        return;
    }
    let positions: HashMap<SegmentRef, Vec3> = segments
        .iter()
        .map(|(a, t)| (SegmentRef { neuron: a.neuron, segment: a.segment }, t.translation()))
        .collect();
    for vs in &view.slots {
        let Some(slot) = models.slots.get(vs.slot) else { continue };
        let at = |l: usize, i: usize| vs.placements.get(l).and_then(|p| p.positions.get(i)).copied();
        let value = |l: usize, i: usize| slot.activations.get(l).and_then(|a| a.tensor.data.get(i)).copied().unwrap_or(0.0);
        let scales: Vec<f32> = slot.activations.iter().map(|a| max_abs(&a.tensor.data)).collect();
        let strength = |l: usize, i: usize| {
            let s = scales.get(l).copied().unwrap_or(0.0);
            if s > 0.0 { (value(l, i).abs() / s).clamp(0.0, 1.0) } else { 0.0 }
        };

        // Outputs hang above the segments they drive.
        for (layer, i, tap) in &vs.taps {
            let (Some(from), Some(to)) = (at(*layer, *i), positions.get(tap)) else { continue };
            let [r, g, b] = WARM;
            gizmos.line(from, *to, LinearRgba::new(r, g, b, 0.08 + 0.9 * strength(*layer, *i)));
        }

        let Some(h) = view.hovered.as_ref().filter(|h| h.slot == vs.slot) else { continue };
        let Some(origin) = at(h.layer, h.index) else { continue };
        let cube = vs.placements[h.layer].cube;
        gizmos.cube(Transform::from_translation(origin).with_scale(Vec3::splat(cube * 1.3)), Color::WHITE);
        for (lines, [r, g, b]) in [(&h.sources, [0.3, 0.9, 1.0]), (&h.consumers, [1.0, 0.65, 0.2])] {
            for (l, i) in lines.iter() {
                if let Some(p) = at(*l, *i) {
                    gizmos.line(origin, p, LinearRgba::new(r, g, b, 0.15 + 0.85 * strength(*l, *i)));
                }
            }
        }
    }
}

/// Name and value of the hovered cube, next to the pointer.
pub fn model_view_tooltip(mut contexts: EguiContexts, view: Res<ModelView>, models: Res<SceneModels>) {
    let Some(h) = view.hovered.as_ref() else { return };
    let Some(act) = models.slots.get(h.slot).and_then(|s| s.activations.get(h.layer)) else { return };
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let Some(pointer) = ctx.pointer_hover_pos().or_else(|| ctx.input(|i| i.pointer.latest_pos())) else { return };
    let value = act.tensor.data.get(h.index).copied().unwrap_or(f32::NAN);
    egui::Area::new(egui::Id::new("model_view_tooltip"))
        .fixed_pos(pointer + egui::vec2(16.0, 16.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(egui::RichText::new(format!("{}{:?} = {value:.4}", act.name, unravel(h.index, &act.tensor.shape))).monospace());
                ui.label(
                    egui::RichText::new(format!(
                        "{} {:?} · reads {} · read by {}",
                        act.op,
                        act.tensor.shape,
                        count(&h.sources),
                        count(&h.consumers)
                    ))
                    .small()
                    .weak(),
                );
            });
        });
}

fn count(lines: &[(usize, usize)]) -> String {
    if lines.len() >= MAX_LINES { format!("{MAX_LINES}+") } else { lines.len().to_string() }
}
