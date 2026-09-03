//! Scene loading.
//!
//! A scene is a root `.ncl` file plus whatever it imports. Sources are
//! fetched asynchronously (over HTTP in the browser, from disk natively),
//! linked by `nb-nickel`, evaluated in-process, decoded into a
//! [`serialize::Scene`](crate::serialize::Scene), and spawned. Sliders in the
//! UI re-evaluate the cached sources with overrides.

use std::collections::{HashMap, HashSet};

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use crossbeam::channel::{unbounded, Receiver, Sender};
use nb_nickel::{plan, read_params, Diagnostic, Override, ParamSpec, Plan};

use crate::integrations::grace::{GraceScene, GraceSceneReceiver, GraceSceneSender};
use crate::integrations::nickel::{normalize_root, scene_from_linked};
use crate::neuron::ecs::Neuron;
use crate::neuron::membrane::MembraneMaterials;
use crate::neuron::segment::ecs::Segment;
use crate::neuron::Junction;
use crate::selection::{Highlight, Selection};
use crate::stimulator::Stimulation;

/// True while sources are being fetched or a load is pending.
#[derive(Resource)]
pub struct IsLoading(pub bool);

/// The root scene file: an absolute URL, or a filesystem path on native builds.
#[derive(Resource)]
pub struct SceneSource(pub String);

impl FromWorld for SceneSource {
    #[cfg(target_arch = "wasm32")]
    fn from_world(_world: &mut World) -> Self {
        SceneSource(window_location_scene())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn from_world(_world: &mut World) -> Self {
        if let Ok(path) = std::env::var("NB_SCENE") {
            return SceneSource(path);
        }
        // nb-lib is its own repository; look for a sibling checkout.
        let candidates = [
            concat!(env!("CARGO_MANIFEST_DIR"), "/../nb-lib/scene.ncl"),
            "../nb-lib/scene.ncl",
            "nb-lib/scene.ncl",
        ];
        match candidates.iter().find(|p| std::path::Path::new(p).exists()) {
            Some(p) => SceneSource(p.to_string()),
            None => SceneSource(String::new()),
        }
    }
}

/// Read the `?scene=` query parameter. A value that is not an absolute URL
/// is resolved against the page origin.
#[cfg(target_arch = "wasm32")]
pub fn window_location_scene() -> String {
    let location = web_sys::window().expect("should have window").location();
    let Ok(search) = location.search() else { return String::new() };
    if search.len() <= 1 {
        return String::new();
    }
    let params = querystring::querify(&search[1..]);
    let Some((_, value)) = params.iter().find(|(k, _)| *k == "scene") else { return String::new() };
    let value = value.to_string();
    if value.starts_with("http://") || value.starts_with("https://") || value.is_empty() {
        value
    } else {
        let origin = location.origin().unwrap_or_default();
        if value.starts_with('/') {
            format!("{origin}{value}")
        } else {
            format!("{origin}/{value}")
        }
    }
}

pub type FetchResult = (String, Result<String, String>);

#[derive(Resource, Clone)]
pub struct FetchSender(pub Sender<FetchResult>);

#[derive(Resource)]
pub struct FetchReceiver(pub Receiver<FetchResult>);

/// Fetched sources and the state of the current load.
#[derive(Resource, Default)]
pub struct SceneLoader {
    /// Resolved path -> file text.
    pub sources: HashMap<String, String>,
    /// Paths with a fetch in flight.
    pub pending: HashSet<String>,
    /// A (re)load has been requested.
    pub requested: bool,
    /// A fetch failed since the last request; stop trying until the next request.
    pub failed: bool,
}

impl SceneLoader {
    /// Request a load. With `refetch`, cached sources are discarded first.
    pub fn request(&mut self, refetch: bool) {
        if refetch {
            self.sources.clear();
        }
        self.requested = true;
        self.failed = false;
    }
}

/// Slider parameters discovered under the scene's `params` record.
#[derive(Resource, Default)]
pub struct SceneParams {
    pub specs: Vec<ParamSpec>,
    /// Current value per parameter name.
    pub values: HashMap<String, f64>,
    /// A value changed and a re-evaluation is due after the debounce interval.
    pub dirty: bool,
    /// Time (seconds since startup) of the last change.
    pub last_change: f64,
}

impl SceneParams {
    /// Overrides for every parameter whose value differs from its default.
    pub fn overrides(&self) -> Vec<Override> {
        self.specs
            .iter()
            .filter_map(|spec| {
                let value = *self.values.get(&spec.name)?;
                (value != spec.default).then(|| Override::number(&spec.path, value))
            })
            .collect()
    }

    pub fn set(&mut self, name: &str, value: f64, now: f64) {
        self.values.insert(name.to_string(), value);
        self.dirty = true;
        self.last_change = now;
    }

    pub fn reset(&mut self, now: f64) {
        for spec in &self.specs {
            self.values.insert(spec.name.clone(), spec.default);
        }
        self.dirty = true;
        self.last_change = now;
    }
}

/// Errors from the last load, mapped to the user's files.
#[derive(Resource, Default)]
pub struct SceneDiagnostics(pub Vec<Diagnostic>);

/// Seconds to wait after the last slider change before re-evaluating.
const DEBOUNCE_SECS: f64 = 0.15;

pub fn setup(app: &mut App) {
    app.insert_resource(IsLoading(false));
    app.init_resource::<SceneSource>();
    app.init_resource::<SceneLoader>();
    app.init_resource::<SceneParams>();
    app.init_resource::<SceneDiagnostics>();
    let (tx, rx) = unbounded();
    app.insert_resource(GraceSceneSender(tx));
    app.insert_resource(GraceSceneReceiver(rx));
    let (ftx, frx) = unbounded();
    app.insert_resource(FetchSender(ftx));
    app.insert_resource(FetchReceiver(frx));
    app.add_systems(Startup, request_initial_load);
    app.add_systems(Update, drive_scene_loader);
}

fn request_initial_load(source: Res<SceneSource>, mut loader: ResMut<SceneLoader>) {
    if !source.0.is_empty() {
        loader.request(true);
    }
}

/// Fetch one source. HTTP(S) URLs work on every target; plain paths are
/// read from disk natively and rejected in the browser.
pub fn fetch_source(path: String, sender: Sender<FetchResult>) {
    if path.starts_with("http://") || path.starts_with("https://") {
        let request = ehttp::Request::get(&path);
        ehttp::fetch(request, move |response| {
            let result = match response {
                Ok(r) if r.ok => r
                    .text()
                    .map(|s| s.to_string())
                    .ok_or_else(|| "response was not text".to_string()),
                Ok(r) => Err(format!("HTTP {} {}", r.status, r.status_text)),
                Err(e) => Err(e),
            };
            let _ = sender.send((path, result));
        });
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = std::fs::read_to_string(&path).map_err(|e| e.to_string());
        let _ = sender.send((path, result));
    }
    #[cfg(target_arch = "wasm32")]
    {
        let message = format!("cannot read `{path}` in the browser; use an absolute URL");
        let _ = sender.send((path, Err(message)));
    }
}

/// Everything that gets despawned when a new scene replaces the old one.
#[derive(SystemParam)]
pub struct SceneEntities<'w, 's> {
    neurons: Query<'w, 's, Entity, With<Neuron>>,
    segments: Query<'w, 's, Entity, With<Segment>>,
    junctions: Query<'w, 's, Entity, With<Junction>>,
    stimulations: Query<'w, 's, Entity, With<Stimulation>>,
}

impl SceneEntities<'_, '_> {
    fn despawn_all(&self, commands: &mut Commands) {
        for e in self.stimulations.iter() {
            commands.entity(e).despawn();
        }
        for e in self.junctions.iter() {
            commands.entity(e).despawn();
        }
        for e in self.segments.iter() {
            commands.entity(e).despawn();
        }
        for e in self.neurons.iter() {
            commands.entity(e).despawn();
        }
    }
}

/// Collect fetched sources, fire fetches for missing imports, and once every
/// import is present link, evaluate, and spawn.
pub fn drive_scene_loader(
    mut commands: Commands,
    mut loader: ResMut<SceneLoader>,
    source: Res<SceneSource>,
    receiver: Res<FetchReceiver>,
    sender: Res<FetchSender>,
    mut params: ResMut<SceneParams>,
    mut diagnostics: ResMut<SceneDiagnostics>,
    mut is_loading: ResMut<IsLoading>,
    time: Res<Time>,
    entities: SceneEntities,
    scene_sender: Res<GraceSceneSender>,
    mut last_debug_reload: Local<f64>,
) {
    while let Ok((path, result)) = receiver.0.try_recv() {
        loader.pending.remove(&path);
        match result {
            Ok(text) => {
                loader.sources.insert(path, text);
            }
            Err(e) => {
                diagnostics.0.push(Diagnostic::plain(format!("could not load {path}: {e}")));
                loader.failed = true;
            }
        }
    }

    let now = time.elapsed_secs_f64();
    if params.dirty && now - params.last_change > DEBOUNCE_SECS {
        params.dirty = false;
        loader.request(false);
    }
    // Diagnostic knob: NB_DEBUG_RELOAD_SECS=n re-evaluates and respawns the
    // scene every n seconds, which exercises the same path as a slider change.
    if let Some(every) = std::env::var("NB_DEBUG_RELOAD_SECS").ok().and_then(|s| s.parse::<f64>().ok()) {
        if now - *last_debug_reload > every && !loader.requested {
            *last_debug_reload = now;
            info!("NB_DEBUG_RELOAD_SECS: respawning the scene");
            loader.request(false);
        }
    }

    is_loading.0 = loader.requested;
    if !loader.requested || !loader.pending.is_empty() {
        return;
    }
    if loader.failed || source.0.is_empty() {
        loader.requested = false;
        is_loading.0 = false;
        return;
    }

    let root = normalize_root(&source.0);
    match plan(&root, &loader.sources) {
        Ok(Plan::NeedSources(missing)) => {
            diagnostics.0.clear();
            for path in missing {
                loader.pending.insert(path.clone());
                fetch_source(path, sender.0.clone());
            }
        }
        Ok(Plan::Ready(linked)) => {
            loader.requested = false;
            is_loading.0 = false;
            match read_params(&linked) {
                Ok(specs) => {
                    params.values.retain(|name, _| specs.iter().any(|s| &s.name == name));
                    for spec in &specs {
                        params.values.entry(spec.name.clone()).or_insert(spec.default);
                    }
                    params.specs = specs;
                }
                Err(d) => {
                    diagnostics.0 = d;
                    return;
                }
            }
            match scene_from_linked(&linked, &params.overrides()) {
                Ok(scene) => {
                    info!(
                        "Loaded scene from {}: {} neurons, {} synapses, {} parameters",
                        source.0,
                        scene.neurons.len(),
                        scene.synapses.len(),
                        params.specs.len()
                    );
                    diagnostics.0.clear();
                    entities.despawn_all(&mut commands);
                    let _ = scene_sender.0.send(GraceScene(scene));
                }
                Err(d) => {
                    for diag in &d {
                        warn!("{}", diag.render());
                    }
                    diagnostics.0 = d;
                }
            }
        }
        Err(e) => {
            loader.requested = false;
            is_loading.0 = false;
            diagnostics.0 = vec![Diagnostic::plain(e.to_string())];
        }
    }
}

/// Spawn scenes as they arrive on the channel.
pub fn handle_loaded_neuron(
    commands: Commands,
    grace_scene_receiver: Res<GraceSceneReceiver>,
    mut meshes: ResMut<Assets<Mesh>>,
    membrane_materials: Res<MembraneMaterials>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    selections: Query<Entity, With<Selection>>,
    highlights: Query<Entity, With<Highlight>>,
) {
    if let Ok(scene) = grace_scene_receiver.0.try_recv() {
        scene.spawn(
            Vec3::new(0.0, 0.0, 0.0),
            commands,
            &mut meshes,
            membrane_materials,
            &mut materials,
            selections,
            highlights,
        );
    }
}
