//! Scene models in the ECS: fetch and compile the `.onnx` files a scene
//! names, then run them between frames and inject their currents.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use burn::tensor::Device;
use crossbeam::channel::{unbounded, Receiver, Sender};

use crate::dimension::{MicroAmpsPerSquareCm, Timestamp};
use crate::neuron::membrane::MembraneVoltage;
use crate::neuron::segment::ecs::{ModelCurrent, SegmentAddress};
use crate::onnx::coupling::{ModelInstance, SimState, Step};
use crate::onnx::{Activation, Program};
use crate::serialize::{SceneModel, SegmentRef};

pub struct OnnxPlugin;

impl Plugin for OnnxPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = unbounded();
        app.insert_resource(ModelFetchSender(tx))
            .insert_resource(ModelFetchReceiver(rx))
            .init_resource::<SceneModels>()
            .add_systems(Update, receive_model_fetches);
        // `run_models` is scheduled by `NbSimPlugin`, before the sim step.
    }
}

type ModelFetch = (String, Result<Vec<u8>, String>);

#[derive(Resource, Clone)]
pub struct ModelFetchSender(pub Sender<ModelFetch>);

#[derive(Resource)]
pub struct ModelFetchReceiver(pub Receiver<ModelFetch>);

/// The current scene's models, and every model compiled so far by URL.
#[derive(Resource, Default)]
pub struct SceneModels {
    cache: HashMap<String, Cached>,
    pub slots: Vec<ModelSlot>,
}

enum Cached {
    Pending,
    Ready(Arc<Program>),
    Failed(String),
}

/// One model of the current scene.
pub struct ModelSlot {
    /// The resolved URL or path.
    pub url: String,
    pub state: ModelState,
    /// Evaluations so far.
    pub runs: u64,
    next_run_sec: f32,
    /// The currents from the latest evaluation; held until the next one.
    currents: Vec<(SegmentRef, f32)>,
    /// Every layer's values from the latest evaluation, for the Models window.
    pub activations: Vec<Activation>,
}

pub enum ModelState {
    Loading(SceneModel),
    Ready(ModelInstance),
    Failed(String),
}

impl SceneModels {
    /// Replace the scene's models. URLs are resolved against `root`; already
    /// compiled models are reused unless `refetch`.
    pub fn load(&mut self, models: &[SceneModel], root: &str, refetch: bool, sender: &ModelFetchSender) {
        if refetch {
            self.cache.retain(|_, c| matches!(c, Cached::Pending));
        }
        self.slots = models
            .iter()
            .map(|spec| {
                let url = nb_nickel::resolve_path(root, &spec.url);
                let state = match self.cache.get(&url) {
                    Some(Cached::Ready(program)) => bind(program.clone(), spec.clone()),
                    Some(Cached::Failed(e)) => ModelState::Failed(e.clone()),
                    Some(Cached::Pending) => ModelState::Loading(spec.clone()),
                    None => {
                        self.cache.insert(url.clone(), Cached::Pending);
                        fetch_bytes(url.clone(), sender.0.clone());
                        ModelState::Loading(spec.clone())
                    }
                };
                ModelSlot { url, state, runs: 0, next_run_sec: 0.0, currents: Vec::new(), activations: Vec::new() }
            })
            .collect();
    }

    fn compiled(&mut self, url: String, result: Result<Arc<Program>, String>) {
        for slot in self.slots.iter_mut().filter(|s| s.url == url) {
            if let ModelState::Loading(spec) = &slot.state {
                slot.state = match &result {
                    Ok(program) => bind(program.clone(), spec.clone()),
                    Err(e) => ModelState::Failed(e.clone()),
                };
            }
        }
        let entry = match result {
            Ok(program) => Cached::Ready(program),
            Err(e) => Cached::Failed(e),
        };
        self.cache.insert(url, entry);
    }
}

fn bind(program: Arc<Program>, spec: SceneModel) -> ModelState {
    match ModelInstance::bind(program, spec) {
        Ok(instance) => ModelState::Ready(instance),
        Err(e) => ModelState::Failed(e),
    }
}

/// Fetch a file's bytes: over HTTP(S) on every target, from disk natively.
pub fn fetch_bytes(path: String, sender: Sender<ModelFetch>) {
    if path.starts_with("http://") || path.starts_with("https://") {
        ehttp::fetch(ehttp::Request::get(&path), move |response| {
            let result = match response {
                Ok(r) if r.ok => Ok(r.bytes),
                Ok(r) => Err(format!("HTTP {} {}", r.status, r.status_text)),
                Err(e) => Err(e),
            };
            let _ = sender.send((path, result));
        });
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = std::fs::read(&path).map_err(|e| e.to_string());
        let _ = sender.send((path, result));
    }
    #[cfg(target_arch = "wasm32")]
    {
        let message = format!("cannot read `{path}` in the browser; use an absolute URL");
        let _ = sender.send((path, Err(message)));
    }
}

fn receive_model_fetches(receiver: Res<ModelFetchReceiver>, mut models: ResMut<SceneModels>) {
    while let Ok((url, result)) = receiver.0.try_recv() {
        let compiled = result
            .map_err(|e| format!("could not load {url}: {e}"))
            .and_then(|bytes| {
                Program::from_bytes(&bytes, Device::default())
                    .map(Arc::new)
                    .map_err(|e| format!("{url}: {e}"))
            });
        match &compiled {
            Ok(program) => info!(
                "Compiled model {url}: {} inputs, {} outputs",
                program.inputs().len(),
                program.outputs().len()
            ),
            Err(e) => warn!("{e}"),
        }
        models.compiled(url, compiled);
    }
}

/// Wall-clock seconds between model status lines in the log.
const LOG_INTERVAL_SECS: f32 = 2.0;

struct Snapshot {
    voltages: HashMap<SegmentRef, f32>,
    time_sec: f32,
}

impl SimState for Snapshot {
    fn voltage_mv(&self, segment: SegmentRef) -> Option<f32> {
        self.voltages.get(&segment).copied()
    }
    fn time_sec(&self) -> f32 {
        self.time_sec
    }
}

/// Run every model that is due, then write each segment's total model current.
pub fn run_models(
    mut models: ResMut<SceneModels>,
    timestamp: Res<Timestamp>,
    voltages: Query<(&SegmentAddress, &MembraneVoltage)>,
    mut currents: Query<(&SegmentAddress, &mut ModelCurrent)>,
    time: Res<Time>,
    mut since_log: Local<f32>,
) {
    if models.slots.is_empty() && !models.is_changed() {
        return;
    }
    let now = timestamp.0;
    *since_log += time.delta_secs();
    let log_now = *since_log >= LOG_INTERVAL_SECS;
    if log_now {
        *since_log = 0.0;
    }
    let due = models.slots.iter().any(|s| matches!(s.state, ModelState::Ready(_)) && now >= s.next_run_sec);
    if due {
        let snapshot = Snapshot {
            voltages: voltages
                .iter()
                .map(|(a, v)| (SegmentRef { neuron: a.neuron, segment: a.segment }, v.0 .0))
                .collect(),
            time_sec: now,
        };
        for slot in models.slots.iter_mut() {
            let ModelState::Ready(instance) = &mut slot.state else { continue };
            if now < slot.next_run_sec {
                continue;
            }
            match instance.step_traced(&snapshot) {
                Ok((Step::Inject(c), activations)) => {
                    slot.currents = c;
                    slot.activations = activations;
                    slot.runs += 1;
                    slot.next_run_sec = now + instance.spec.interval_sec;
                }
                Ok((Step::NotReady, _)) => {}
                Err(e) => {
                    warn!("model {}: {e}", slot.url);
                    slot.currents.clear();
                    slot.activations.clear();
                    slot.state = ModelState::Failed(e);
                }
            }
        }
    }

    if log_now {
        for slot in &models.slots {
            if let ModelState::Ready(_) = slot.state {
                let taps: Vec<String> = slot
                    .currents
                    .iter()
                    .map(|(tap, i)| {
                        let v = voltages
                            .iter()
                            .find(|(a, _)| a.neuron == tap.neuron && a.segment == tap.segment)
                            .map_or(f32::NAN, |(_, v)| v.0 .0);
                        format!("n{}s{}: {v:.1} mV, {i:+.1} µA/cm²", tap.neuron, tap.segment)
                    })
                    .collect();
                info!("model {} ({} runs): {}", slot.url, slot.runs, taps.join("; "));
            }
        }
    }

    let mut totals: HashMap<SegmentRef, f32> = HashMap::new();
    for slot in &models.slots {
        for (tap, current) in &slot.currents {
            *totals.entry(*tap).or_default() += current;
        }
    }
    for (address, mut current) in &mut currents {
        let tap = SegmentRef { neuron: address.neuron, segment: address.segment };
        let total = totals.get(&tap).copied().unwrap_or(0.0);
        if current.0 .0 != total {
            current.0 = MicroAmpsPerSquareCm(total);
        }
    }
}
