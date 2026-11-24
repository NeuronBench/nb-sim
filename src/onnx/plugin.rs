//! Scene models in the ECS: fetch and compile the `.onnx` files a scene
//! names, then run them between frames and inject their currents.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use burn::tensor::Device;
use crossbeam::channel::{unbounded, Receiver, Sender};

use crate::dimension::{MicroAmpsPerSquareCm, Timestamp};
use crate::gpu::buffers::MAX_CHANNELS;
use crate::integrations::grace::{ModelSynapseGain, SynapseAddress};
use crate::neuron::membrane::{Membrane, MembraneVoltage};
use crate::neuron::segment::ecs::{ModelConductanceScale, ModelCurrent, SegmentAddress};
use crate::onnx::coupling::{Effects, ModelInstance, SimState, Step};
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
    /// Scenes loaded whose neurons have not been spawned yet. Models wait
    /// for them: a model bound to a new scene would otherwise run against
    /// the old scene's despawned segments and fail on missing taps.
    unspawned: u32,
    /// Skip one more run after a spawn, until its commands are applied.
    settling: bool,
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
    /// What the latest evaluation asked for; held until the next one.
    pub effects: Effects,
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
        self.unspawned += 1;
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
                ModelSlot { url, state, runs: 0, next_run_sec: 0.0, effects: Effects::default(), activations: Vec::new() }
            })
            .collect();
    }

    /// A loaded scene's neurons were spawned (their commands apply at the
    /// end of this frame).
    pub fn scene_spawned(&mut self) {
        self.unspawned = self.unspawned.saturating_sub(1);
        self.settling = true;
    }

    /// Whether models must wait for their scene's neurons to exist.
    fn waiting_for_scene(&mut self) -> bool {
        self.unspawned > 0 || std::mem::take(&mut self.settling)
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
    /// Voltage and channel count per segment.
    segments: HashMap<SegmentRef, (f32, usize)>,
    synapses: std::collections::HashSet<u32>,
    time_sec: f32,
}

impl SimState for Snapshot {
    fn voltage_mv(&self, segment: SegmentRef) -> Option<f32> {
        self.segments.get(&segment).map(|s| s.0)
    }
    fn time_sec(&self) -> f32 {
        self.time_sec
    }
    fn channel_count(&self, segment: SegmentRef) -> Option<usize> {
        self.segments.get(&segment).map(|s| s.1)
    }
    fn has_synapse(&self, index: u32) -> bool {
        self.synapses.contains(&index)
    }
}

/// What models write into the simulation, per segment and per synapse.
type SegmentSinks<'a> = (&'a SegmentAddress, &'a mut ModelCurrent, &'a mut ModelConductanceScale);

/// Run every model that is due, then write what all models ask of each
/// segment and synapse: summed currents, multiplied conductance scales and
/// synapse gains.
pub fn run_models(
    mut models: ResMut<SceneModels>,
    timestamp: Res<Timestamp>,
    segments: Query<(&SegmentAddress, &MembraneVoltage, &Membrane)>,
    synapse_addresses: Query<&SynapseAddress>,
    mut segment_sinks: Query<SegmentSinks>,
    mut synapse_sinks: Query<(&SynapseAddress, &mut ModelSynapseGain)>,
    time: Res<Time>,
    mut since_log: Local<f32>,
    canvases: Option<Res<crate::canvas::SceneCanvases>>,
) {
    let painting = canvases.as_ref().is_some_and(|c| !c.canvases.is_empty() || c.is_changed());
    if models.slots.is_empty() && !models.is_changed() && !painting {
        return;
    }
    let now = timestamp.0;
    *since_log += time.delta_secs();
    let log_now = *since_log >= LOG_INTERVAL_SECS;
    if log_now {
        *since_log = 0.0;
    }
    let waiting = models.waiting_for_scene();
    let due = !waiting && models.slots.iter().any(|s| matches!(s.state, ModelState::Ready(_)) && now >= s.next_run_sec);
    if due {
        let snapshot = Snapshot {
            segments: segments
                .iter()
                .map(|(a, v, m)| {
                    let channels = m.membrane_channels.len().min(MAX_CHANNELS);
                    (SegmentRef { neuron: a.neuron, segment: a.segment }, (v.0 .0, channels))
                })
                .collect(),
            synapses: synapse_addresses.iter().map(|a| a.0).collect(),
            time_sec: now,
        };
        for slot in models.slots.iter_mut() {
            let ModelState::Ready(instance) = &mut slot.state else { continue };
            if now < slot.next_run_sec {
                continue;
            }
            match instance.step_traced(&snapshot) {
                Ok((Step::Ran(effects), activations)) => {
                    slot.effects = effects;
                    slot.activations = activations;
                    slot.runs += 1;
                    slot.next_run_sec = now + instance.spec.interval_sec;
                }
                Ok((Step::NotReady, _)) => {}
                Err(e) => {
                    warn!("model {}: {e}", slot.url);
                    slot.effects = Effects::default();
                    slot.activations.clear();
                    slot.state = ModelState::Failed(e);
                }
            }
        }
    }

    if log_now {
        for slot in &models.slots {
            if let ModelState::Ready(_) = slot.state {
                info!("model {} ({} runs): {}", slot.url, slot.runs, describe(&slot.effects, &segments));
            }
        }
    }

    let mut currents: HashMap<SegmentRef, f32> = HashMap::new();
    for (tap, current) in canvases.iter().flat_map(|c| c.currents()) {
        *currents.entry(tap).or_default() += current;
    }
    let mut scales: HashMap<SegmentRef, [f32; MAX_CHANNELS]> = HashMap::new();
    let mut gains: HashMap<u32, f32> = HashMap::new();
    for effects in models.slots.iter().map(|s| &s.effects) {
        for (tap, current) in &effects.currents {
            *currents.entry(*tap).or_default() += current;
        }
        for (tap, channel, factor) in &effects.conductance_scales {
            let scale = scales.entry(*tap).or_insert([1.0; MAX_CHANNELS]);
            match channel {
                Some(c) => scale[*c as usize] *= factor,
                None => scale.iter_mut().for_each(|s| *s *= factor),
            }
        }
        for (synapse, factor) in &effects.synapse_gains {
            *gains.entry(*synapse).or_insert(1.0) *= factor;
        }
    }
    for (address, mut current, mut scale) in &mut segment_sinks {
        let tap = SegmentRef { neuron: address.neuron, segment: address.segment };
        let total = currents.get(&tap).copied().unwrap_or(0.0);
        if current.0 .0 != total {
            current.0 = MicroAmpsPerSquareCm(total);
        }
        let factors = scales.get(&tap).copied().unwrap_or([1.0; MAX_CHANNELS]);
        if scale.0 != factors {
            scale.0 = factors;
        }
    }
    for (address, mut gain) in &mut synapse_sinks {
        let total = gains.get(&address.0).copied().unwrap_or(1.0);
        if gain.0 != total {
            gain.0 = total;
        }
    }
}

/// One log line's worth of a model's effects, with each tapped segment's voltage.
fn describe(effects: &Effects, segments: &Query<(&SegmentAddress, &MembraneVoltage, &Membrane)>) -> String {
    let voltage = |tap: &SegmentRef| {
        segments
            .iter()
            .find(|(a, ..)| a.neuron == tap.neuron && a.segment == tap.segment)
            .map_or(f32::NAN, |(_, v, _)| v.0 .0)
    };
    let mut parts: Vec<String> = Vec::new();
    for (tap, i) in &effects.currents {
        parts.push(format!("n{}s{}: {:.1} mV, {i:+.1} µA/cm²", tap.neuron, tap.segment, voltage(tap)));
    }
    for (tap, channel, factor) in &effects.conductance_scales {
        let which = channel.map_or("all channels".to_string(), |c| format!("channel {c}"));
        parts.push(format!("n{}s{} {which}: ×{factor:.3}", tap.neuron, tap.segment));
    }
    for (synapse, factor) in &effects.synapse_gains {
        parts.push(format!("synapse {synapse}: ×{factor:.3}"));
    }
    parts.join("; ")
}
