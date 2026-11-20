//! Binding a compiled [`Program`] to a scene: turning simulation state into
//! input tensors and output tensors into injected currents. Plain functions,
//! independent of Bevy, so they can be tested directly.

use std::sync::Arc;

use crate::onnx::{Activation, HostTensor, Port, Program};
use crate::serialize::{ModelInput, ModelOutput, SceneModel, SegmentRef};

/// A model whose bindings have been checked against its graph.
pub struct ModelInstance {
    pub program: Arc<Program>,
    pub spec: SceneModel,
    /// Tensor shape for each input, with dynamic dims resolved.
    input_shapes: Vec<Vec<usize>>,
    /// For each `Feedback` input: the index of the output it reads.
    feedback: Vec<Option<usize>>,
    /// For each `Feedback` input: the values for the next run.
    state: Vec<Vec<f32>>,
}

/// Simulation state a model can read, and the targets it can write.
pub trait SimState {
    /// Membrane voltage of a segment in mV, or `None` if it doesn't exist.
    fn voltage_mv(&self, segment: SegmentRef) -> Option<f32>;
    /// Simulated time in seconds.
    fn time_sec(&self) -> f32;
    /// How many channels a segment's membrane has, or `None` if the segment
    /// doesn't exist.
    fn channel_count(&self, segment: SegmentRef) -> Option<usize>;
    /// Whether the scene has synapse `index`.
    fn has_synapse(&self, index: u32) -> bool;
}

/// What one evaluation asks of the simulation.
#[derive(Debug, Default, PartialEq)]
pub struct Effects {
    /// Currents to inject, in µA/cm², per segment. A segment named by several
    /// outputs gets the sum.
    pub currents: Vec<(SegmentRef, f32)>,
    /// Conductance multipliers per segment and channel (`None`: all of the
    /// segment's channels). Several outputs on one target multiply.
    pub conductance_scales: Vec<(SegmentRef, Option<u32>, f32)>,
    /// Receptor current multipliers per synapse index. Several outputs on
    /// one synapse multiply.
    pub synapse_gains: Vec<(u32, f32)>,
}

/// The outcome of one evaluation.
#[derive(Debug, PartialEq)]
pub enum Step {
    Ran(Effects),
    /// Some input was not a finite number yet (for example GPU voltages
    /// before the first readback); the model was not run.
    NotReady,
}

impl ModelInstance {
    /// Check `spec` against the program's graph.
    pub fn bind(program: Arc<Program>, spec: SceneModel) -> Result<ModelInstance, String> {
        let ports = program.inputs();
        if spec.inputs.len() != ports.len() {
            return Err(format!(
                "the model has {} inputs ({}), but the scene binds {}",
                ports.len(),
                port_names(ports),
                spec.inputs.len()
            ));
        }
        let outputs = program.outputs();
        if spec.outputs.len() != outputs.len() {
            return Err(format!(
                "the model has {} outputs ({}), but the scene binds {}",
                outputs.len(),
                port_names(outputs),
                spec.outputs.len()
            ));
        }
        if !(spec.interval_sec >= 0.0) {
            return Err(format!("interval_sec must be at least 0, got {}", spec.interval_sec));
        }

        let mut input_shapes = Vec::with_capacity(ports.len());
        let mut feedback = Vec::with_capacity(ports.len());
        let mut state = Vec::with_capacity(ports.len());
        for (port, input) in ports.iter().zip(&spec.inputs) {
            let (len, source) = match input {
                ModelInput::Feedback { output } => {
                    let Some(j) = outputs.iter().position(|o| &o.name == output) else {
                        return Err(format!(
                            "input `{}` is fed back from output `{output}`, but the model's outputs are {}",
                            port.name,
                            port_names(outputs)
                        ));
                    };
                    let Some(len) = static_len(port) else {
                        return Err(format!(
                            "input `{}` has shape {}; Feedback needs a fixed shape",
                            port.name,
                            shape_text(&port.shape)
                        ));
                    };
                    (len, Some(j))
                }
                other => (input_len(other), None),
            };
            input_shapes.push(fit_shape(port, len)?);
            state.push(if source.is_some() { vec![0.0; len] } else { Vec::new() });
            feedback.push(source);
        }

        for (port, output) in outputs.iter().zip(&spec.outputs) {
            // Only checkable now when the model fixes the output shape.
            if let (Some((sink, count)), Some(len)) = (targets(output), static_len(port)) {
                if len != count {
                    return Err(format!("output `{}` has {len} values, but {sink} names {count} targets", port.name));
                }
            }
        }

        Ok(ModelInstance { program, spec, input_shapes, feedback, state })
    }

    /// Every segment this model reads or writes.
    pub fn taps(&self) -> impl Iterator<Item = SegmentRef> + '_ {
        let ins = self.spec.inputs.iter().flat_map(|i| match i {
            ModelInput::Voltages { taps, .. } => taps.as_slice(),
            _ => &[],
        });
        let outs = self.spec.outputs.iter().flat_map(|o| match o {
            ModelOutput::InjectCurrent { taps, .. } | ModelOutput::ScaleConductance { taps, .. } => taps.as_slice(),
            ModelOutput::SynapticGain { .. } | ModelOutput::Ignore => &[],
        });
        ins.chain(outs).copied()
    }

    /// Gather inputs from the simulation, run the model, and map its outputs
    /// to currents.
    pub fn step(&mut self, sim: &impl SimState) -> Result<Step, String> {
        self.evaluate(sim, false).map(|(step, _)| step)
    }

    /// Like [`ModelInstance::step`], and also return every layer's values.
    /// The trace is empty when the model didn't run.
    pub fn step_traced(&mut self, sim: &impl SimState) -> Result<(Step, Vec<Activation>), String> {
        self.evaluate(sim, true)
    }

    fn evaluate(&mut self, sim: &impl SimState, trace: bool) -> Result<(Step, Vec<Activation>), String> {
        let mut inputs = Vec::with_capacity(self.spec.inputs.len());
        for ((input, shape), state) in self.spec.inputs.iter().zip(&self.input_shapes).zip(&self.state) {
            let data = match input {
                ModelInput::Voltages { taps, offset_mv, scale } => taps
                    .iter()
                    .map(|tap| {
                        sim.voltage_mv(*tap)
                            .map(|v| (v - offset_mv) * scale)
                            .ok_or_else(|| format!("no segment {} in neuron {}", tap.segment, tap.neuron))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                ModelInput::Time => vec![sim.time_sec()],
                ModelInput::Constant { values } => values.clone(),
                ModelInput::Feedback { .. } => state.clone(),
            };
            if data.iter().any(|v| !v.is_finite()) {
                return Ok((Step::NotReady, Vec::new()));
            }
            inputs.push(HostTensor::new(shape.clone(), data));
        }

        let (outputs, activations) = if trace {
            self.program.run_traced(&inputs)
        } else {
            self.program.run(&inputs).map(|o| (o, Vec::new()))
        }
        .map_err(|e| e.to_string())?;

        let mut next_state = Vec::new();
        for (i, source) in self.feedback.iter().enumerate() {
            let Some(j) = source else { continue };
            let (port, values) = (&self.program.outputs()[*j], &outputs[*j].data);
            if values.len() != self.state[i].len() {
                return Err(format!(
                    "output `{}` has {} values, but feeds back into input `{}`, which takes {}",
                    port.name,
                    values.len(),
                    self.program.inputs()[i].name,
                    self.state[i].len()
                ));
            }
            if let Some(v) = values.iter().find(|v| !v.is_finite()) {
                return Err(format!("output `{}` produced {v}, which would feed back forever", port.name));
            }
            next_state.push((i, values.clone()));
        }

        let mut effects = Effects::default();
        for ((output, port), tensor) in self.spec.outputs.iter().zip(self.program.outputs()).zip(&outputs) {
            let Some((sink, count)) = targets(output) else { continue };
            if tensor.data.len() != count {
                return Err(format!(
                    "output `{}` has {} values, but {sink} names {count} targets",
                    port.name,
                    tensor.data.len()
                ));
            }
            if let Some(v) = tensor.data.iter().find(|v| !v.is_finite()) {
                return Err(format!("output `{}` produced {v}", port.name));
            }
            let missing = |tap: &SegmentRef| format!("{sink}: no segment {} in neuron {}", tap.segment, tap.neuron);
            match output {
                ModelOutput::InjectCurrent { taps, scale_uamps_per_square_cm } => {
                    for (tap, value) in taps.iter().zip(&tensor.data) {
                        sim.voltage_mv(*tap).ok_or_else(|| missing(tap))?;
                        let current = value * scale_uamps_per_square_cm;
                        match effects.currents.iter_mut().find(|(t, _)| t == tap) {
                            Some((_, total)) => *total += current,
                            None => effects.currents.push((*tap, current)),
                        }
                    }
                }
                ModelOutput::ScaleConductance { taps, channel, offset, scale } => {
                    for (tap, value) in taps.iter().zip(&tensor.data) {
                        let channels = sim.channel_count(*tap).ok_or_else(|| missing(tap))?;
                        if let Some(c) = channel.filter(|c| *c as usize >= channels) {
                            return Err(format!(
                                "{sink}: segment {} in neuron {} has {channels} channels, so there is no channel {c}",
                                tap.segment, tap.neuron
                            ));
                        }
                        let factor = multiplier(*value, *offset, *scale);
                        match effects.conductance_scales.iter_mut().find(|(t, c, _)| t == tap && c == channel) {
                            Some((_, _, total)) => *total *= factor,
                            None => effects.conductance_scales.push((*tap, *channel, factor)),
                        }
                    }
                }
                ModelOutput::SynapticGain { synapses, offset, scale } => {
                    for (synapse, value) in synapses.iter().zip(&tensor.data) {
                        if !sim.has_synapse(*synapse) {
                            return Err(format!("{sink}: the scene has no synapse {synapse}"));
                        }
                        let factor = multiplier(*value, *offset, *scale);
                        match effects.synapse_gains.iter_mut().find(|(s, _)| s == synapse) {
                            Some((_, total)) => *total *= factor,
                            None => effects.synapse_gains.push((*synapse, factor)),
                        }
                    }
                }
                ModelOutput::Ignore => {}
            }
        }
        for (i, values) in next_state {
            self.state[i] = values;
        }
        Ok((Step::Ran(effects), activations))
    }
}

/// A multiplier from a model output: `max(0, offset + scale * value)`.
pub fn multiplier(value: f32, offset: f32, scale: f32) -> f32 {
    (offset + scale * value).max(0.0)
}

/// The sink an output drives and how many targets it names.
fn targets(output: &ModelOutput) -> Option<(&'static str, usize)> {
    match output {
        ModelOutput::InjectCurrent { taps, .. } => Some(("InjectCurrent", taps.len())),
        ModelOutput::ScaleConductance { taps, .. } => Some(("ScaleConductance", taps.len())),
        ModelOutput::SynapticGain { synapses, .. } => Some(("SynapticGain", synapses.len())),
        ModelOutput::Ignore => None,
    }
}

fn input_len(input: &ModelInput) -> usize {
    match input {
        ModelInput::Voltages { taps, .. } => taps.len(),
        ModelInput::Time => 1,
        ModelInput::Constant { values } => values.len(),
        ModelInput::Feedback { .. } => unreachable!("Feedback sizes come from the input port"),
    }
}

fn static_len(port: &Port) -> Option<usize> {
    port.shape.iter().copied().product::<Option<usize>>()
}

/// The tensor shape for `len` values fed into `port`: fixed dims as given,
/// and at most one dynamic dim inferred (several are all taken as 1, except
/// that a lone dynamic dim absorbs the remainder).
pub fn fit_shape(port: &Port, len: usize) -> Result<Vec<usize>, String> {
    let fixed: usize = port.shape.iter().map(|d| d.unwrap_or(1)).product();
    let dynamic: Vec<usize> = port.shape.iter().enumerate().filter(|(_, d)| d.is_none()).map(|(i, _)| i).collect();
    let mut shape: Vec<usize> = port.shape.iter().map(|d| d.unwrap_or(1)).collect();
    match dynamic.as_slice() {
        [i] if fixed > 0 && len % fixed == 0 => shape[*i] = len / fixed,
        _ if fixed == len => {}
        _ => {
            return Err(format!(
                "input `{}` has shape {}, which doesn't fit the {len} values bound to it",
                port.name,
                shape_text(&port.shape)
            ));
        }
    }
    Ok(shape)
}

fn shape_text(shape: &[Option<usize>]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.map_or("?".to_string(), |n| n.to_string())).collect();
    format!("[{}]", dims.join(", "))
}

fn port_names(ports: &[Port]) -> String {
    ports.iter().map(|p| format!("`{}` {}", p.name, shape_text(&p.shape))).collect::<Vec<_>>().join(", ")
}
