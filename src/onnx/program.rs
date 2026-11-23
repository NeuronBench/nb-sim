//! Runtime ONNX evaluation: `onnx-ir` parses the model, [`Program::compile`]
//! turns its typed nodes into a flat list of steps, and [`Program::run`]
//! evaluates them with burn tensor ops.
//!
//! burn tensors carry their rank in the type (`Tensor<D>`), but an
//! interpreter only learns ranks at runtime. Every value is therefore stored
//! as a `Tensor<4>` with leading 1s plus its logical rank. Left-padding with
//! ones is exactly ONNX (numpy) broadcasting, and batched matmul over the two
//! leading dims matches ONNX MatMul, so most ops need no rank bookkeeping at
//! all. Models with tensors above rank 4 are rejected at compile time.

use std::collections::HashMap;
use std::fmt;

use burn::tensor::ops::{ConvOptions, MaxPoolOptions};
use burn::tensor::{activation, module, DType, Device, Tensor, TensorData};
use onnx_ir::{Argument, Node, OnnxGraph, OnnxGraphBuilder};

/// Highest tensor rank a program may use.
pub const MAX_RANK: usize = 4;

/// A host-side tensor: row-major `f32` data and its shape. Inputs and
/// outputs of [`Program::run`] use this, so callers never touch burn types.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }
}

/// One value computed while running a program, kept for visualization.
#[derive(Clone, Debug, PartialEq)]
pub struct Activation {
    /// The ONNX value name (a graph input, or a node's output).
    pub name: String,
    /// The op that produced it, or `"Input"` for a graph input.
    pub op: &'static str,
    pub tensor: HostTensor,
}

/// A graph input or output: its name, and its shape where the model fixes it
/// (`None` for a dynamic dimension such as a batch axis).
#[derive(Clone, Debug, PartialEq)]
pub struct Port {
    pub name: String,
    pub shape: Vec<Option<usize>>,
}

/// Why a model could not be compiled.
#[derive(Clone, Debug, PartialEq)]
pub struct CompileError {
    /// The ONNX node at fault, when there is one.
    pub node: Option<String>,
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.node {
            Some(node) => write!(f, "node `{node}`: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for CompileError {}

/// Why a compiled program could not run on the given inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct RunError(pub String);

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RunError {}

/// A compiled ONNX model, ready to run.
pub struct Program {
    device: Device,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    /// Slot of each graph input, in `inputs` order.
    input_slots: Vec<Slot>,
    /// Slot of each graph output, in `outputs` order.
    output_slots: Vec<Slot>,
    /// Constants (weights), materialized once at compile time.
    constants: Vec<(Slot, Value)>,
    steps: Vec<Step>,
    slot_count: usize,
}

type Slot = usize;

#[derive(Clone)]
struct Value {
    tensor: Tensor<MAX_RANK>,
    rank: usize,
}

impl Value {
    /// The logical (unpadded) shape.
    fn shape(&self) -> Vec<usize> {
        self.tensor.dims()[MAX_RANK - self.rank..].to_vec()
    }

    fn with(&self, tensor: Tensor<MAX_RANK>) -> Value {
        Value { tensor, rank: self.rank }
    }
}

struct Step {
    node: String,
    op: Op,
    inputs: Vec<Slot>,
    /// One entry per value the op produces, in ONNX output order.
    outputs: Vec<StepOutput>,
}

struct StepOutput {
    /// `None` when the model doesn't use this output.
    slot: Option<Slot>,
    /// Logical rank.
    rank: usize,
    /// What to call it in a trace: the graph output's name when it is one,
    /// else the node name (onnx-ir renames intermediate values).
    name: String,
}

enum Op {
    Identity,
    Add,
    Sub,
    Mul,
    Div,
    Relu,
    Tanh,
    Sigmoid,
    LeakyRelu { alpha: f64 },
    Clip { min: f32, max: f32 },
    /// `axis` is in the padded rank-4 frame.
    Softmax { axis: usize },
    /// `axis` is in the logical frame of the input.
    Flatten { axis: usize },
    /// ONNX reshape spec: 0 copies the input dim, -1 is inferred.
    Reshape { shape: Vec<i64> },
    /// `axis` is in the padded rank-4 frame.
    Concat { axis: usize },
    MatMul,
    Gemm { alpha: f32, beta: f32, trans_a: bool, trans_b: bool },
    Linear { transpose_weight: bool },
    Conv1d { kernel: usize, stride: usize, dilation: usize, groups: usize, padding: ConvPadding },
    /// Input `[N, C, H, W]`; `padding` is per spatial axis.
    Conv2d { kernel: [usize; 2], stride: [usize; 2], dilation: [usize; 2], groups: usize, padding: [ConvPadding; 2] },
    MaxPool2d { kernel: [usize; 2], stride: [usize; 2], dilation: [usize; 2], ceil_mode: bool, padding: [ConvPadding; 2] },
    Gru(Recurrent),
    Lstm(Recurrent),
}

#[derive(Clone, Copy, Debug)]
enum ConvPadding {
    Explicit(usize, usize),
    /// ONNX auto_pad SAME_UPPER (extra padding at the end) or SAME_LOWER.
    Same { upper: bool },
}

/// GRU and LSTM settings. Inputs are always laid out as X, W, R, then the
/// optional ones that are present, in ONNX order.
#[derive(Clone, Debug)]
struct Recurrent {
    hidden: usize,
    directions: Direction,
    has_bias: bool,
    has_initial_h: bool,
    /// LSTM only.
    has_initial_c: bool,
    clip: Option<f32>,
    /// GRU only.
    linear_before_reset: bool,
    /// LSTM only: the forget gate is 1 - input gate.
    input_forget: bool,
    gate: Act,
    /// GRU's hidden activation; LSTM's cell activation.
    cell: Act,
    /// LSTM's hidden (output) activation.
    hidden_act: Act,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Direction {
    Forward,
    Reverse,
    Bidirectional,
}

#[derive(Clone, Copy, Debug)]
enum Act {
    Sigmoid,
    Tanh,
    Relu,
}

impl Act {
    fn apply(self, t: Tensor<2>) -> Tensor<2> {
        match self {
            Act::Sigmoid => activation::sigmoid(t),
            Act::Tanh => activation::tanh(t),
            Act::Relu => activation::relu(t),
        }
    }
}

impl Op {
    fn name(&self) -> &'static str {
        match self {
            Op::Identity => "Identity",
            Op::Add => "Add",
            Op::Sub => "Sub",
            Op::Mul => "Mul",
            Op::Div => "Div",
            Op::Relu => "Relu",
            Op::Tanh => "Tanh",
            Op::Sigmoid => "Sigmoid",
            Op::LeakyRelu { .. } => "LeakyRelu",
            Op::Clip { .. } => "Clip",
            Op::Softmax { .. } => "Softmax",
            Op::Flatten { .. } => "Flatten",
            Op::Reshape { .. } => "Reshape",
            Op::Concat { .. } => "Concat",
            Op::MatMul => "MatMul",
            Op::Gemm { .. } => "Gemm",
            Op::Linear { .. } => "Linear",
            Op::Conv1d { .. } => "Conv1d",
            Op::Conv2d { .. } => "Conv2d",
            Op::MaxPool2d { .. } => "MaxPool2d",
            Op::Gru(_) => "GRU",
            Op::Lstm(_) => "LSTM",
        }
    }
}

impl Program {
    /// Parse and compile an ONNX model from its bytes.
    pub fn from_bytes(bytes: &[u8], device: Device) -> Result<Program, CompileError> {
        let graph = OnnxGraphBuilder::new().parse_bytes(bytes).map_err(|e| CompileError {
            node: None,
            message: format!("could not parse the ONNX model: {e}"),
        })?;
        let mut program = Program::compile(graph, device)?;
        program.restore_output_names(bytes);
        Ok(program)
    }

    /// onnx-ir renames graph outputs after the node producing them
    /// (`gemm2_out1`). Put back the names the model file gives them.
    fn restore_output_names(&mut self, bytes: &[u8]) {
        use onnx_ir::{Message, ModelProto};
        let Ok(model) = ModelProto::parse_from_bytes(bytes) else { return };
        let names: Vec<String> = model.graph.output.iter().map(|o| o.name.clone()).collect();
        if names.len() != self.outputs.len() || names.iter().any(String::is_empty) {
            return;
        }
        for ((port, slot), name) in self.outputs.iter_mut().zip(&self.output_slots).zip(names) {
            for out in self.steps.iter_mut().flat_map(|s| &mut s.outputs).filter(|o| o.slot == Some(*slot)) {
                out.name = name.clone();
            }
            port.name = name;
        }
    }

    /// Compile a parsed graph, rejecting anything this interpreter can't run.
    pub fn compile(graph: OnnxGraph, device: Device) -> Result<Program, CompileError> {
        let mut compiler = Compiler { device: device.clone(), slots: HashMap::new(), constants: Vec::new(), slot_count: 0 };

        let mut inputs = Vec::new();
        let mut input_slots = Vec::new();
        for arg in &graph.inputs {
            check_float_tensor(arg, None)?;
            inputs.push(port(arg));
            input_slots.push(compiler.define(&arg.name));
        }

        let mut steps = Vec::new();
        for node in &graph.nodes {
            if let Some(step) = compiler.step(node)? {
                steps.push(step);
            }
        }

        let mut outputs = Vec::new();
        let mut output_slots = Vec::new();
        for arg in &graph.outputs {
            check_float_tensor(arg, None)?;
            outputs.push(port(arg));
            output_slots.push(compiler.input(arg, "graph output")?);
        }
        for out in steps.iter_mut().flat_map(|s: &mut Step| &mut s.outputs) {
            if let Some(i) = output_slots.iter().position(|s| Some(*s) == out.slot) {
                out.name = outputs[i].name.clone();
            }
        }

        Ok(Program {
            device,
            inputs,
            outputs,
            input_slots,
            output_slots,
            constants: compiler.constants,
            steps,
            slot_count: compiler.slot_count,
        })
    }

    pub fn inputs(&self) -> &[Port] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Port] {
        &self.outputs
    }

    /// Evaluate the program. `inputs` are in [`Program::inputs`] order; the
    /// result is in [`Program::outputs`] order.
    pub fn run(&self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, RunError> {
        self.evaluate(inputs, None)
    }

    /// Like [`Program::run`], and also return every graph input and every
    /// intermediate value, in evaluation order.
    pub fn run_traced(&self, inputs: &[HostTensor]) -> Result<(Vec<HostTensor>, Vec<Activation>), RunError> {
        let mut trace = Vec::with_capacity(inputs.len() + self.steps.len());
        let outputs = self.evaluate(inputs, Some(&mut trace))?;
        Ok((outputs, trace))
    }

    fn evaluate(&self, inputs: &[HostTensor], mut trace: Option<&mut Vec<Activation>>) -> Result<Vec<HostTensor>, RunError> {
        if inputs.len() != self.inputs.len() {
            return Err(RunError(format!(
                "the model takes {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }

        let mut values: Vec<Option<Value>> = vec![None; self.slot_count];
        for (slot, value) in &self.constants {
            values[*slot] = Some(value.clone());
        }
        for ((port, host), slot) in self.inputs.iter().zip(inputs).zip(&self.input_slots) {
            check_input_shape(port, host)?;
            values[*slot] = Some(to_value(host, &self.device));
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(Activation { name: port.name.clone(), op: "Input", tensor: host.clone() });
            }
        }

        for step in &self.steps {
            let args: Vec<&Value> = step
                .inputs
                .iter()
                .map(|slot| values[*slot].as_ref().expect("compile order guarantees inputs are set"))
                .collect();
            let results = eval(step, &args).map_err(|e| RunError(format!("node `{}`: {e}", step.node)))?;
            for (out, value) in step.outputs.iter().zip(results) {
                let Some(slot) = out.slot else { continue };
                if let Some(trace) = trace.as_deref_mut() {
                    trace.push(Activation { name: out.name.clone(), op: step.op.name(), tensor: to_host(&value)? });
                }
                values[slot] = Some(value);
            }
        }

        self.output_slots
            .iter()
            .map(|slot| to_host(values[*slot].as_ref().expect("outputs are set")))
            .collect()
    }
}

struct Compiler {
    device: Device,
    slots: HashMap<String, Slot>,
    constants: Vec<(Slot, Value)>,
    slot_count: usize,
}

impl Compiler {
    fn define(&mut self, name: &str) -> Slot {
        let slot = self.slot_count;
        self.slot_count += 1;
        self.slots.insert(name.to_string(), slot);
        slot
    }

    /// The slot holding an argument: a value computed earlier, or a constant
    /// materialized on first use.
    fn input(&mut self, arg: &Argument, node: &str) -> Result<Slot, CompileError> {
        if let Some(slot) = self.slots.get(&arg.name) {
            return Ok(*slot);
        }
        let Some(data) = arg.value() else {
            return Err(err(node, format!("input `{}` is neither computed nor constant", arg.name)));
        };
        let shape = data.shape().to_vec();
        if shape.len() > MAX_RANK {
            return Err(err(node, format!("constant `{}` has rank {}; at most {MAX_RANK} is supported", arg.name, shape.len())));
        }
        let floats: Vec<f32> = data
            .convert::<f32>()
            .try_to_vec()
            .map_err(|e| err(node, format!("constant `{}` is not numeric: {e:?}", arg.name)))?;
        let value = to_value(&HostTensor::new(shape, floats), &self.device);
        // Anonymous constants (embedded, empty name) get a fresh slot each use.
        let slot = if arg.name.is_empty() { self.fresh() } else { self.define(&arg.name) };
        self.constants.push((slot, value));
        Ok(slot)
    }

    fn fresh(&mut self) -> Slot {
        let slot = self.slot_count;
        self.slot_count += 1;
        slot
    }

    fn step(&mut self, node: &Node) -> Result<Option<Step>, CompileError> {
        let name = node.name().to_string();
        let ins = node.inputs();
        let outs = node.outputs();

        // Initializers become Constant nodes; consumers read them through
        // `Argument::value()`, so there is nothing to evaluate.
        if matches!(node, Node::Constant(_)) {
            return Ok(None);
        }

        let in_rank = |i: usize| ins[i].ty.rank();
        let padded = |axis: usize, rank: usize| axis + MAX_RANK - rank;

        let (op, operands): (Op, &[Argument]) = match node {
            Node::Identity(_) => (Op::Identity, &ins[..1]),
            Node::Add(_) => (Op::Add, &ins[..2]),
            Node::Sub(_) => (Op::Sub, &ins[..2]),
            Node::Mul(_) => (Op::Mul, &ins[..2]),
            Node::Div(_) => (Op::Div, &ins[..2]),
            Node::Relu(_) => (Op::Relu, &ins[..1]),
            Node::Tanh(_) => (Op::Tanh, &ins[..1]),
            Node::Sigmoid(_) => (Op::Sigmoid, &ins[..1]),
            Node::LeakyRelu(n) => (Op::LeakyRelu { alpha: n.config.alpha }, &ins[..1]),
            Node::Clip(n) => {
                let bound = |b: &Option<onnx_ir::clip::ClipInput>, default: f32| match b {
                    None => Ok(default),
                    Some(onnx_ir::clip::ClipInput::Static(v)) => Ok(*v as f32),
                    Some(onnx_ir::clip::ClipInput::Runtime(_)) => {
                        Err(err(&name, "Clip bounds computed at runtime are not supported".into()))
                    }
                };
                let op = Op::Clip { min: bound(&n.config.min, f32::MIN)?, max: bound(&n.config.max, f32::MAX)? };
                (op, &ins[..1])
            }
            Node::Softmax(n) => (Op::Softmax { axis: padded(n.config.axis, in_rank(0)) }, &ins[..1]),
            Node::Flatten(n) => (Op::Flatten { axis: n.config.axis }, &ins[..1]),
            Node::Reshape(n) => match &n.config.shape {
                onnx_ir::reshape::ReshapeInput::Static(shape) => (Op::Reshape { shape: shape.clone() }, &ins[..1]),
                onnx_ir::reshape::ReshapeInput::Runtime(_) => {
                    return Err(err(&name, "Reshape to a shape computed at runtime is not supported".into()));
                }
            },
            Node::Concat(n) => (Op::Concat { axis: padded(n.config.axis, in_rank(0)) }, ins),
            Node::MatMul(_) => {
                if in_rank(1) < 2 {
                    return Err(err(&name, "MatMul with a rank-1 right operand is not supported".into()));
                }
                (Op::MatMul, &ins[..2])
            }
            Node::Gemm(n) => {
                let c = &n.config;
                let op = Op::Gemm { alpha: c.alpha, beta: c.beta, trans_a: c.trans_a != 0, trans_b: c.trans_b != 0 };
                (op, ins)
            }
            Node::Linear(n) => (Op::Linear { transpose_weight: n.config.transpose_weight }, ins),
            Node::Conv1d(n) => {
                use onnx_ir::padding::{AutoPad, PaddingConfig1d};
                let c = &n.config;
                let padding = match (&c.auto_pad, &c.padding) {
                    (AutoPad::SameUpper, _) => ConvPadding::Same { upper: true },
                    (AutoPad::SameLower, _) => ConvPadding::Same { upper: false },
                    (AutoPad::Valid, _) | (AutoPad::NotSet, PaddingConfig1d::Valid) => ConvPadding::Explicit(0, 0),
                    (AutoPad::NotSet, PaddingConfig1d::Explicit(l, r)) => ConvPadding::Explicit(*l, *r),
                };
                let op = Op::Conv1d {
                    kernel: c.kernel_size,
                    stride: c.stride,
                    dilation: c.dilation,
                    groups: c.groups,
                    padding,
                };
                (op, ins)
            }
            Node::Conv2d(n) => {
                let c = &n.config;
                let op = Op::Conv2d {
                    kernel: c.kernel_size,
                    stride: c.stride,
                    dilation: c.dilation,
                    groups: c.groups,
                    padding: padding_2d(&c.auto_pad, &c.padding),
                };
                (op, ins)
            }
            Node::MaxPool2d(n) => {
                if outs.iter().skip(1).any(present) {
                    return Err(err(&name, "MaxPool's Indices output is not supported".into()));
                }
                let c = &n.config;
                let op = Op::MaxPool2d {
                    kernel: c.kernel_size,
                    stride: c.strides,
                    dilation: c.dilation,
                    ceil_mode: c.ceil_mode,
                    padding: padding_2d(&c.auto_pad, &c.padding),
                };
                (op, &ins[..1])
            }
            Node::Gru(n) => return self.recurrent(node, recurrent_gru(&n.config).map_err(|m| err(&name, m))?, 2),
            Node::Lstm(n) => return self.recurrent(node, recurrent_lstm(&n.config).map_err(|m| err(&name, m))?, 3),
            other => {
                return Err(err(
                    &name,
                    format!("ONNX op `{}` is not supported yet", other.node_type()),
                ));
            }
        };

        let operands: Vec<&Argument> = operands.iter().filter(|a| present(a)).collect();

        let mut slots = Vec::with_capacity(operands.len());
        for arg in operands {
            check_float_tensor(arg, Some(&name))?;
            slots.push(self.input(arg, &name)?);
        }

        let outs: Vec<&Argument> = outs.iter().filter(|a| present(a)).collect();
        let [out] = outs[..] else {
            return Err(err(&name, format!("expected one output, found {}", outs.len())));
        };
        check_float_tensor(out, Some(&name))?;
        let output = StepOutput { slot: Some(self.define(&out.name)), rank: out.ty.rank(), name: name.clone() };

        Ok(Some(Step { node: name, op, inputs: slots, outputs: vec![output] }))
    }

    /// A GRU (`outputs` = 2: Y, Y_h) or LSTM (3: Y, Y_h, Y_c). Operands are
    /// X, W, R and then the optional inputs that are present.
    fn recurrent(&mut self, node: &Node, op: Op, outputs: usize) -> Result<Option<Step>, CompileError> {
        let name = node.name().to_string();
        let ins = node.inputs();
        let rec = match &op {
            Op::Gru(r) | Op::Lstm(r) => r.clone(),
            _ => unreachable!(),
        };
        let optional = |i: usize| ins.get(i).filter(|a| present(a));
        if optional(4).is_some() {
            return Err(err(&name, "sequence_lens is not supported".into()));
        }
        let mut operands: Vec<&Argument> = ins[..3].iter().collect();
        operands.extend(optional(3).filter(|_| rec.has_bias));
        operands.extend(optional(5).filter(|_| rec.has_initial_h));
        operands.extend(optional(6).filter(|_| rec.has_initial_c));
        let mut slots = Vec::with_capacity(operands.len());
        for arg in operands {
            check_float_tensor(arg, Some(&name))?;
            slots.push(self.input(arg, &name)?);
        }

        let outs = node.outputs();
        if outs.len() > outputs {
            return Err(err(&name, format!("expected at most {outputs} outputs, found {}", outs.len())));
        }
        let ranks = [4, 3, 3];
        let mut step_outputs = Vec::with_capacity(outputs);
        for (i, rank) in ranks.iter().enumerate().take(outputs) {
            let slot = match outs.get(i).filter(|a| present(a)) {
                Some(out) => {
                    check_float_tensor(out, Some(&name))?;
                    Some(self.define(&out.name))
                }
                None => None,
            };
            step_outputs.push(StepOutput { slot, rank: *rank, name: name.clone() });
        }
        Ok(Some(Step { node: name, op, inputs: slots, outputs: step_outputs }))
    }
}

fn eval(step: &Step, args: &[&Value]) -> Result<Vec<Value>, String> {
    match &step.op {
        Op::Gru(r) => return gru(r, args),
        Op::Lstm(r) => return lstm(r, args),
        _ => {}
    }
    eval_single(step, args).map(|v| vec![v])
}

fn eval_single(step: &Step, args: &[&Value]) -> Result<Value, String> {
    let rank = step.outputs[0].rank;
    let unary = |f: fn(Tensor<MAX_RANK>) -> Tensor<MAX_RANK>| {
        Ok::<_, String>(Value { tensor: f(args[0].tensor.clone()), rank })
    };
    let binary = |f: fn(Tensor<MAX_RANK>, Tensor<MAX_RANK>) -> Tensor<MAX_RANK>| {
        Ok::<_, String>(Value { tensor: f(args[0].tensor.clone(), args[1].tensor.clone()), rank })
    };

    match &step.op {
        Op::Identity => Ok(args[0].clone()),
        Op::Add => binary(|a, b| a + b),
        Op::Sub => binary(|a, b| a - b),
        Op::Mul => binary(|a, b| a * b),
        Op::Div => binary(|a, b| a / b),
        Op::Relu => unary(activation::relu),
        Op::Tanh => unary(activation::tanh),
        Op::Sigmoid => unary(activation::sigmoid),
        Op::LeakyRelu { alpha } => Ok(args[0].with(activation::leaky_relu(args[0].tensor.clone(), *alpha))),
        Op::Clip { min, max } => Ok(args[0].with(args[0].tensor.clone().clamp(*min, *max))),
        Op::Softmax { axis } => Ok(args[0].with(activation::softmax(args[0].tensor.clone(), *axis))),
        Op::Flatten { axis } => {
            let shape = args[0].shape();
            let outer: usize = shape[..*axis].iter().product();
            let inner: usize = shape[*axis..].iter().product();
            reshape(args[0], &[outer, inner])
        }
        Op::Reshape { shape: spec } => {
            let target = resolve_reshape(&args[0].shape(), spec)?;
            reshape(args[0], &target)
        }
        Op::Concat { axis } => {
            let tensors = args.iter().map(|v| v.tensor.clone()).collect();
            Ok(Value { tensor: Tensor::cat(tensors, *axis), rank })
        }
        Op::MatMul => binary(|a, b| a.matmul(b)),
        Op::Gemm { alpha, beta, trans_a, trans_b } => {
            let a = transpose_if(args[0].tensor.clone(), *trans_a);
            let b = transpose_if(args[1].tensor.clone(), *trans_b);
            let mut y = a.matmul(b);
            if *alpha != 1.0 {
                y = y * *alpha;
            }
            if let Some(c) = args.get(2) {
                y = y + c.tensor.clone() * *beta;
            }
            Ok(Value { tensor: y, rank })
        }
        Op::Linear { transpose_weight } => {
            let w = transpose_if(args[1].tensor.clone(), *transpose_weight);
            let mut y = args[0].tensor.clone().matmul(w);
            if let Some(bias) = args.get(2) {
                y = y + bias.tensor.clone();
            }
            Ok(Value { tensor: y, rank })
        }
        Op::Conv1d { kernel, stride, dilation, groups, padding } => {
            let x = rank3(args[0]);
            let w = rank3(args[1]);
            let bias = args.get(2).map(|b| b.tensor.clone().reshape([b.tensor.dims()[MAX_RANK - 1]]));
            let length = x.dims()[2];
            let padding = padding.resolve(length, *kernel, *stride, *dilation);
            let options = ConvOptions { stride: [*stride], padding: [padding], dilation: [*dilation], groups: *groups };
            Ok(from_rank3(module::conv1d(x, w, bias, options)))
        }
        Op::Conv2d { kernel, stride, dilation, groups, padding } => {
            let x = spatial_input(args[0])?;
            let bias = args.get(2).map(|b| b.tensor.clone().reshape([b.tensor.dims()[MAX_RANK - 1]]));
            let padding = resolve_2d(padding, &x, kernel, stride, dilation);
            let options = ConvOptions { stride: *stride, padding, dilation: *dilation, groups: *groups };
            Ok(Value { tensor: module::conv2d(x, args[1].tensor.clone(), bias, options), rank: 4 })
        }
        Op::MaxPool2d { kernel, stride, dilation, ceil_mode, padding } => {
            let x = spatial_input(args[0])?;
            let padding = resolve_2d(padding, &x, kernel, stride, dilation);
            let options = MaxPoolOptions { kernel_size: *kernel, stride: *stride, padding, dilation: *dilation, ceil_mode: *ceil_mode };
            Ok(Value { tensor: module::max_pool2d(x, options), rank: 4 })
        }
        Op::Gru(_) | Op::Lstm(_) => unreachable!("handled in eval"),
    }
}

impl ConvPadding {
    /// `(begin, end)` padding for one spatial axis of `length`.
    fn resolve(self, length: usize, kernel: usize, stride: usize, dilation: usize) -> (usize, usize) {
        match self {
            ConvPadding::Explicit(begin, end) => (begin, end),
            ConvPadding::Same { upper } => {
                let out = length.div_ceil(stride);
                let span = (kernel - 1) * dilation + 1;
                let total = ((out - 1) * stride + span).saturating_sub(length);
                let (small, big) = (total / 2, total - total / 2);
                if upper { (small, big) } else { (big, small) }
            }
        }
    }
}

fn padding_2d(auto_pad: &onnx_ir::padding::AutoPad, padding: &onnx_ir::padding::PaddingConfig2d) -> [ConvPadding; 2] {
    use onnx_ir::padding::AutoPad;
    match auto_pad {
        AutoPad::SameUpper => [ConvPadding::Same { upper: true }; 2],
        AutoPad::SameLower => [ConvPadding::Same { upper: false }; 2],
        AutoPad::Valid => [ConvPadding::Explicit(0, 0); 2],
        AutoPad::NotSet => {
            let (top, left, bottom, right) = padding.as_tuple();
            [ConvPadding::Explicit(top, bottom), ConvPadding::Explicit(left, right)]
        }
    }
}

fn resolve_2d(
    padding: &[ConvPadding; 2],
    x: &Tensor<4>,
    kernel: &[usize; 2],
    stride: &[usize; 2],
    dilation: &[usize; 2],
) -> [(usize, usize); 2] {
    let dims = x.dims();
    std::array::from_fn(|i| padding[i].resolve(dims[2 + i], kernel[i], stride[i], dilation[i]))
}

/// A 2D conv or pool input must be `[N, C, H, W]`.
fn spatial_input(v: &Value) -> Result<Tensor<4>, String> {
    if v.rank != 4 {
        return Err(format!("expected an [N, C, H, W] input, got shape {:?}", v.shape()));
    }
    Ok(v.tensor.clone())
}

fn rank3(v: &Value) -> Tensor<3> {
    let [_, a, b, c] = v.tensor.dims();
    v.tensor.clone().reshape([a, b, c])
}

fn from_rank3(t: Tensor<3>) -> Value {
    let [a, b, c] = t.dims();
    Value { tensor: t.reshape([1, a, b, c]), rank: 3 }
}

/// Row `d` of a `[n, x, y]` tensor, as `[x, y]`.
fn row(t: &Tensor<3>, d: usize) -> Tensor<2> {
    let [_, x, y] = t.dims();
    t.clone().narrow(0, d, 1).reshape([x, y])
}

/// Columns `[gate * h, (gate + 1) * h)` of a `[rows, k * h]` tensor.
fn gate(t: &Tensor<2>, gate: usize, h: usize) -> Tensor<2> {
    t.clone().narrow(1, gate * h, h)
}

fn clip(t: Tensor<2>, c: Option<f32>) -> Tensor<2> {
    match c {
        Some(c) => t.clamp(-c, c),
        None => t,
    }
}

fn one_minus(t: Tensor<2>) -> Tensor<2> {
    (t * -1.0).add_scalar(1.0)
}

/// Per-direction weights and initial state shared by GRU and LSTM.
struct RecurrentArgs {
    x: Tensor<3>,
    /// W^T and R^T per direction: `[input, k*h]` and `[h, k*h]`.
    w_t: Vec<Tensor<2>>,
    r_t: Vec<Tensor<2>>,
    /// W and R biases per direction, `[1, k*h]` each (zeros when absent).
    w_b: Vec<Tensor<2>>,
    r_b: Vec<Tensor<2>>,
    h0: Vec<Tensor<2>>,
    c0: Vec<Tensor<2>>,
    seq: usize,
    batch: usize,
}

fn recurrent_args(r: &Recurrent, args: &[&Value], gates: usize, lstm: bool) -> Result<RecurrentArgs, String> {
    let h = r.hidden;
    let x = rank3(args[0]);
    let [seq, batch, _] = x.dims();
    let w = rank3(args[1]);
    let rr = rank3(args[2]);
    let directions = w.dims()[0];
    let expected = if r.directions == Direction::Bidirectional { 2 } else { 1 };
    if directions != expected {
        return Err(format!("W has {directions} directions, expected {expected}"));
    }
    let device = x.device();
    let mut next = 3;
    let mut take = |present: bool| {
        present.then(|| {
            next += 1;
            args[next - 1]
        })
    };
    let bias = take(r.has_bias).map(|b| {
        let [_, _, d, n] = b.tensor.dims();
        b.tensor.clone().reshape([d, n])
    });
    let h0 = take(r.has_initial_h).map(rank3);
    let c0 = take(lstm && r.has_initial_c).map(rank3);

    let zeros = |rows: usize, cols: usize| Tensor::<2>::zeros([rows, cols], &device);
    let mut out = RecurrentArgs {
        x,
        w_t: vec![],
        r_t: vec![],
        w_b: vec![],
        r_b: vec![],
        h0: vec![],
        c0: vec![],
        seq,
        batch,
    };
    for d in 0..directions {
        out.w_t.push(row(&w, d).transpose());
        out.r_t.push(row(&rr, d).transpose());
        match &bias {
            Some(b) => {
                let b = b.clone().narrow(0, d, 1);
                out.w_b.push(b.clone().narrow(1, 0, gates * h));
                out.r_b.push(b.narrow(1, gates * h, gates * h));
            }
            None => {
                out.w_b.push(zeros(1, gates * h));
                out.r_b.push(zeros(1, gates * h));
            }
        }
        out.h0.push(h0.as_ref().map_or_else(|| zeros(batch, h), |t| row(t, d)));
        out.c0.push(c0.as_ref().map_or_else(|| zeros(batch, h), |t| row(t, d)));
    }
    Ok(out)
}

/// Time steps in processing order for direction `d`.
fn time_order(r: &Recurrent, d: usize, seq: usize) -> Vec<usize> {
    let reverse = r.directions == Direction::Reverse || d == 1;
    if reverse { (0..seq).rev().collect() } else { (0..seq).collect() }
}

/// Stack per-direction `[seq] x [batch, h]` histories into Y `[seq, dirs, batch, h]`
/// and final states into `[dirs, batch, h]`.
fn recurrent_outputs(histories: Vec<Vec<Tensor<2>>>, finals: Vec<Vec<Tensor<2>>>) -> Vec<Value> {
    let per_direction: Vec<Tensor<3>> = histories.into_iter().map(|steps| Tensor::stack(steps, 0)).collect();
    let y: Tensor<4> = Tensor::stack(per_direction, 1);
    let mut out = vec![Value { tensor: y, rank: 4 }];
    for states in finals {
        out.push(from_rank3(Tensor::stack(states, 0)));
    }
    out
}

/// ONNX GRU, gates z, r, h.
fn gru(r: &Recurrent, args: &[&Value]) -> Result<Vec<Value>, String> {
    let h = r.hidden;
    let a = recurrent_args(r, args, 3, false)?;
    let (mut histories, mut finals) = (vec![], vec![]);
    for d in 0..a.w_t.len() {
        let mut ht = a.h0[d].clone();
        let mut history = vec![None; a.seq];
        let rh_t = gate(&a.r_t[d], 2, h);
        for t in time_order(r, d, a.seq) {
            let xt = a.x.clone().narrow(0, t, 1).reshape([a.batch, a.x.dims()[2]]);
            let gx = xt.matmul(a.w_t[d].clone()) + a.w_b[d].clone();
            let gh = ht.clone().matmul(a.r_t[d].clone()) + a.r_b[d].clone();
            let z = r.gate.apply(clip(gate(&gx, 0, h) + gate(&gh, 0, h), r.clip));
            let reset = r.gate.apply(clip(gate(&gx, 1, h) + gate(&gh, 1, h), r.clip));
            let candidate = if r.linear_before_reset {
                gate(&gx, 2, h) + reset * gate(&gh, 2, h)
            } else {
                gate(&gx, 2, h) + (reset * ht.clone()).matmul(rh_t.clone()) + gate(&a.r_b[d], 2, h)
            };
            let candidate = r.cell.apply(clip(candidate, r.clip));
            ht = one_minus(z.clone()) * candidate + z * ht;
            history[t] = Some(ht.clone());
        }
        histories.push(history.into_iter().map(|t| t.expect("every step ran")).collect());
        finals.push(ht);
    }
    Ok(recurrent_outputs(histories, vec![finals]))
}

/// ONNX LSTM, gates i, o, f, c.
fn lstm(r: &Recurrent, args: &[&Value]) -> Result<Vec<Value>, String> {
    let h = r.hidden;
    let a = recurrent_args(r, args, 4, true)?;
    let (mut histories, mut finals_h, mut finals_c) = (vec![], vec![], vec![]);
    for d in 0..a.w_t.len() {
        let (mut ht, mut ct) = (a.h0[d].clone(), a.c0[d].clone());
        let mut history = vec![None; a.seq];
        for t in time_order(r, d, a.seq) {
            let xt = a.x.clone().narrow(0, t, 1).reshape([a.batch, a.x.dims()[2]]);
            let g = xt.matmul(a.w_t[d].clone()) + a.w_b[d].clone() + ht.clone().matmul(a.r_t[d].clone()) + a.r_b[d].clone();
            let i = r.gate.apply(clip(gate(&g, 0, h), r.clip));
            let o = r.gate.apply(clip(gate(&g, 1, h), r.clip));
            let f = if r.input_forget { one_minus(i.clone()) } else { r.gate.apply(clip(gate(&g, 2, h), r.clip)) };
            let c = r.cell.apply(clip(gate(&g, 3, h), r.clip));
            ct = f * ct + i * c;
            ht = o * r.hidden_act.apply(ct.clone());
            history[t] = Some(ht.clone());
        }
        histories.push(history.into_iter().map(|t| t.expect("every step ran")).collect());
        finals_h.push(ht);
        finals_c.push(ct);
    }
    Ok(recurrent_outputs(histories, vec![finals_h, finals_c]))
}

fn recurrent_gru(c: &onnx_ir::gru::GruConfig) -> Result<Op, String> {
    use onnx_ir::gru::{GruActivationFunction as F, GruDirection as D};
    let act = |f: F| match f {
        F::Sigmoid => Ok(Act::Sigmoid),
        F::Tanh => Ok(Act::Tanh),
        F::Relu => Ok(Act::Relu),
        other => Err(format!("GRU activation {other:?} is not supported")),
    };
    if c.batch_first {
        return Err("GRU with layout=1 (batch first) is not supported".into());
    }
    Ok(Op::Gru(Recurrent {
        hidden: c.hidden_size,
        directions: match c.direction {
            D::Forward => Direction::Forward,
            D::Reverse => Direction::Reverse,
            D::Bidirectional => Direction::Bidirectional,
        },
        has_bias: c.has_bias,
        has_initial_h: c.has_initial_h,
        has_initial_c: false,
        clip: c.clip,
        linear_before_reset: c.linear_before_reset,
        input_forget: false,
        gate: act(c.gate_activation)?,
        cell: act(c.hidden_activation)?,
        hidden_act: Act::Tanh,
    }))
}

fn recurrent_lstm(c: &onnx_ir::lstm::LstmConfig) -> Result<Op, String> {
    use onnx_ir::lstm::{LstmActivationFunction as F, LstmDirection as D};
    let act = |f: F| match f {
        F::Sigmoid => Ok(Act::Sigmoid),
        F::Tanh => Ok(Act::Tanh),
        F::Relu => Ok(Act::Relu),
        other => Err(format!("LSTM activation {other:?} is not supported")),
    };
    if c.batch_first {
        return Err("LSTM with layout=1 (batch first) is not supported".into());
    }
    Ok(Op::Lstm(Recurrent {
        hidden: c.hidden_size,
        directions: match c.direction {
            D::Forward => Direction::Forward,
            D::Reverse => Direction::Reverse,
            D::Bidirectional => Direction::Bidirectional,
        },
        has_bias: c.has_bias,
        has_initial_h: c.has_initial_h,
        has_initial_c: c.has_initial_c,
        clip: c.clip,
        linear_before_reset: false,
        input_forget: c.input_forget,
        gate: act(c.gate_activation)?,
        cell: act(c.cell_activation)?,
        hidden_act: act(c.hidden_activation)?,
    }))
}

/// Whether an optional input or output was actually provided.
fn present(arg: &Argument) -> bool {
    !arg.is_optional() && !(arg.name.is_empty() && arg.value().is_none())
}

fn transpose_if(t: Tensor<MAX_RANK>, transpose: bool) -> Tensor<MAX_RANK> {
    if transpose { t.swap_dims(MAX_RANK - 2, MAX_RANK - 1) } else { t }
}

fn reshape(value: &Value, shape: &[usize]) -> Result<Value, String> {
    if shape.len() > MAX_RANK {
        return Err(format!("reshape to rank {} exceeds the supported {MAX_RANK}", shape.len()));
    }
    Ok(Value { tensor: value.tensor.clone().reshape(pad(shape)), rank: shape.len() })
}

/// Apply ONNX reshape rules: 0 copies the input dim, -1 takes the remainder.
fn resolve_reshape(input: &[usize], spec: &[i64]) -> Result<Vec<usize>, String> {
    let total: usize = input.iter().product();
    let mut out = Vec::with_capacity(spec.len());
    let mut infer = None;
    for (i, &d) in spec.iter().enumerate() {
        match d {
            0 => out.push(*input.get(i).ok_or_else(|| format!("reshape dim {i} copies a missing input dim"))?),
            -1 if infer.is_none() => {
                infer = Some(i);
                out.push(1);
            }
            d if d > 0 => out.push(d as usize),
            d => return Err(format!("invalid reshape dim {d}")),
        }
    }
    if let Some(i) = infer {
        let known: usize = out.iter().product();
        if known == 0 || total % known != 0 {
            return Err(format!("cannot reshape {input:?} to {spec:?}"));
        }
        out[i] = total / known;
    }
    if out.iter().product::<usize>() != total {
        return Err(format!("cannot reshape {input:?} to {spec:?}"));
    }
    Ok(out)
}

fn pad(shape: &[usize]) -> [usize; MAX_RANK] {
    let mut padded = [1; MAX_RANK];
    padded[MAX_RANK - shape.len()..].copy_from_slice(shape);
    padded
}

fn to_value(host: &HostTensor, device: &Device) -> Value {
    let data = TensorData::new(host.data.clone(), pad(&host.shape).to_vec());
    Value { tensor: Tensor::from_data(data, device), rank: host.shape.len() }
}

fn to_host(value: &Value) -> Result<HostTensor, RunError> {
    let data = value
        .tensor
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|e| RunError(format!("could not read back an output: {e:?}")))?;
    Ok(HostTensor::new(value.shape(), data))
}

fn port(arg: &Argument) -> Port {
    let shape = match arg.ty.static_shape() {
        Some(dims) => dims.clone(),
        None => vec![None; arg.ty.rank()],
    };
    Port { name: arg.name.clone(), shape }
}

fn check_float_tensor(arg: &Argument, node: Option<&str>) -> Result<(), CompileError> {
    let fail = |message: String| CompileError { node: node.map(str::to_string), message };
    if arg.ty.is_shape() {
        return Err(fail(format!("`{}` is a shape value; shape arithmetic is not supported", arg.name)));
    }
    if arg.ty.rank() > MAX_RANK {
        return Err(fail(format!("`{}` has rank {}; at most {MAX_RANK} is supported", arg.name, arg.ty.rank())));
    }
    // Constants of any numeric type are converted to f32 when materialized;
    // computed values must already be float.
    let dtype = arg.ty.elem_type();
    if arg.value().is_none() && !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
        return Err(fail(format!("`{}` has element type {dtype:?}; only float tensors are supported", arg.name)));
    }
    Ok(())
}

fn check_input_shape(port: &Port, host: &HostTensor) -> Result<(), RunError> {
    let fits = port.shape.len() == host.shape.len()
        && port.shape.iter().zip(&host.shape).all(|(want, got)| want.map_or(true, |w| w == *got));
    if !fits {
        return Err(RunError(format!("input `{}` expects shape {:?}, got {:?}", port.name, port.shape, host.shape)));
    }
    if host.data.len() != host.shape.iter().product::<usize>() {
        return Err(RunError(format!(
            "input `{}` has {} values but shape {:?}",
            port.name,
            host.data.len(),
            host.shape
        )));
    }
    Ok(())
}

fn err(node: &str, message: String) -> CompileError {
    CompileError { node: Some(node.to_string()), message }
}
