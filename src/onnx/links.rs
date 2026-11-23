//! Which input elements each element of a traced value depends on, so the
//! 3D view can draw a node's connections.
//!
//! [`Link`] says how an op maps its inputs to its output; [`sources`] and
//! [`consumers`] walk it for one element. Weights and other constants are
//! not traced, so only connections between traced values are reported.

use crate::onnx::Activation;

/// How an op's output elements depend on its (traced) inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum Link {
    /// A graph input, or an op whose element dependencies aren't modeled
    /// (recurrent cells, Conv1d).
    Opaque,
    /// Each output element reads the same position of every input, with
    /// numpy broadcasting (Add, Relu, ...).
    Elementwise,
    /// Output element i is input element i (Reshape, Flatten, Identity).
    SameIndex,
    /// `[N, C, H, W]` sliding window. `channels_per_group` is the number of
    /// input channels each output channel reads (Conv2d), or `None` when an
    /// output channel reads only its own input channel (pooling).
    Window {
        kernel: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        /// Padding before each spatial axis.
        pad: [usize; 2],
        channels_per_group: Option<usize>,
        out_channels_per_group: usize,
    },
    /// Matrix product over the last two dims: out[.., m, n] reads row m of
    /// input 0 and column n of input 1 (where they're traced).
    Dense { trans_a: bool, trans_b: bool },
    /// Every element along a logical axis (Softmax).
    Axis { axis: usize },
    /// Inputs stacked along a logical axis.
    Concat { axis: usize },
}

/// The elements `(trace index, flat index)` that element `index` of
/// `trace[layer]` is computed from.
pub fn sources(trace: &[Activation], layer: usize, index: usize) -> Vec<(usize, usize)> {
    let act = &trace[layer];
    let out_shape = &act.tensor.shape;
    if index >= act.tensor.data.len() {
        return Vec::new();
    }
    let o = unravel(index, out_shape);
    let input = |j: usize| act.inputs.get(j).copied().flatten().map(|t| (t, &trace[t].tensor.shape));
    let mut found = Vec::new();
    match &act.link {
        Link::Opaque => {}
        Link::Elementwise => {
            for j in 0..act.inputs.len() {
                if let Some((t, shape)) = input(j) {
                    found.push((t, ravel(&broadcast(&o, shape), shape)));
                }
            }
        }
        Link::SameIndex => {
            if let Some((t, shape)) = input(0) {
                if index < shape.iter().product::<usize>() {
                    found.push((t, index));
                }
            }
        }
        Link::Window { kernel, stride, dilation, pad, channels_per_group, out_channels_per_group } => {
            let Some((t, shape)) = input(0) else { return found };
            if shape.len() != 4 || o.len() != 4 {
                return found;
            }
            let channels = match channels_per_group {
                Some(per) => {
                    let group = o[1] / out_channels_per_group.max(&1);
                    group * per..(group + 1) * per
                }
                None => o[1]..o[1] + 1,
            };
            for c in channels.filter(|c| *c < shape[1]) {
                for ky in 0..kernel[0] {
                    for kx in 0..kernel[1] {
                        let y = (o[2] * stride[0] + ky * dilation[0]).checked_sub(pad[0]);
                        let x = (o[3] * stride[1] + kx * dilation[1]).checked_sub(pad[1]);
                        if let (Some(y), Some(x)) = (y, x) {
                            if y < shape[2] && x < shape[3] {
                                found.push((t, ravel(&[o[0].min(shape[0] - 1), c, y, x], shape)));
                            }
                        }
                    }
                }
            }
        }
        Link::Dense { trans_a, trans_b } => {
            let rank = o.len();
            let (m, n) = match rank {
                0 => return found,
                1 => (0, o[0]),
                _ => (o[rank - 2], o[rank - 1]),
            };
            let lead = &o[..rank.saturating_sub(2)];
            if let Some((t, shape)) = input(0) {
                found.extend(matrix_line(shape, lead, m, *trans_a, true).into_iter().map(|i| (t, i)));
            }
            if let Some((t, shape)) = input(1) {
                found.extend(matrix_line(shape, lead, n, *trans_b, false).into_iter().map(|i| (t, i)));
            }
        }
        Link::Axis { axis } => {
            if let Some((t, shape)) = input(0) {
                if shape == out_shape && *axis < shape.len() {
                    let mut c = o.clone();
                    for k in 0..shape[*axis] {
                        c[*axis] = k;
                        found.push((t, ravel(&c, shape)));
                    }
                }
            }
        }
        Link::Concat { axis } => {
            let mut offset = 0;
            for j in 0..act.inputs.len() {
                let Some(shape) = act.input_shapes.get(j) else { break };
                let size = shape.get(*axis).copied().unwrap_or(0);
                if o.get(*axis).is_some_and(|&p| p >= offset && p < offset + size) {
                    if let Some((t, shape)) = input(j) {
                        let mut c = o.clone();
                        c[*axis] -= offset;
                        found.push((t, ravel(&c, shape)));
                    }
                    break;
                }
                offset += size;
            }
        }
    }
    found
}

/// The elements of later layers that read element `index` of `trace[layer]`.
pub fn consumers(trace: &[Activation], layer: usize, index: usize) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    for (t, act) in trace.iter().enumerate().skip(layer + 1) {
        if !act.inputs.contains(&Some(layer)) {
            continue;
        }
        match act.link {
            // Shortcut the common one-to-one case.
            Link::SameIndex => {
                if index < act.tensor.data.len() {
                    found.push((t, index));
                }
            }
            Link::Opaque => {}
            _ => {
                for i in 0..act.tensor.data.len() {
                    if sources(trace, t, i).contains(&(layer, index)) {
                        found.push((t, i));
                    }
                }
            }
        }
    }
    found
}

/// Row `m` (for input 0) or column `n` (for input 1) of a matrix operand,
/// as flat indices; `lead` are the output's batch coordinates.
fn matrix_line(shape: &[usize], lead: &[usize], line: usize, transposed: bool, left: bool) -> Vec<usize> {
    let rank = shape.len();
    if rank == 0 {
        return Vec::new();
    }
    if rank == 1 {
        return (0..shape[0]).collect();
    }
    let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
    // For the left operand `line` is a row of A; for the right, a column of B.
    let along_rows = left == transposed;
    let mut base = broadcast(lead, &shape[..rank - 2]);
    base.extend([0, 0]);
    let (count, fixed) = if along_rows { (rows, cols) } else { (cols, rows) };
    if line >= fixed {
        return Vec::new();
    }
    (0..count)
        .map(|k| {
            let (r, c) = if along_rows { (k, line) } else { (line, k) };
            base[rank - 2] = r;
            base[rank - 1] = c;
            ravel(&base, shape)
        })
        .collect()
}

/// Map output coordinates onto an input `shape` with numpy broadcasting
/// (shapes aligned on the right; size-1 dims always read 0).
fn broadcast(out: &[usize], shape: &[usize]) -> Vec<usize> {
    let skip = out.len().saturating_sub(shape.len());
    let pad = shape.len().saturating_sub(out.len());
    shape
        .iter()
        .enumerate()
        .map(|(i, &d)| if d == 1 || i < pad { 0 } else { out[skip + i - pad] })
        .collect()
}

pub fn unravel(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for (c, d) in coords.iter_mut().zip(shape).rev() {
        if *d > 0 {
            *c = index % d;
            index /= d;
        }
    }
    coords
}

pub fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    coords.iter().zip(shape).fold(0, |acc, (c, d)| acc * d + c)
}
