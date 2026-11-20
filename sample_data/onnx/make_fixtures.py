"""Generate the ONNX fixtures used by tests/onnx_runtime.rs.

Each fixture is `<name>.onnx` plus `<name>.json` holding the test inputs and
the expected outputs, computed with onnx's own reference evaluator (an
implementation independent of ours), or with onnxruntime where the reference
evaluator falls short (it ignores LSTM's `clip`).

    uv run --with onnx --with onnxruntime --with numpy python -I sample_data/onnx/make_fixtures.py

Rerun after editing; commit the regenerated files.
"""

import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

OUT = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def f32(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def save(name, nodes, inputs, outputs, initializers, feeds, evaluator="reference"):
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
        [numpy_helper.from_array(a, n) for n, a in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT / f"{name}.onnx")

    if evaluator == "onnxruntime":
        import onnxruntime

        session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        results = session.run(None, feeds)
    else:
        results = ReferenceEvaluator(model).run(None, feeds)
    expected = {
        "inputs": [
            {"name": n, "shape": list(feeds[n].shape), "data": feeds[n].ravel().tolist()}
            for n, _ in inputs
        ],
        "outputs": [
            {"name": n, "shape": list(r.shape), "data": r.ravel().tolist()}
            for (n, _), r in zip(outputs, results)
        ],
    }
    (OUT / f"{name}.json").write_text(json.dumps(expected, indent=1) + "\n")


# A controller-shaped MLP as PyTorch exports it: Gemm with transB=1
# (weights stored [out, in]), tanh hidden layer.
save(
    "mlp",
    [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h"], transB=1),
        helper.make_node("Tanh", ["h"], ["a"]),
        helper.make_node("Gemm", ["a", "w2", "b2"], ["y"], transB=1),
    ],
    inputs=[("x", [1, 4])],
    outputs=[("y", [1, 2])],
    initializers={"w1": f32(8, 4), "b1": f32(8), "w2": f32(2, 8), "b2": f32(2)},
    feeds={"x": np.array([[0.5, -1.0, 2.0, 0.25]], np.float32)},
)

# Every op in the first op set, on a rank-3 input so broadcasting and the
# rank padding get exercised.
save(
    "ops",
    [
        # [2,3,4] @ [4,5] -> [2,3,5]; + [5] broadcasts.
        helper.make_node("MatMul", ["x", "w"], ["mm"]),
        helper.make_node("Add", ["mm", "bias"], ["lin"]),
        helper.make_node("Relu", ["lin"], ["relu"]),
        helper.make_node("Sigmoid", ["lin"], ["sig"]),
        helper.make_node("LeakyRelu", ["lin"], ["leaky"], alpha=0.1),
        helper.make_node("Clip", ["lin", "lo", "hi"], ["clip"]),
        # Elementwise with a [3,1] operand: broadcast over batch and columns.
        helper.make_node("Sub", ["relu", "col"], ["sub"]),
        helper.make_node("Mul", ["sub", "sig"], ["mul"]),
        helper.make_node("Div", ["mul", "two"], ["div"]),
        helper.make_node("Identity", ["div"], ["ident"]),
        # Concat on the last axis -> [2,3,15], then softmax over it.
        helper.make_node("Concat", ["ident", "leaky", "clip"], ["cat"], axis=-1),
        helper.make_node("Softmax", ["cat"], ["soft"], axis=-1),
        # [2,3,15] -> [2,45] -> [2,5,9] -> flatten(axis=1) -> [2,45].
        helper.make_node("Flatten", ["soft"], ["flat"], axis=1),
        helper.make_node("Reshape", ["flat", "shape"], ["resh"]),
        helper.make_node("Flatten", ["resh"], ["y"], axis=1),
    ],
    inputs=[("x", [2, 3, 4])],
    outputs=[("y", [2, 45])],
    initializers={
        "w": f32(4, 5),
        "bias": f32(5),
        "col": f32(3, 1),
        "two": np.array(2.0, np.float32),
        "lo": np.array(-0.5, np.float32),
        "hi": np.array(0.5, np.float32),
        "shape": np.array([0, 5, -1], np.int64),
    },
    feeds={"x": f32(2, 3, 4)},
)

# onnx-ir fuses MatMul+Add into Linear and drops Identity, so "ops" never
# reaches a bare MatMul. Here both MatMul operands are computed (nothing to
# fuse), with batch broadcasting [2,3,4] @ [1,4,5]; plus a Gemm with
# non-default alpha, beta and transA.
save(
    "matmul",
    [
        helper.make_node("Tanh", ["a"], ["ta"]),
        helper.make_node("Sigmoid", ["b"], ["sb"]),
        helper.make_node("MatMul", ["ta", "sb"], ["y"]),
        helper.make_node("Gemm", ["g", "gw", "gc"], ["z"], alpha=0.5, beta=2.0, transA=1),
    ],
    inputs=[("a", [2, 3, 4]), ("b", [1, 4, 5]), ("g", [4, 3])],
    outputs=[("y", [2, 3, 5]), ("z", [3, 2])],
    initializers={"gw": f32(4, 2), "gc": f32(1, 2)},
    feeds={"a": f32(2, 3, 4), "b": f32(1, 4, 5), "g": f32(4, 3)},
)

# The soft voltage clamp from nb-lib's models/: drive = tanh(gain * (target - v)),
# here with two taps so per-tap outputs and broadcasting are exercised.
save(
    "clamp",
    [
        helper.make_node("Sub", ["target_mv", "v_mv"], ["error_mv"]),
        helper.make_node("Mul", ["error_mv", "gain_per_mv"], ["scaled"]),
        helper.make_node("Tanh", ["scaled"], ["drive"]),
    ],
    inputs=[("v_mv", [1, 2]), ("target_mv", [1, 1]), ("gain_per_mv", [1, 1])],
    outputs=[("drive", [1, 2])],
    initializers={},
    feeds={
        "v_mv": np.array([[-70.0, -40.0]], np.float32),
        "target_mv": np.array([[-50.0]], np.float32),
        "gain_per_mv": np.array([[0.1]], np.float32),
    },
)

# Conv1d: explicit asymmetric pads with stride and dilation, then a grouped
# conv with SAME_UPPER auto padding and no bias.
save(
    "conv1d",
    [
        helper.make_node("Conv", ["x", "w1", "b1"], ["c1"], pads=[1, 2], strides=[2], dilations=[2]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2"], ["y"], group=2, auto_pad="SAME_UPPER"),
    ],
    inputs=[("x", [2, 2, 9])],
    outputs=[("y", [2, 4, 4])],
    initializers={"w1": f32(4, 2, 3), "b1": f32(4), "w2": f32(4, 2, 2)},
    feeds={"x": f32(2, 2, 9)},
)

H, I, SEQ, BATCH = 4, 3, 3, 2

# GRU: forward with linear_before_reset (as PyTorch exports it), bias and an
# initial state fed in as a graph input; and bidirectional without either,
# with the default linear_before_reset=0.
save(
    "gru",
    [
        helper.make_node("GRU", ["x", "w", "r", "b", "", "h0"], ["y", "y_h"], hidden_size=H, linear_before_reset=1),
        helper.make_node("GRU", ["x", "w2", "r2"], ["y2"], hidden_size=H, direction="bidirectional"),
    ],
    inputs=[("x", [SEQ, BATCH, I]), ("h0", [1, BATCH, H])],
    outputs=[("y", [SEQ, 1, BATCH, H]), ("y_h", [1, BATCH, H]), ("y2", [SEQ, 2, BATCH, H])],
    initializers={
        "w": 0.5 * f32(1, 3 * H, I),
        "r": 0.5 * f32(1, 3 * H, H),
        "b": 0.5 * f32(1, 6 * H),
        "w2": 0.5 * f32(2, 3 * H, I),
        "r2": 0.5 * f32(2, 3 * H, H),
    },
    feeds={"x": f32(SEQ, BATCH, I), "h0": 0.5 * f32(1, BATCH, H)},
)

# LSTM: forward with bias and initial h and c; and reverse with clipping,
# using only Y. Checked with onnxruntime: the reference evaluator ignores clip.
save(
    "lstm",
    [
        helper.make_node("LSTM", ["x", "w", "r", "b", "", "h0", "c0"], ["y", "y_h", "y_c"], hidden_size=H),
        helper.make_node("LSTM", ["x", "w2", "r2"], ["y2"], hidden_size=H, direction="reverse", clip=0.5),
    ],
    inputs=[("x", [SEQ, BATCH, I]), ("h0", [1, BATCH, H]), ("c0", [1, BATCH, H])],
    outputs=[
        ("y", [SEQ, 1, BATCH, H]),
        ("y_h", [1, BATCH, H]),
        ("y_c", [1, BATCH, H]),
        ("y2", [SEQ, 1, BATCH, H]),
    ],
    initializers={
        "w": 0.5 * f32(1, 4 * H, I),
        "r": 0.5 * f32(1, 4 * H, H),
        "b": 0.5 * f32(1, 8 * H),
        "w2": 0.5 * f32(1, 4 * H, I),
        "r2": 0.5 * f32(1, 4 * H, H),
    },
    feeds={"x": f32(SEQ, BATCH, I), "h0": 0.5 * f32(1, BATCH, H), "c0": 0.5 * f32(1, BATCH, H)},
    evaluator="onnxruntime",
)

# An op outside the supported set; compiling it must fail with a clear error.
save(
    "unsupported",
    [helper.make_node("Cos", ["x"], ["y"])],
    inputs=[("x", [1, 4])],
    outputs=[("y", [1, 4])],
    initializers={},
    feeds={"x": f32(1, 4)},
)
