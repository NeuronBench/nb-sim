"""Generate the ONNX fixtures used by tests/onnx_runtime.rs.

Each fixture is `<name>.onnx` plus `<name>.json` holding the test inputs and
the expected outputs, computed with onnx's own reference evaluator (an
implementation independent of ours).

    uv run --with onnx --with numpy python -I sample_data/onnx/make_fixtures.py

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


def save(name, nodes, inputs, outputs, initializers, feeds):
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

# An op outside the supported set; compiling it must fail with a clear error.
save(
    "unsupported",
    [helper.make_node("Cos", ["x"], ["y"])],
    inputs=[("x", [1, 4])],
    outputs=[("y", [1, 4])],
    initializers={},
    feeds={"x": f32(1, 4)},
)
