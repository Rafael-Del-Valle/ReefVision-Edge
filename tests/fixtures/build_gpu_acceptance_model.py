from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_gpu_acceptance_model(path: Path) -> Path:
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    output_info = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 6, 2]
    )
    base = np.array(
        [
            [32.0, 20.0],
            [32.0, 28.0],
            [16.0, 10.0],
            [8.0, 8.0],
            [0.9, 0.2],
            [0.1, 0.8],
        ],
        dtype=np.float32,
    ).reshape(1, 12)
    weights = np.zeros((1, 12), dtype=np.float32)
    weights[0, 8] = 0.01
    weights[0, 11] = 0.01
    nodes = [
        helper.make_node(
            "ReduceMean",
            ["images"],
            ["mean"],
            axes=[1, 2, 3],
            keepdims=0,
        ),
        helper.make_node("Unsqueeze", ["mean", "axes"], ["mean_2d"]),
        helper.make_node("MatMul", ["mean_2d", "weights"], ["delta"]),
        helper.make_node("Add", ["base", "delta"], ["flat_output"]),
        helper.make_node("Reshape", ["flat_output", "shape"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "reefvision-edge-gpu-acceptance",
        [input_info],
        [output_info],
        initializer=[
            numpy_helper.from_array(base, "base"),
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.array([1, 6, 2], dtype=np.int64), "shape"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="ReefVision-Edge GPU acceptance",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path
