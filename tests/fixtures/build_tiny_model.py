from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_tiny_yolo_model(path: Path, class_count: int = 2) -> Path:
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 640, 640]
    )
    output_info = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 4 + class_count, 2]
    )
    values = np.zeros((1, 4 + class_count, 2), dtype=np.float32)
    values[0, 0:4, 0] = [320.0, 320.0, 160.0, 80.0]
    values[0, 0:4, 1] = [100.0, 200.0, 50.0, 50.0]
    if class_count >= 1:
        values[0, 4, :] = [0.9, 0.2]
    if class_count >= 2:
        values[0, 5, :] = [0.1, 0.8]
    constant = helper.make_node(
        "Constant", inputs=[], outputs=["output0"], value=numpy_helper.from_array(values)
    )
    graph = helper.make_graph([constant], "tiny-yolo", [input_info], [output_info])
    model = helper.make_model(
        graph,
        producer_name="ReefVision-Edge tests",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path
