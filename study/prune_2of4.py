"""Make TensorRT-compatible 2:4 sparse MatMul weights and report the exact rate."""

import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

source, destination = sys.argv[1:3]
model = onnx.load(source)
changed = 0
zeros_before = 0
zeros_after = 0
elements = 0

for index, initializer in enumerate(model.graph.initializer):
    if initializer.data_type != onnx.TensorProto.FLOAT or len(initializer.dims) != 2:
        continue
    array = numpy_helper.to_array(initializer)
    # TensorRT requires groups along K. For ONNX MatMul weights [K, N], every
    # column therefore retains the two largest magnitudes from each four rows.
    if array.shape[0] < 4 or array.shape[0] % 4:
        continue
    values = np.array(array, copy=True)
    groups = values.reshape(values.shape[0] // 4, 4, values.shape[1])
    magnitudes = np.abs(groups)
    remove = np.argpartition(magnitudes, 2, axis=1)[:, :2, :]
    mask = np.ones(groups.shape, dtype=bool)
    np.put_along_axis(mask, remove, False, axis=1)
    zeros_before += int(np.count_nonzero(groups == 0))
    elements += groups.size
    groups[~mask] = 0
    zeros_after += int(np.count_nonzero(groups == 0))
    model.graph.initializer[index].CopyFrom(numpy_helper.from_array(values, initializer.name))
    changed += 1

if not changed:
    raise RuntimeError("No compatible float32 2D MatMul initializers were found.")

onnx.checker.check_model(model, full_check=False)
onnx.save(model, destination, save_as_external_data=False)
print(f"pruned={changed} zero_fraction={zeros_after / elements:.4f}")
print(f"output={os.path.abspath(destination)}")
