"""Compare the shipped fp16 CUDA-graph bucket against an fp32 CPU reference.

The TensorRT study found trt_fp16_enable collapses the router (cosine 0.141),
so the bucket we actually ship needs the same check.
"""

import json
import os

import numpy as np
import onnxruntime as ort

ort.preload_dlls()
ort.set_default_logger_severity(3)
ROOT = os.path.abspath(".cache/huggingface/hub/kucukkanat--LFM2.5-Encoder-350M-Prompt-Router-ONNX/onnx")
BUCKET = 32
ids = np.array([[1, 544, 32830, 1334, 522, 23810, 6690, 1517, 708, 522, 17013,
                 4862, 708, 522, 13663, 8253, 509, 6570, 1334]], dtype=np.int64)

padded = np.zeros((1, BUCKET), dtype=np.int64)
padded[0, :ids.shape[1]] = ids
mask = np.zeros((1, BUCKET), dtype=np.int64)
mask[0, :ids.shape[1]] = 1

reference = ort.InferenceSession(os.path.join(ROOT, "model_fp32.onnx"), providers=["CPUExecutionProvider"]).run(
    None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
output = ort.InferenceSession(os.path.join(ROOT, f"model_fp16_fixed{BUCKET}.onnx"),
                              providers=["CUDAExecutionProvider"]).run(
    None, {"input_ids": padded, "attention_mask": mask})

rows = []
for expected, actual in zip(reference, output):
    actual = actual[:, :ids.shape[1]] if actual.ndim == 3 and actual.shape[1] == BUCKET else actual
    a, b = expected.ravel().astype(np.float64), actual.ravel().astype(np.float64)
    rows.append({"shape": list(expected.shape),
                 "cosine": round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 9),
                 "max_abs": round(float(np.abs(expected - actual).max()), 6)})
print(json.dumps(rows))
