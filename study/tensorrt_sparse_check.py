"""Benchmark and compare the real 2:4 sparse TensorRT engine."""

import ctypes
import json
import os
import statistics
import time

TRT_LIBS = os.environ["TRT_LIBS"]
ORT_CAPI = os.environ["ORT_CAPI"]
os.environ["PATH"] = os.pathsep.join([TRT_LIBS, ORT_CAPI, os.environ["PATH"]])
os.add_dll_directory(TRT_LIBS); os.add_dll_directory(ORT_CAPI)
for dll in ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll"):
    ctypes.WinDLL(os.path.join(TRT_LIBS, dll))

import numpy as np
import onnxruntime as ort

ort.preload_dlls(); ort.set_default_logger_severity(3)
root = os.path.abspath(".cache/huggingface/hub/kucukkanat--LFM2.5-Encoder-350M-Prompt-Router-ONNX/onnx")
base = os.path.join(root, "model_fp32_trt.onnx")
sparse = os.path.join(root, "model_fp32_trt_2of4.onnx")
cache = os.path.abspath(".cache/perf/trt_study/sparse_2of4")
os.makedirs(cache, exist_ok=True)
ids = np.array([[1, 544, 32830, 1334, 522, 23810, 6690, 1517, 708, 522, 17013,
                 4862, 708, 522, 13663, 8253, 509, 6570, 1334, 6478, 768,
                 42236, 875, 3676, 5112, 523]], dtype=np.int64)
feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def session(path, sparse_enable=False):
    options = {"trt_fp16_enable": True, "trt_cuda_graph_enable": True,
               "trt_sparsity_enable": sparse_enable, "trt_min_subgraph_size": 1,
               "trt_builder_optimization_level": 1, "trt_engine_cache_enable": True,
               "trt_engine_cache_path": cache + ("_sparse" if sparse_enable else "_base"),
               "trt_timing_cache_enable": True, "trt_timing_cache_path": cache}
    return ort.InferenceSession(path, providers=[("TensorrtExecutionProvider", options), "CPUExecutionProvider"])


# Do not build the dense and sparse graphs in the same ORT process: TRT's
# in-memory lookup keys this identical topology before considering changed
# initializers. CPU provides an independent fp32 reference instead.
reference = ort.InferenceSession(base, providers=["CPUExecutionProvider"]).run(None, feed)
cache = os.path.abspath(".cache/perf/trt_study/sparse_2of4_independent")
sparse_session = session(sparse, True)
output = sparse_session.run(None, feed)
cosines = []
max_abs = 0.0
max_relative = 0.0
for expected, actual in zip(reference, output):
    a, b = expected[0].sum(axis=0).astype(np.float64), actual[0].sum(axis=0).astype(np.float64)
    cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    delta = np.abs(expected - actual)
    max_abs = max(max_abs, float(delta.max()))
    max_relative = max(max_relative, float(delta.max() / max(np.abs(expected).max(), 1e-12)))
for _ in range(10): sparse_session.run(None, feed)
samples = []
for _ in range(50):
    start = time.perf_counter(); sparse_session.run(None, feed)
    samples.append((time.perf_counter() - start) * 1000)
samples.sort()
engine = next(path for path in os.listdir(cache + "_sparse") if path.endswith(".engine"))
print(json.dumps({"p50_ms": round(statistics.median(samples), 3), "p95_ms": round(samples[47], 3),
                  "cosine": round(min(cosines), 9), "max_abs": round(max_abs, 6),
                  "max_relative": round(max_relative, 6),
                  "engine_mb": round(os.path.getsize(os.path.join(cache + "_sparse", engine)) / 1048576, 1)}))
