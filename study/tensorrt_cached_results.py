"""Measure already-built fp32 TensorRT study engines without rebuilding them."""

import ctypes
import json
import os
import statistics
import time

TRT_LIBS = os.environ["TRT_LIBS"]
ORT_CAPI = os.environ["ORT_CAPI"]
ROOT = os.path.abspath(".cache/huggingface/hub/kucukkanat--LFM2.5-Encoder-350M-Prompt-Router-ONNX/onnx")
CACHE_ROOT = os.path.abspath(".cache/perf/trt_study")
os.environ["PATH"] = os.pathsep.join([TRT_LIBS, ORT_CAPI, os.environ["PATH"]])
os.add_dll_directory(TRT_LIBS)
os.add_dll_directory(ORT_CAPI)
for dll in ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll"):
    ctypes.WinDLL(os.path.join(TRT_LIBS, dll))

import numpy as np
import onnxruntime as ort

ort.preload_dlls()
ort.set_default_logger_severity(3)
ids = np.array([[1, 544, 32830, 1334, 522, 23810, 6690, 1517, 708, 522, 17013,
                 4862, 708, 522, 13663, 8253, 509, 6570, 1334, 6478, 768,
                 42236, 875, 3676, 5112, 523]], dtype=np.int64)
feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
reference = ort.InferenceSession(os.path.join(ROOT, "model_fp32_trt.onnx"), providers=["CPUExecutionProvider"]).run(None, feed)


def bench(name, fp16, graph):
    cache = os.path.join(CACHE_ROOT, name)
    options = {
        "device_id": 0, "trt_max_workspace_size": 1024 * 1024 * 1024,
        "trt_min_subgraph_size": 1, "trt_builder_optimization_level": 1,
        "trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
        "trt_fp16_enable": fp16, "trt_cuda_graph_enable": graph,
    }
    start = time.perf_counter()
    session = ort.InferenceSession(os.path.join(ROOT, "model_fp32_trt.onnx"),
                                   providers=[("TensorrtExecutionProvider", options), "CPUExecutionProvider"])
    load_ms = (time.perf_counter() - start) * 1000
    output = session.run(None, feed)
    cosine = min(float(a.ravel().astype(np.float64) @ b.ravel().astype(np.float64) /
                 (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))
                 for a, b in zip(reference, output))
    for _ in range(10): session.run(None, feed)
    samples = []
    for _ in range(100):
        start = time.perf_counter(); session.run(None, feed)
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    row = {"name": name, "load_ms": round(load_ms, 1), "p50_ms": round(statistics.median(samples), 3),
           "p95_ms": round(samples[94], 3), "p99_ms": round(samples[98], 3),
           "cosine_cpu_reference": round(cosine, 9)}
    print(json.dumps(row), flush=True)


bench("fp32", False, False)
bench("fp32_graph", False, True)
bench("fp32_engine_fp16", True, False)
bench("fp32_engine_fp16_graph", True, True)
