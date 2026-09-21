"""Peso-somente int8/int4 (MatMulNBits) sobre o bucket fp16, medido no CUDA EP.

Os 2.2 ms fixos do bucket sao 548 MB de peso de MatMul lidos por forward; este
script corta os bytes e mede latencia e cosseno contra a referencia fp32 na CPU.
"""

import json
import os
import statistics
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

ort.preload_dlls()
ort.set_default_logger_severity(3)
ROOT = os.path.abspath(".cache/huggingface/hub/kucukkanat--LFM2.5-Encoder-350M-Prompt-Router-ONNX/onnx")
OUT = os.path.abspath(".cache/perf")
BUCKET = 32
SOURCE = os.path.join(ROOT, f"model_fp16_fixed{BUCKET}.onnx")

ids = np.zeros((1, BUCKET), dtype=np.int64); ids[0, :19] = 1
mask = np.zeros((1, BUCKET), dtype=np.int64); mask[0, :19] = 1
feed = {"input_ids": ids, "attention_mask": mask}

reference = ort.InferenceSession(SOURCE, providers=["CPUExecutionProvider"]).run(None, feed)


def measure(path, label):
    session = ort.InferenceSession(path, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": True})])
    output = session.run(None, feed)
    cosine = min(float(a.ravel().astype(np.float64) @ b.ravel().astype(np.float64) /
                 (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))
                 for a, b in zip(reference, output))
    for _ in range(50):
        session.run(None, feed)
    samples = []
    for _ in range(300):
        start = time.perf_counter(); session.run(None, feed)
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    print(json.dumps({"name": label, "mb": round(os.path.getsize(path) / 1048576),
                      "p50_ms": round(statistics.median(samples), 3),
                      "p95_ms": round(samples[284], 3),
                      "cosine": round(cosine, 6)}), flush=True)


measure(SOURCE, "fp16 (atual)")

for bits, block in ((8, 128), (4, 128), (4, 32)):
    path = os.path.join(OUT, f"nbits{bits}_b{block}_fixed{BUCKET}.onnx")
    if not os.path.exists(path):
        quantizer = MatMulNBitsQuantizer(
            onnx.load(SOURCE),
            algo_config=DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=bits),
        )
        quantizer.process()
        quantizer.model.save_model_to_file(path, use_external_data_format=False)
    measure(path, f"int{bits} weight-only bloco {block}")
