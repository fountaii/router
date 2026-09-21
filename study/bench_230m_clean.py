"""Mede fp32/int8/int4 do 230M sem o torch no processo (o pool OMP dele competia).

Duas variantes de int4 para separar as causas: uma so com MatMul quantizado e
outra com a embedding tambem, que e o que deixa o arquivo menor que o int8.
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

OUT = Path(".cache/perf/lfm230")
ROUTER = ".cache/huggingface/hub/kucukkanat--LFM2.5-Encoder-350M-Prompt-Router-ONNX/onnx"

opts = ort.SessionOptions()
opts.intra_op_num_threads = 6
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

rng = np.random.default_rng(0)
cases = {n: rng.integers(1, 30000, size=(1, n)).astype(np.int64) for n in (19, 26, 54)}

models = [
    ("router 350M int8 (atual)", f"{ROUTER}/model_quantized_convfloat.onnx"),
    ("230M int8", OUT / "model_int8.onnx"),
    ("230M int4 (so MatMul)", OUT / "model_int4_matmul.onnx"),
    ("230M int4 (MatMul+embed)", OUT / "model_int4.onnx"),
    ("230M fp32", OUT / "model_fp32.onnx"),
]
models = [(n, str(p)) for n, p in models if Path(p).exists()]

times: dict[str, dict[int, list[float]]] = {}
for rounds in range(3):
    for name, path in models:
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        times.setdefault(name, {n: [] for n in cases})
        for n, ids in cases.items():
            feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
            for _ in range(5):
                sess.run(None, feed)
            for _ in range(25):
                t = time.perf_counter()
                sess.run(None, feed)
                times[name][n].append((time.perf_counter() - t) * 1000)
        del sess

print("latencia p50, 6 threads, rodadas intercaladas\n")
for name, path in models:
    mb = Path(path).stat().st_size / 1048576
    row = "  ".join(f"{n}tok {np.median(times[name][n]):7.2f}ms" for n in cases)
    print(f"  {name:26s} {mb:6.0f}MB  {row}")
