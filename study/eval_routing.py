"""Reimplementa route() (buildInputs + pool + cosine head) em Python pra
validar variantes cirurgicas do modelo contra a logica real do router.ts,
nao so cosseno cru de embedding. Roda so na CPU/fp32 -- e checagem de
qualidade, a latencia final se mede depois via CUDA graph (bench_variant.py).
"""

import json
import math
import os
import sys

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
from tokenizers import Tokenizer

ROOT = os.path.abspath(".cache/huggingface/hub/kucukkanat--LFM2.5-Encoder-350M-Prompt-Router-ONNX")
TOK = Tokenizer.from_file(os.path.join(ROOT, "tokenizer.json"))
HEAD = json.load(open(os.path.join(ROOT, "config.json")))["head"]

# mesmo formato de src/router.ts: buildPrefix/catRanges
def build_prefix(cats):
    return "Categories:\n" + "\n".join(f"- {c}" for c in cats) + "\n\nText:\n"

def cat_ranges(cats):
    ranges = []
    pos = len("Categories:\n")
    for cat in cats:
        start = pos + 2
        ranges.append((start, start + len(cat)))
        pos = start + len(cat) + 1
    return ranges

def pool_normalized(indices, proj):
    # proj: (seq, dim)
    out = proj[indices].mean(axis=0)
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out

def route(session, text, cats):
    prefix = build_prefix(cats)
    encoding = TOK.encode(prefix + text)
    ids = encoding.ids
    offsets = encoding.offsets

    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    outputs = session.run(["token_proj", "rule_proj"], {"input_ids": input_ids, "attention_mask": attention_mask})
    token_proj, rule_proj = outputs[0][0], outputs[1][0]  # (seq, proj_dim)

    text_indices = [i for i, (a, b) in enumerate(offsets) if b > len(prefix) and a != b]
    query = pool_normalized(text_indices, token_proj)

    ranges = cat_ranges(cats)
    logits = []
    for start, end in ranges:
        cat_indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and a != b]
        lane = pool_normalized(cat_indices, rule_proj)
        cosine = float(lane @ query)
        logits.append(cosine * HEAD["scale"] + HEAD["bias"])

    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    total = sum(ex)
    probs = [x / total for x in ex]
    return probs, int(np.argmax(probs))


# (texto, categorias, indice esperado) -- casos de bom senso, cobrindo dominios
# variados, pra pegar degradacao de ranking que o cosseno cru do embedding
# poderia esconder.
CASES = [
    ("Set a timer for 5 minutes.", ["Simple tool use", "Creative writing"], 0),
    ("Write a short poem about the ocean at night.", ["Simple tool use", "Creative writing"], 1),
    ("What's the weather in Tokyo right now?", ["Weather lookup", "Code generation"], 0),
    ("Write a Python function that sorts a list.", ["Weather lookup", "Code generation"], 1),
    ("Translate 'good morning' into French.", ["Translation", "Math"], 0),
    ("What is the derivative of x^2 + 3x?", ["Translation", "Math"], 1),
    ("Book me a table for two tonight at 8pm.", ["Restaurant reservation", "Customer support"], 0),
    ("My order hasn't arrived and it's been two weeks.", ["Restaurant reservation", "Customer support"], 1),
    ("Summarize this article in three bullet points.", ["Summarization", "Image generation"], 0),
    ("Generate an image of a cat riding a bicycle.", ["Summarization", "Image generation"], 1),
    ("Cancel my subscription immediately.", ["Account management", "Small talk"], 0),
    ("How's your day going?", ["Account management", "Small talk"], 1),
    ("Convert 100 fahrenheit to celsius.", ["Unit conversion", "Storytelling"], 0),
    ("Tell me a bedtime story about a dragon.", ["Unit conversion", "Storytelling"], 1),
    ("Find flights from Sao Paulo to Lisbon next Tuesday.", ["Flight search", "Recipe suggestion"], 0),
    ("Give me a recipe for chocolate chip cookies.", ["Flight search", "Recipe suggestion"], 1),
]


def evaluate(model_path, label, reference_probs=None):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    correct = 0
    cosines = []
    results = []
    for text, cats, expected in CASES:
        probs, top = route(session, text, cats)
        correct += int(top == expected)
        results.append(probs)
    if reference_probs is not None:
        for a, b in zip(reference_probs, results):
            a, b = np.array(a), np.array(b)
            cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    print(json.dumps({
        "name": label,
        "top1_accuracy": f"{correct}/{len(CASES)}",
        "probs_cosine_vs_reference": round(float(np.mean(cosines)), 6) if cosines else None,
        "probs_cosine_min": round(float(np.min(cosines)), 6) if cosines else None,
    }))
    return results


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("uso: python study/eval_routing.py <modelo1.onnx> [modelo2.onnx ...]")
    reference = None
    for i, path in enumerate(paths):
        label = os.path.basename(path)
        results = evaluate(path, label, reference_probs=reference)
        if i == 0:
            reference = results
