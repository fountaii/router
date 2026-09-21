"""Injeta o modulo destilado (study/distill_layer.py) no modelo PyTorch
completo, no lugar do self_attn original, e roda o mesmo harness de rota
(route()) usado em study/eval_routing.py -- so que aqui e no PyTorch
diretamente, contra o mesmo conjunto de 26 casos (16 faceis + 10 dificeis).
"""

import argparse
import math
import types

import torch
import transformers

MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"

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
    ("Reset my password, I can't log in.", ["Account security", "Account billing"], 0),
    ("Why was I charged twice this month?", ["Account security", "Account billing"], 1),
    ("Schedule a meeting with the design team for Thursday.", ["Calendar management", "Task management"], 0),
    ("Add 'review the Q3 report' to my to-do list.", ["Calendar management", "Task management"], 1),
    ("What's 15% of 240?", ["Arithmetic", "Unit conversion"], 0),
    ("How many kilometers is 50 miles?", ["Arithmetic", "Unit conversion"], 1),
    ("Explain how photosynthesis works.", ["Science explanation", "History explanation"], 0),
    ("Explain the causes of World War I.", ["Science explanation", "History explanation"], 1),
    ("Summarize the key points of this contract.", ["Document summarization", "Document translation"], 0),
    ("Translate this contract into Spanish.", ["Document summarization", "Document translation"], 1),
]


def build_prefix(cats):
    return "Categories:\n" + "\n".join(f"- {c}" for c in cats) + "\n\nText:\n"


def cat_ranges(cats):
    ranges, pos = [], len("Categories:\n")
    for cat in cats:
        start = pos + 2
        ranges.append((start, start + len(cat)))
        pos = start + len(cat) + 1
    return ranges


def pool_normalized(indices, proj):
    out = proj[indices].mean(dim=0)
    norm = out.norm()
    return out / norm if norm > 0 else out


def route(model, tok, head_scale, head_bias, text, cats):
    prefix = build_prefix(cats)
    enc = tok(prefix + text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        hidden = model.lfm2(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[0]
        token_proj = model.tok_proj(hidden)
        rule_proj = model.rule_proj(hidden)

    text_indices = [i for i, (a, b) in enumerate(offsets) if b > len(prefix) and a != b]
    query = pool_normalized(text_indices, token_proj)

    logits = []
    for start, end in cat_ranges(cats):
        cat_indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and a != b]
        lane = pool_normalized(cat_indices, rule_proj)
        cosine = float((lane @ query).item())
        logits.append(cosine * head_scale + head_bias)
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    total = sum(ex)
    probs = [x / total for x in ex]
    return probs, probs.index(max(probs))


def evaluate(model, tok, head_scale, head_bias, label, reference=None):
    correct, results, cosines = 0, [], []
    for text, cats, expected in CASES:
        probs, top = route(model, tok, head_scale, head_bias, text, cats)
        correct += int(top == expected)
        results.append(probs)
    if reference is not None:
        for a, b in zip(reference, results):
            a_t, b_t = torch.tensor(a), torch.tensor(b)
            cosines.append(float((a_t @ b_t / (a_t.norm() * b_t.norm() + 1e-12)).item()))
    print(f"{label:<40} top1={correct}/{len(CASES)}"
          + (f"  cosseno_medio={sum(cosines)/len(cosines):.6f}  cosseno_min={min(cosines):.6f}" if cosines else ""))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layers", type=int, nargs="+")
    ap.add_argument("--checkpoints", type=str, nargs="+", required=True)
    ap.add_argument("--module", choices=["conv", "rope_free_attn"], default="conv")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    # scale/bias do head cosseno -- vem do config.json do pacote ONNX do
    # router (kucukkanat/LFM2.5-Encoder-350M-Prompt-Router-ONNX), nao existe
    # no checkpoint base do PyTorch.
    head_scale = 1.3714938163757324
    head_bias = -0.2723352313041687

    reference = evaluate(model, tok, head_scale, head_bias, "original (referencia)")

    donor_conv = None
    for l in model.lfm2.layers:
        if not l.is_attention_layer:
            donor_conv = l.conv
            break

    for layer_idx, ckpt in zip(args.layers, args.checkpoints):
        target = model.lfm2.layers[layer_idx]
        if args.module == "conv":
            student = type(donor_conv)(model.config, layer_idx=layer_idx)
            student.load_state_dict(torch.load(ckpt, map_location="cpu"))
            student.eval()
            target.conv = student
            target.is_attention_layer = False
            del target.self_attn
        else:
            from rope_free_attn import RopeFreeAttention
            student = RopeFreeAttention(model.config, layer_idx=layer_idx)
            student.load_state_dict(torch.load(ckpt, map_location="cpu"))
            student.eval()
            target.self_attn = student

    label = f"camadas {args.layers}: attn -> conv destilada"
    evaluate(model, tok, head_scale, head_bias, label, reference)


if __name__ == "__main__":
    main()
