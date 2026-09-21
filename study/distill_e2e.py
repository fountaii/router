"""Destilacao fim-a-fim: em vez de MSE no estado oculto intermediario (que
convergia bem localmente mas nao segurava depois de mais 8 camadas -- 3 delas
ainda atencao de verdade, sensivel a conteudo, que amplificam erro pequeno),
otimiza direto contra a saida final do router: os embeddings normalizados
(token_proj/rule_proj apos pooling) que alimentam o head de cosseno. O
gradiente flui pela rede inteira, entao os estudantes aprendem a compensar o
que realmente importa pra decisao, nao um proxy que nao segura.

Usa o MESMO formato de entrada do router de producao (Categories:\\n-
...\\n\\nText:\\n<texto>) com categorias sinteticas variadas, e MSE nos
embeddings normalizados de query + cada categoria (mais estavel pra treinar
que KL direto no softmax de poucas classes).

Uso: python study/distill_e2e.py 2 5 8 --epochs 60 --n-samples 300 --warm-start ...
"""

import argparse
import random
import time

import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator

from distill_layer import CATEGORIES, TEXTS, MODEL_ID
from rope_free_attn import RopeFreeAttention


def build_prefix(cats: list[str]) -> str:
    return "Categories:\n" + "\n".join(f"- {c}" for c in cats) + "\n\nText:\n"


def cat_ranges(cats: list[str]) -> list[tuple[int, int]]:
    ranges, pos = [], len("Categories:\n")
    for cat in cats:
        start = pos + 2
        ranges.append((start, start + len(cat)))
        pos = start + len(cat) + 1
    return ranges


def pool_normalized_batch(weights: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    # weights: (seq,) 0/1 mask ja normalizado (soma 1); proj: (seq, dim)
    out = weights @ proj
    return out / (out.norm() + 1e-12)


def build_sample(text: str, cats: list[str], tok):
    prefix = build_prefix(cats)
    enc = tok(prefix + text, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    ids = torch.tensor([enc["input_ids"]])
    mask = torch.ones_like(ids)

    text_idx = [i for i, (a, b) in enumerate(offsets) if b > len(prefix) and a != b]
    text_w = torch.zeros(len(offsets))
    text_w[text_idx] = 1.0 / len(text_idx)

    cat_ws = []
    for start, end in cat_ranges(cats):
        idx = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and a != b]
        w = torch.zeros(len(offsets))
        w[idx] = 1.0 / len(idx)
        cat_ws.append(w)

    return ids, mask, text_w, torch.stack(cat_ws)


def forward_embeddings(model, ids, mask, text_w, cat_w, device):
    hidden = model.lfm2(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state[0]
    token_proj = model.tok_proj(hidden)
    rule_proj = model.rule_proj(hidden)
    query = pool_normalized_batch(text_w.to(device), token_proj)
    lanes = torch.stack([pool_normalized_batch(w.to(device), rule_proj) for w in cat_w])
    return query, lanes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layers", type=int, nargs="+")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--warm-start", type=str, nargs="*", default=None,
                    help="checkpoints ja treinados (distill_joint.py) pra iniciar cada camada -- so pra --module conv")
    ap.add_argument("--patience", type=int, default=6,
                    help="epocas sem melhora na validacao antes de parar cedo")
    ap.add_argument("--module", choices=["conv", "rope_free_attn"], default="conv",
                    help="conv: gated short conv (mais barato, warm-start emprestado). "
                         "rope_free_attn: mesma atencao original sem RoPE (warm-start = pesos originais da propria camada)")
    args = ap.parse_args()
    layers_sorted = sorted(args.layers)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print(f"device: {device}  mixed_precision: {accelerator.mixed_precision}", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.float32).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.lfm2.layers
    donor_conv_type = None
    for l in layers:
        if not l.is_attention_layer:
            donor_conv_type = type(l.conv)
            break

    rng = random.Random(0)
    all_samples = []
    for _ in range(args.n_samples):
        n_cats = rng.randint(2, 5)
        cats = rng.sample(CATEGORIES, n_cats)
        text = rng.choice(TEXTS)
        all_samples.append(build_sample(text, cats, tok))
    rng.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * 0.15))
    val_samples, samples = all_samples[:n_val], all_samples[n_val:]
    print(f"dataset sintetico: {len(samples)} treino + {len(val_samples)} validacao "
          f"(formato real do router)", flush=True)

    # alvos do professor original (congelado, forward completo) -- treino e
    # validacao, capturados juntos antes de trocar os modulos.
    print("capturando alvos do professor...", flush=True)
    def capture_targets(dataset):
        out = []
        with torch.no_grad():
            for ids, mask, text_w, cat_w in dataset:
                query, lanes = forward_embeddings(model, ids, mask, text_w, cat_w, device)
                out.append((query.clone(), lanes.clone()))
        return out
    teacher_targets = capture_targets(samples)
    val_targets = capture_targets(val_samples)
    print(f"  alvos capturados: {len(teacher_targets)} treino, {len(val_targets)} validacao", flush=True)

    # troca os self_attn pelos estudantes
    students = {}
    for i, layer_idx in enumerate(layers_sorted):
        if args.module == "conv":
            student = donor_conv_type(model.config, layer_idx=layer_idx).to(device)
            student.load_state_dict(torch.load(args.warm_start[i], map_location=device))
            layers[layer_idx].conv = student
            layers[layer_idx].is_attention_layer = False
            del layers[layer_idx].self_attn
        else:  # rope_free_attn -- warm-start = pesos originais da propria camada
            original = layers[layer_idx].self_attn
            student = RopeFreeAttention(model.config, layer_idx=layer_idx).to(device)
            student.load_state_dict(original.state_dict())
            layers[layer_idx].self_attn = student
        for p in student.parameters():
            p.requires_grad_(True)
        students[layer_idx] = student

    params = [p for s in students.values() for p in s.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.MSELoss()
    model, opt = accelerator.prepare(model, opt)

    def val_loss():
        total = 0.0
        with torch.no_grad():
            for (ids, mask, text_w, cat_w), (t_query, t_lanes) in zip(val_samples, val_targets):
                query, lanes = forward_embeddings(model, ids, mask, text_w, cat_w, device)
                total += (loss_fn(query.float(), t_query.float()) + loss_fn(lanes.float(), t_lanes.float())).item()
        return total / len(val_samples)

    print("treinando fim-a-fim (com early stopping na validacao)...", flush=True)
    t_start = time.time()
    best_val, best_epoch, patience, bad_epochs = float("inf"), -1, args.patience, 0
    for epoch in range(args.epochs):
        order = list(range(len(samples)))
        random.shuffle(order)
        total = 0.0
        for idx in order:
            ids, mask, text_w, cat_w = samples[idx]
            t_query, t_lanes = teacher_targets[idx]
            opt.zero_grad()
            with accelerator.autocast():
                query, lanes = forward_embeddings(model, ids, mask, text_w, cat_w, device)
                loss = loss_fn(query.float(), t_query.float()) + loss_fn(lanes.float(), t_lanes.float())
            accelerator.backward(loss)
            opt.step()
            total += loss.item()
        vloss = val_loss()
        elapsed = time.time() - t_start
        marker = ""
        if vloss < best_val:
            best_val, best_epoch, bad_epochs = vloss, epoch, 0
            marker = "  <- melhor na validacao ate agora"
        else:
            bad_epochs += 1
        # salva TODA epoca (nao so a "melhor" pela validacao sintetica -- ela
        # bateu mal com os 26 casos reais da ultima vez). Escolha final e
        # feita depois validando cada uma contra study/eval_distilled.py.
        for layer_idx, student in students.items():
            torch.save(accelerator.unwrap_model(student).state_dict(),
                      f".cache/perf/e2e_layer{layer_idx}_epoch{epoch}.pt")
        print(f"epoch {epoch:3d}  loss treino {total / len(samples):.6f}  "
              f"loss val {vloss:.6f}  {elapsed:.0f}s decorridos{marker}", flush=True)
        if bad_epochs >= patience:
            print(f"parando cedo: {patience} epocas sem melhora na validacao "
                  f"(melhor foi a epoca {best_epoch}, loss val {best_val:.6f})", flush=True)
            break

    print(f"treino completo -- melhor checkpoint: epoca {best_epoch}, loss val {best_val:.6f}, "
          f"salvo em .cache/perf/e2e_layer*.pt", flush=True)


if __name__ == "__main__":
    main()
