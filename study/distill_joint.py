"""Destilacao conjunta: treina os N modulos conv substitutos ao MESMO tempo,
dentro do modelo hibrido de verdade (nao snapshots isolados do professor
original). Cada estudante recebe a entrada real que vai chegar nele em
producao -- incluindo o erro propagado pelos outros estudantes upstream -- e
o alvo continua sendo a saida do professor original (congelado) na mesma
posicao. Corrige o efeito de composicao que a versao anterior (treino por
camada isolada) nao capturava: sozinhas, as 3 camadas criticas do LFM2.5
Router davam cosseno ~1.0; juntas sem esse ajuste, cosseno caia pra 0.13.

Rapido de verdade (accelerate + batching + corte do forward apos a ultima
camada alvo, ja que layers depois dela nao entram na loss):
- mixed precision (bf16) via accelerate
- lotes de --batch-size amostras por passo, com padding + attention_mask
- forward interrompido logo apos a ultima camada alvo (hook levanta uma
  excecao sentinela) -- nao roda as ~8 camadas seguintes a toa

Uso: python study/distill_joint.py 2 5 8 --epochs 40 --n-samples 400 --batch-size 16
"""

import argparse
import random
import time

import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator

from distill_layer import MODEL_ID, synthetic_dataset


class _StopForward(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layers", type=int, nargs="+")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warm-start", type=str, nargs="*", default=None,
                    help="checkpoints de study/distill_layer.py pra iniciar cada camada")
    args = ap.parse_args()
    layers_sorted = sorted(args.layers)
    last_layer = layers_sorted[-1]

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

    # 1) monta os lotes (padded) do dataset sintetico
    texts = synthetic_dataset(args.n_samples)
    print(f"dataset sintetico: {len(texts)} amostras, camadas: {args.layers}", flush=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    batches = []
    for i in range(0, len(texts), args.batch_size):
        chunk = texts[i:i + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        batches.append(enc)

    # 2) captura os alvos do professor (congelado, forward completo cortado
    # logo apos a ultima camada alvo -- as posteriores nao afetam a loss)
    teacher_targets_by_batch = []
    captured = {}
    def make_capture_hook(layer_idx, stop_after):
        def hook(_module, hargs, hkwargs, output):
            captured[layer_idx] = (output[0] if isinstance(output, tuple) else output).detach()
            if stop_after:
                raise _StopForward()
        return hook

    handles = [
        layers[l].self_attn.register_forward_hook(
            make_capture_hook(l, l == last_layer), with_kwargs=True
        )
        for l in layers_sorted
    ]
    with torch.no_grad():
        for bi, enc in enumerate(batches):
            captured.clear()
            try:
                model.lfm2(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            except _StopForward:
                pass
            teacher_targets_by_batch.append({l: captured[l].clone() for l in layers_sorted})
            if (bi + 1) % 5 == 0:
                print(f"  alvos capturados: lote {bi + 1}/{len(batches)}", flush=True)
    for h in handles:
        h.remove()

    # 3) troca os self_attn pelos estudantes
    students = {}
    for layer_idx in layers_sorted:
        student = donor_conv_type(model.config, layer_idx=layer_idx).to(device)
        if args.warm_start:
            ckpt_path = args.warm_start[layers_sorted.index(layer_idx)]
            student.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            donor_idx = min((l for l in range(len(layers)) if not layers[l].is_attention_layer),
                            key=lambda l: abs(l - layer_idx))
            student.load_state_dict(layers[donor_idx].conv.state_dict())
        for p in student.parameters():
            p.requires_grad_(True)
        layers[layer_idx].conv = student
        layers[layer_idx].is_attention_layer = False
        del layers[layer_idx].self_attn
        students[layer_idx] = student

    student_outputs = {}
    def make_student_hook(layer_idx, stop_after):
        def hook(_module, hargs, hkwargs, output):
            student_outputs[layer_idx] = output
            if stop_after:
                raise _StopForward()
        return hook
    student_handles = [
        students[l].register_forward_hook(make_student_hook(l, l == last_layer), with_kwargs=True)
        for l in layers_sorted
    ]

    params = [p for s in students.values() for p in s.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.MSELoss()
    model, opt = accelerator.prepare(model, opt)

    print("treinando (conjunto, com accelerate)...", flush=True)
    t_start = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(batches)))
        random.shuffle(order)
        total = 0.0
        for step, bi in enumerate(order):
            enc = batches[bi]
            targets = teacher_targets_by_batch[bi]
            opt.zero_grad()
            student_outputs.clear()
            with accelerator.autocast():
                try:
                    model.lfm2(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                except _StopForward:
                    pass
                loss = sum(
                    loss_fn(student_outputs[l].float(), targets[l].float())
                    for l in layers_sorted
                )
            accelerator.backward(loss)
            opt.step()
            total += loss.item()
        elapsed = time.time() - t_start
        print(f"epoch {epoch:3d} COMPLETA  loss medio {total / len(batches):.6f}  {elapsed:.0f}s decorridos", flush=True)
        for layer_idx, student in students.items():
            torch.save(accelerator.unwrap_model(student).state_dict(), f".cache/perf/joint_layer{layer_idx}.pt")

    for h in student_handles:
        h.remove()
    print("treino completo, checkpoints finais salvos em .cache/perf/joint_layer*.pt", flush=True)


if __name__ == "__main__":
    main()
