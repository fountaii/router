"""Aplica no ONNX o resultado da destilacao fim-a-fim (study/distill_e2e.py
--module rope_free_attn): substitui os pesos de Q/K/V/O/QK-Norm de uma camada
pelos do checkpoint treinado, e remove SO os nos de RoPE do grafo -- religa
Q e K (ja normalizados pelo QK-Norm) direto pro scaling/QK^T, sem tocar em
GQA, softmax, AV ou no resto da camada. Limpeza de nos orfaos e automatica
(dead-code elimination local, nao precisa listar nome por nome).

Uso: python study/export_rope_free.py <origem.onnx> <saida.onnx> \
       --layer 2 --checkpoint .cache/perf/e2e_layer2_epoch0.pt \
       --layer 5 --checkpoint .cache/perf/e2e_layer5_epoch0.pt \
       --layer 8 --checkpoint .cache/perf/e2e_layer8_epoch0.pt
"""

import argparse

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def load_weights(model: onnx.ModelProto, layer: int, checkpoint: str) -> None:
    """Sobrescreve os initializers de q/k/v/out_proj e q/k_layernorm da
    camada com os pesos do checkpoint PyTorch (RopeFreeAttention)."""
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    matmul_by_prefix = {}
    for n in graph.node:
        for suffix in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if n.name == f"/lfm2/layers.{layer}/self_attn/{suffix}/MatMul_quant_float":
                matmul_by_prefix[suffix] = n

    sd = torch.load(checkpoint, map_location="cpu")
    mapping = {
        "q_proj.weight": ("q_proj", True),
        "k_proj.weight": ("k_proj", True),
        "v_proj.weight": ("v_proj", True),
        "out_proj.weight": ("out_proj", True),
    }
    for key, (suffix, transpose) in mapping.items():
        arr = sd[key].numpy().astype(np.float32)
        if transpose:
            arr = arr.T  # PyTorch Linear.weight e [out,in]; o ONNX espera [in,out] (X @ W)
        init = inits[matmul_by_prefix[suffix].input[1]]
        assert list(init.dims) == list(arr.shape), f"{suffix}: {list(init.dims)} != {arr.shape}"
        init.CopyFrom(numpy_helper.from_array(arr, init.name))

    for key, norm_name in (
        ("q_layernorm.weight", f"model.lfm2.layers.{layer}.self_attn.q_layernorm.weight"),
        ("k_layernorm.weight", f"model.lfm2.layers.{layer}.self_attn.k_layernorm.weight"),
    ):
        arr = sd[key].numpy().astype(np.float32)
        init = inits[norm_name]
        init.CopyFrom(numpy_helper.from_array(arr, init.name))

    print(f"  camada {layer}: pesos de q/k/v/out_proj e q/k_layernorm substituidos "
          f"pelo checkpoint {checkpoint}")


def remove_rope(model: onnx.ModelProto, layer: int) -> None:
    graph = model.graph
    prefix = f"/lfm2/layers.{layer}/self_attn/"
    by_name = {n.name: n for n in graph.node}

    q_pre_rope = by_name[f"{prefix}Transpose"].output[0]     # Q pos QK-Norm, antes do RoPE
    k_pre_rope = by_name[f"{prefix}Transpose_1"].output[0]   # K pos QK-Norm, antes do RoPE
    q_post_rope = by_name[f"{prefix}Add"].output[0]          # Q com RoPE aplicado
    k_post_rope = by_name[f"{prefix}Add_1"].output[0]        # K com RoPE aplicado

    rewired = 0
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == q_post_rope:
                n.input[i] = q_pre_rope
                rewired += 1
            elif inp == k_post_rope:
                n.input[i] = k_pre_rope
                rewired += 1
    print(f"  camada {layer}: religados {rewired} consumidor(es) de Q/K pos-RoPE")

    # dead-code elimination local: remove iterativamente qualquer no do bloco
    # self_attn cujo output nao e mais consumido por ninguem (nem dentro nem
    # fora do bloco) -- isso pega Slice/Neg/Concat/Mul/Add do RoPE sem listar
    # nome por nome, e para sozinho quando so sobra o que ainda e usado.
    removed_total = 0
    while True:
        block_nodes = [n for n in graph.node if n.name.startswith(prefix)]
        all_inputs = {inp for n in graph.node for inp in n.input}
        dead = [n for n in block_nodes if n.output[0] not in all_inputs
               and n.output[0] not in {o.name for o in graph.output}]
        if not dead:
            break
        dead_names = {n.name for n in dead}
        remaining = [n for n in graph.node if n.name not in dead_names]
        del graph.node[:]
        graph.node.extend(remaining)
        removed_total += len(dead)

    print(f"  camada {layer}: RoPE removido, {removed_total} nos orfaos eliminados")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("destination")
    ap.add_argument("--layer", type=int, action="append", required=True)
    ap.add_argument("--checkpoint", action="append", required=True)
    args = ap.parse_args()
    if len(args.layer) != len(args.checkpoint):
        raise SystemExit("numero de --layer e --checkpoint precisa bater")

    model = onnx.load(args.source)
    for layer, ckpt in zip(args.layer, args.checkpoint):
        load_weights(model, layer, ckpt)
        remove_rope(model, layer)

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, args.destination, save_as_external_data=False)
    print(f"salvo em {args.destination}")


if __name__ == "__main__":
    main()
