"""Reduz a largura intermediaria do FFN (SwiGLU: w1/w3 = gate/up 1024x4608,
w2 = down 4608x1024) por poda de magnitude, nao truncamento cru dos primeiros
N canais. Os 3 pesos do FFN sao ~14.2M parametros por camada -- a maior fatia
dos 548 MB de peso de MatMul que o forward le a cada chamada (ver
study/latency_findings.md, weight_census).

Selecao: pra cada camada, o canal intermediario j tem um score de importancia
= norma(w1[:,j]) + norma(w3[:,j]) + norma(w2[j,:]) (soma as tres pontas que
tocam esse neuronio -- um canal so importa se as tres normas dele forem
relevantes). Mantem os top-K por score. E poda estrutural sem retreino, mesma
familia da poda de camadas (study/remove_layers.py) mas dentro do bloco em vez
de no bloco inteiro -- valide com study/eval_routing.py antes de aceitar.

Uso: python study/prune_ffn.py <origem.onnx> <saida.onnx> <largura_nova>
"""

import sys

import numpy as np
import onnx
from onnx import numpy_helper


def prune_ffn(model: onnx.ModelProto, new_width: int) -> None:
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    matmul_by_name = {n.name: n for n in graph.node if n.op_type == "MatMul"}

    layers = set()
    for name in matmul_by_name:
        if "/feed_forward/w1/" in name:
            layers.add(name.split("/feed_forward/")[0])

    for layer_prefix in sorted(layers, key=lambda s: int(s.split(".")[-1])):
        w1_node = matmul_by_name[f"{layer_prefix}/feed_forward/w1/MatMul_quant_float"]
        w3_node = matmul_by_name[f"{layer_prefix}/feed_forward/w3/MatMul_quant_float"]
        w2_node = matmul_by_name[f"{layer_prefix}/feed_forward/w2/MatMul_quant_float"]

        w1_init = inits[w1_node.input[1]]
        w3_init = inits[w3_node.input[1]]
        w2_init = inits[w2_node.input[1]]

        w1 = numpy_helper.to_array(w1_init)  # [1024, 4608]
        w3 = numpy_helper.to_array(w3_init)  # [1024, 4608]
        w2 = numpy_helper.to_array(w2_init)  # [4608, 1024]

        width = w1.shape[1]
        if new_width >= width:
            continue

        score = (
            np.linalg.norm(w1, axis=0)
            + np.linalg.norm(w3, axis=0)
            + np.linalg.norm(w2, axis=1)
        )
        keep = np.sort(np.argpartition(score, -new_width)[-new_width:])

        w1_new = numpy_helper.from_array(np.ascontiguousarray(w1[:, keep]), w1_init.name)
        w3_new = numpy_helper.from_array(np.ascontiguousarray(w3[:, keep]), w3_init.name)
        w2_new = numpy_helper.from_array(np.ascontiguousarray(w2[keep, :]), w2_init.name)

        for old, new in ((w1_init, w1_new), (w3_init, w3_new), (w2_init, w2_new)):
            old.CopyFrom(new)

        print(f"  [{layer_prefix}] FFN {width} -> {new_width} canais "
              f"(score medio mantido {score[keep].mean():.3f} vs descartado "
              f"{np.delete(score, keep).mean():.3f})")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("uso: python study/prune_ffn.py <origem.onnx> <saida.onnx> <largura_nova>")
    source, destination, new_width = sys.argv[1], sys.argv[2], int(sys.argv[3])

    model = onnx.load(source)
    prune_ffn(model, new_width)

    # value_info do export original trava a dimensao 4608 (a shape inference
    # do ORT no load usa esse cache em vez de re-inferir); sem isso ele recusa
    # o w2/MatMul com "incompatible dimensions" mesmo com os pesos corretos.
    del model.graph.value_info[:]

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, destination, save_as_external_data=False)
    print(f"salvo em {destination} (FFN -> {new_width})")


if __name__ == "__main__":
    main()
