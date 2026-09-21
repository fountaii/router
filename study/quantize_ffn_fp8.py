"""Converte os 3 pesos do FFN (w1/w3 gate+up, w2 down -- a maior fatia dos
548 MB de peso de MatMul, ver weight_census em study/latency_findings.md)
para FLOAT8E4M3FN com DequantizeLinear por tensor, escala = max_abs/448
(448 = maior valor representavel em e4m3). Peso-somente: a entrada continua
fp32/fp16, so o peso estatico troca de formato de armazenamento.

Uso: python study/quantize_ffn_fp8.py <origem.onnx> <saida.onnx>
"""

import sys

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

E4M3_MAX = 448.0


def quantize_ffn_fp8(model: onnx.ModelProto) -> None:
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    matmul_by_name = {n.name: n for n in graph.node}

    targets = [n for n in graph.node if n.op_type == "MatMul" and "/feed_forward/w" in n.name]
    for node in targets:
        weight_init = inits[node.input[1]]
        w = numpy_helper.to_array(weight_init).astype(np.float32)

        scale = float(np.abs(w).max()) / E4M3_MAX
        scale = scale if scale > 0 else 1.0
        w_f8 = (w / scale).astype(ml_dtypes.float8_e4m3fn)

        w_f8_name = weight_init.name + "_f8"
        scale_name = weight_init.name + "_scale"
        deq_name = weight_init.name + "_deq"

        graph.initializer.append(numpy_helper.from_array(w_f8, w_f8_name))
        graph.initializer.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), scale_name))
        deq_node = helper.make_node(
            "DequantizeLinear", [w_f8_name, scale_name], [deq_name],
            name=f"{node.name}_fp8_dequant",
        )
        # insere antes do consumidor -- initializers nao precisam de ordem
        # topologica, mas nos precisam, e so appendar no fim deixaria o
        # DequantizeLinear depois do MatMul que o consome.
        node_index = list(graph.node).index(node)
        graph.node.insert(node_index, deq_node)
        node.input[1] = deq_name
        graph.initializer.remove(weight_init)

    print(f"  {len(targets)} matrizes de FFN convertidas para FLOAT8E4M3FN")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("uso: python study/quantize_ffn_fp8.py <origem.onnx> <saida.onnx>")
    source, destination = sys.argv[1], sys.argv[2]

    model = onnx.load(source)
    quantize_ffn_fp8(model)
    del model.graph.value_info[:]

    # FLOAT8E4M3FN so e valido a partir do opset 19 (o CUDA EP recusa o tipo
    # com o opset original 17, mesmo o tensor estando correto). Bump exige
    # migrar ReduceMean: opset 18 moveu `axes` de atributo pra input.
    graph = model.graph
    for n in graph.node:
        if n.op_type == "ReduceMean":
            axes_attr = next((a for a in n.attribute if a.name == "axes"), None)
            if axes_attr is not None:
                axes_name = f"{n.name}_axes"
                graph.initializer.append(
                    numpy_helper.from_array(np.array(list(axes_attr.ints), dtype=np.int64), axes_name)
                )
                n.input.append(axes_name)
                n.attribute.remove(axes_attr)
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            imp.version = 19
    if model.ir_version > 9:
        model.ir_version = 9

    onnx.save(model, destination, save_as_external_data=False)
    print(f"salvo em {destination}")


if __name__ == "__main__":
    main()
