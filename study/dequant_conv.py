"""Troca ops inteiras do modelo quantizado pelas equivalentes em float.

Por padrao mexe so nos ConvInteger, que e a otimizacao que o router usa em CPU.
Com --matmuls troca tambem os MatMulInteger, produzindo um modelo fp32 inteiro:
as contrib ops int8 (MatMulIntegerToFloat, DynamicQuantizeMatMul) sao so de CPU,
entao nenhum provedor de GPU as aceita e o grafo parte. Esse modo existe para o
caminho de GPU.


ONNX Runtime nao tem kernel depthwise int8: com group=1024 o ConvInteger vira
1024 GEMMs minusculos e custa ~526 us por no (16% do run) para 58k MACs. Em
float o mesmo conv cai no caminho depthwise do MLAS. Os pesos sao 1024x1x3 por
camada, entao voltar 10 deles para float custa ~123 KB no arquivo.

Padrao reescrito (por camada):
    x -> DynamicQuantizeLinear -> (x_q, x_scale, x_zp)
    ConvInteger(x_q, w_q, x_zp, w_zp) -> Cast -> Mul(x_scale * w_scale) -> y
vira:
    Conv(x, w_q * w_scale) -> y
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

src = sys.argv[1]
dst = sys.argv[2]

model = onnx.load(src)
graph = model.graph
init = {i.name: i for i in graph.initializer}
produced_by = {out: n for n in graph.node for out in n.output}

consumers = {}
for node in graph.node:
    for name in node.input:
        consumers.setdefault(name, []).append(node)

drop = set()
new_weights = []
replacements = {}

INTEGER_TO_FLOAT = {"ConvInteger": "Conv", "MatMulInteger": "MatMul"}
targets = ["ConvInteger"] + (["MatMulInteger"] if "--matmuls" in sys.argv else [])

for conv in [n for n in graph.node if n.op_type in targets]:
    x_q, w_name, x_zp, w_zp_name = conv.input

    quantize = produced_by[x_q]
    assert quantize.op_type == "DynamicQuantizeLinear", quantize.op_type
    x = quantize.input[0]

    cast = consumers[conv.output[0]]
    assert len(cast) == 1 and cast[0].op_type == "Cast", cast
    cast = cast[0]

    scale_mul = consumers[cast.output[0]]
    assert len(scale_mul) == 1 and scale_mul[0].op_type == "Mul", scale_mul
    scale_mul = scale_mul[0]

    # o outro operando do Mul e scales_mul = x_scale * w_scale
    scales = next(i for i in scale_mul.input if i != cast.output[0])
    scales_node = produced_by[scales]
    w_scale = next(i for i in scales_node.input if i in init)

    w = numpy_helper.to_array(init[w_name]).astype(np.float32)
    w_zp = numpy_helper.to_array(init[w_zp_name]).astype(np.float32)
    w_float = (w - w_zp) * numpy_helper.to_array(init[w_scale])

    float_name = w_name + "_dequantized"
    new_weights.append(numpy_helper.from_array(w_float, float_name))

    replacements[conv.name] = helper.make_node(
        INTEGER_TO_FLOAT[conv.op_type],
        inputs=[x, float_name],
        outputs=[scale_mul.output[0]],
        name=conv.name + "_float",
        **{a.name: list(a.ints) if a.ints else a.i for a in conv.attribute},
    )

    drop.update({conv.name, cast.name, scale_mul.name, scales_node.name})

    # a quantizacao da entrada so existia para este conv
    if all(c.name in drop or c is conv for c in consumers[quantize.output[0]]):
        if all(
            c.name in drop
            for out in quantize.output
            for c in consumers.get(out, [])
        ):
            drop.add(quantize.name)

rebuilt = []
for node in graph.node:
    if node.name in replacements:
        rebuilt.append(replacements[node.name])
    elif node.name not in drop:
        rebuilt.append(node)

del graph.node[:]
graph.node.extend(rebuilt)
graph.initializer.extend(new_weights)

# sem isso o ORT reclama de cada peso int8 orfao em toda carga da sessao
used = {name for n in graph.node for name in n.input}
kept = [i for i in graph.initializer if i.name in used]
dropped_init = len(graph.initializer) - len(kept)
del graph.initializer[:]
graph.initializer.extend(kept)

onnx.checker.check_model(model, full_check=False)

print(f"convs trocados: {len(replacements)}  nos removidos: {len(drop)}")
print(f"pesos orfaos removidos: {dropped_init}  nos: {len(rebuilt)}")

# O onnxsim dobra o encanamento de shape em constantes. O ORT ja faz quase tudo
# isso ao carregar, entao o grafo executado muda pouco e o ganho e de ~5%, mas a
# saida e bit-identica. Sem onnxsim o modelo sai igual, so um pouco mais lento.
try:
    import onnxsim

    simplified, ok = onnxsim.simplify(model)
    if ok:
        print(f"onnxsim: {len(rebuilt)} -> {len(simplified.graph.node)} nos")
        model = simplified
    else:
        print("onnxsim: verificacao falhou, mantendo o grafo sem simplificar")
except ImportError:
    print("onnxsim ausente: pulando a simplificacao (pip install onnxsim)")

# em fp32 o modelo passa de 1.4 GB e encosta no limite de 2 GB do protobuf
external = model.ByteSize() > 1_500_000_000
onnx.save(model, dst, save_as_external_data=external, all_tensors_to_one_file=True,
          location=Path(dst).name + ".data" if external else None)
print(f"salvo em {dst}{' (+ .data)' if external else ''}")


# "Set a timer for ten minutes." com tres categorias, direto do tokenizer do router
REAL_IDS = [1, 544, 32830, 1334, 522, 23810, 6690, 1517, 708, 522, 17013, 4862, 708,
            522, 13663, 8253, 509, 6570, 1334, 6478, 768, 42236, 875, 3676, 5112, 523]


def check() -> None:
    """Confere que a troca preserva o roteamento do modelo original.

    Os valores nao batem exatamente de proposito: o caminho float deixa de
    quantizar a entrada do conv, entao fica mais perto do modelo sem quantizacao,
    nao mais longe. O que precisa continuar valendo e a direcao dos vetores, que
    e o que a cabeca cosseno usa para decidir a categoria.
    """
    import numpy as np
    import onnxruntime as ort

    ids = np.array([REAL_IDS], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}

    before = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
    after = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, feed)

    # os tokens que divergem sao os de norma ~4 (pontuacao/estrutura); os de norma
    # ~94, que dominam o pooling, ficam em cosseno 1.000. Por isso a conferencia e
    # no vetor agrupado, que e o que a cabeca realmente compara.
    for name, a, b in zip(["token_proj", "rule_proj"], before, after):
        pooled = [x.reshape(-1, x.shape[-1]).sum(0) for x in (a, b)]
        cos = float(
            pooled[0] @ pooled[1]
            / (np.linalg.norm(pooled[0]) * np.linalg.norm(pooled[1]) + 1e-9)
        )
        print(f"{name}: cosseno agrupado {cos:.5f}")
        assert cos > 0.999, f"{name} mudou de direcao: {cos}"


if "--check" in sys.argv:
    check()
