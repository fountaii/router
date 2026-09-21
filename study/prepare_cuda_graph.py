"""Create the fp16, fixed-shape CUDA-graph variants of the GPU router model."""

import sys

import onnx
from onnxconverter_common import float16
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed
from onnxruntime.transformers.optimizer import optimize_model


def convert(source: str, destination: str, sequence_length: int) -> None:
    model = onnx.load(source)
    for input_name in ("input_ids", "attention_mask"):
        make_input_shape_fixed(model.graph, input_name, [1, sequence_length])
    fix_output_shapes(model)

    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
        op_block_list=[*float16.DEFAULT_OP_BLOCK_LIST, "DequantizeLinear"],
    )
    del converted.graph.value_info[:]

    original_casts = {node.name for node in model.graph.node if node.op_type == "Cast"}
    dequant_outputs = {
        node.input[0] for node in converted.graph.node if node.op_type == "DequantizeLinear"
    }
    for node in converted.graph.node:
        if node.op_type != "Cast" or node.name not in original_casts or any(
            output in dequant_outputs for output in node.output
        ):
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                attribute.i = onnx.TensorProto.FLOAT16

    # Funde Mul/Add/Pow/Sqrt/Div/ReduceMean em SkipSimplifiedLayerNormalization: o
    # bucket tem ~200 desses nos, metade do tempo de GPU nao e MatMul (medido em
    # study/latency_findings.md), e essa fusao e a unica que rendeu algo (~5%,
    # cosseno 0.999997) sem trocar precisao nem exigir kernel proprio.
    fused = optimize_model(
        converted, model_type="bert", num_heads=16, hidden_size=1024,
        opt_level=0, use_gpu=True, only_onnxruntime=False,
    ).model

    # ponytail: o fusor cria SkipSimplifiedLayerNormalization com domain correto
    # (com.microsoft) mas deixa o SimplifiedLayerNormalization solto (o primeiro
    # de cada bloco, sem residual anterior) com domain="" -- e essa versao "" e
    # a unica que esta build do ORT sabe executar; com domain com.microsoft
    # explicito ele recusa em runtime ("not a registered function/op"). Falta
    # declarar o opset com.microsoft (o modelo de entrada nao tinha nenhum op
    # dele), senao o Skip* fica com domain sem import. onnx.checker nao aceita
    # esse domain="" hibrido (nao e um op ai.onnx padrao), entao a validacao
    # real e o proprio ORT conseguir carregar e rodar o arquivo, abaixo.
    if not any(imp.domain == "com.microsoft" for imp in fused.opset_import):
        fused.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    onnx.save(fused, destination, save_as_external_data=False)

    import onnxruntime as ort
    ort.InferenceSession(destination, providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare_cuda_graph.py <input.onnx> <output.onnx> <sequence-length>")
    convert(sys.argv[1], sys.argv[2], int(sys.argv[3]))
