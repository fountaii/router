"""Exporta o LFM2.5-Encoder-230M oficial para ONNX e gera int8 e int4.

Exporta so o encoder (sem a cabeca de masked-LM, cujo vocab de 65k dominaria a
medicao). A saida e o last_hidden_state [batch, sequence, 1024], que e de onde um
router tiraria as projecoes.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM


class Encoder(torch.nn.Module):
    """So o encoder; a cabeca de masked-LM fica de fora."""

    def __init__(self, lfm2):
        super().__init__()
        self.lfm2 = lfm2

    def forward(self, input_ids, attention_mask):
        return self.lfm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state

MODEL_ID = "LiquidAI/LFM2.5-Encoder-230M"
out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/perf/lfm230")
out_dir.mkdir(parents=True, exist_ok=True)
fp32 = out_dir / "model_fp32.onnx"

if not fp32.exists():
    print(f"carregando {MODEL_ID} ...")
    loaded = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model = Encoder(loaded.lfm2)
    model.eval()

    ids = torch.randint(1, 30000, (1, 26), dtype=torch.long)
    mask = torch.ones_like(ids)

    print("exportando para ONNX ...")
    torch.onnx.export(
        model,
        (ids, mask),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"fp32: {fp32.stat().st_size / 1048576:.0f} MB")

# int8 dinamico, igual ao formato do modelo que o router usa hoje
int8 = out_dir / "model_int8.onnx"
if not int8.exists():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("quantizando int8 ...")
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"int8: {int8.stat().st_size / 1048576:.0f} MB")

# int4 com as duas coisas que faltavam nos int4 existentes: bloco 128 E accuracy_level 4
int4 = out_dir / "model_int4.onnx"
if not int4.exists():
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    print("quantizando int4 (bloco 128, accuracy_level 4) ...")
    # sem o Gather a tabela de embedding (65536x1024) fica em fp32 e o arquivo
    # int4 sai maior que o int8 — foi o que estragou o model_q4 do hub
    quantizer = MatMulNBitsQuantizer(
        onnx.load(str(fp32)),
        algo_config=DefaultWeightOnlyQuantConfig(
            block_size=128,
            is_symmetric=True,
            accuracy_level=4,
            op_types_to_quantize=("MatMul", "Gather"),
            quant_axes=(("MatMul", 0), ("Gather", 1)),
        ),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(int4), use_external_data_format=False)

    # o quantizador sobe ai.onnx para 21 sem converter os ops; no 18 o ReduceMean
    # passou a receber axes como entrada, e o grafo fica invalido. Nenhum op
    # ai.onnx foi alterado aqui, entao voltar a declaracao para 17 e correto.
    fixed = onnx.load(str(int4))
    for opset in fixed.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 17
    onnx.checker.check_model(fixed, full_check=False)
    onnx.save(fixed, str(int4))
    print(f"int4: {int4.stat().st_size / 1048576:.0f} MB")

for path in (fp32, int8, int4):
    print(f"{path.name:18s} {path.stat().st_size / 1048576:8.0f} MB")
