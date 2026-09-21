"""Substitui o bloco de self-attention de uma camada pelo bloco de convolucao
curta gated de OUTRA camada ja treinada (reaproveita os pesos, nao inventa
nada novo). Diferente de remove_layers.py (que pula a camada inteira), aqui a
camada continua existindo -- so troca o "sequence block" (attn por conv),
mantendo a norma e o FFN da propria camada intactos.

Os pesos da conv doadora sao REFERENCIADOS, nao copiados (ONNX deixa varios
nos lerem o mesmo initializer) -- zero peso novo, zero bytes a mais.

Uso: python study/attn_to_conv.py <origem.onnx> <saida.onnx> <camada_attn> <camada_conv_doadora>
"""

import sys

import onnx
from onnx import helper


def attn_to_conv(model: onnx.ModelProto, attn_layer: int, donor_layer: int) -> None:
    graph = model.graph
    attn_prefix = f"/lfm2/layers.{attn_layer}/self_attn/"
    donor_prefix = f"/lfm2/layers.{donor_layer}/conv/"

    donor_nodes = [n for n in graph.node if n.name.startswith(donor_prefix)]
    if not donor_nodes:
        raise ValueError(f"Nenhum no de conv encontrado para a camada {donor_layer}")

    donor_in = next(
        n.output[0] for n in graph.node
        if n.name == f"/lfm2/layers.{donor_layer}/operator_norm/Mul_1"
    )
    donor_out = next(
        n.output[0] for n in graph.node
        if n.name.startswith(f"{donor_prefix}out_proj/MatMul")
    )

    # tensores produzidos DENTRO do bloco doador (precisam de nome novo pra
    # nao colidir com a instancia original, que continua existindo na camada
    # doadora); tensores de fora (initializers, constantes globais /lfm2/...
    # e /lfm2/layers.0/conv/... compartilhadas entre todas as convs) ficam
    # como estao -- sao referenciados, nao clonados.
    donor_internal_outputs = {o for n in donor_nodes for o in n.output}

    def remap(tensor_name: str) -> str:
        if tensor_name == donor_in:
            return attn_in  # entrada real: a norma da propria camada attn
        if tensor_name in donor_internal_outputs:
            return tensor_name.replace(
                f"layers.{donor_layer}/conv/", f"layers.{attn_layer}/conv_from_{donor_layer}/"
            )
        return tensor_name  # initializer ou constante global -- referencia direto

    attn_in = next(
        n.output[0] for n in graph.node
        if n.name == f"/lfm2/layers.{attn_layer}/operator_norm/Mul_1"
    )

    cloned = []
    for n in donor_nodes:
        new_node = helper.make_node(
            n.op_type,
            [remap(i) for i in n.input],
            [remap(o) for o in n.output],
            name=n.name.replace(f"layers.{donor_layer}/conv/", f"layers.{attn_layer}/conv_from_{donor_layer}/"),
        )
        new_node.attribute.extend(n.attribute)
        cloned.append(new_node)

    new_block_out = remap(donor_out)

    # religa: quem consumia a saida do self_attn (o Add residual da propria
    # camada attn) passa a consumir a saida do bloco conv clonado.
    attn_out = next(
        n.output[0] for n in graph.node
        if n.name.startswith(f"{attn_prefix}out_proj/MatMul")
    ) if any(n.name.startswith(f"{attn_prefix}out_proj/MatMul") for n in graph.node) else None
    if attn_out is None:
        # o LFM2 nomeia a saida da attention como .../o_proj/MatMul, nao out_proj
        attn_out = next(
            n.output[0] for n in graph.node
            if n.name.startswith(attn_prefix) and n.name.endswith("MatMul_quant_float")
            and "o_proj" in n.name
        )

    rewired = 0
    for n in graph.node:
        if n.name.startswith(attn_prefix):
            continue
        for i, inp in enumerate(n.input):
            if inp == attn_out:
                n.input[i] = new_block_out
                rewired += 1
    if rewired == 0:
        raise ValueError("Nada consumia a saida do self_attn; nomes inesperados no grafo.")

    # remove os nos do self_attn original (protegendo qualquer coisa que
    # escape pra fora do bloco, igual remove_layers.py)
    attn_nodes = [n for n in graph.node if n.name.startswith(attn_prefix)]
    attn_outputs = {o for n in attn_nodes for o in n.output}
    consumed_elsewhere = {
        inp for n in graph.node if not n.name.startswith(attn_prefix)
        for inp in n.input if inp in attn_outputs
    }
    protected = {n.name for n in attn_nodes if set(n.output) & consumed_elsewhere}
    if protected:
        print(f"  preservando {len(protected)} no(s) do self_attn compartilhado com outras camadas")

    keep_names = {n.name for n in attn_nodes if n.name not in protected}
    original_nodes = list(graph.node)
    first_removed_index = min(
        i for i, n in enumerate(original_nodes) if n.name in keep_names
    )
    remaining = [n for n in original_nodes if n.name not in keep_names]
    # insere os clonados onde o primeiro no removido estava, preservando a
    # ordem topologica (a entrada dos clonados e a norma da propria camada,
    # que ja existe antes desse ponto; a saida alimenta o Add residual, que
    # vem depois).
    insert_at = sum(1 for n in original_nodes[:first_removed_index] if n.name not in keep_names)
    del graph.node[:]
    graph.node.extend(remaining[:insert_at])
    graph.node.extend(cloned)
    graph.node.extend(remaining[insert_at:])

    print(f"  camada {attn_layer}: self_attn ({len(attn_nodes) - len(protected)} nos removidos) "
          f"-> conv da camada {donor_layer} ({len(cloned)} nos clonados, pesos referenciados)")


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "uso: python study/attn_to_conv.py <origem.onnx> <saida.onnx> <camada_attn> <camada_conv_doadora>"
        )
    source, destination, attn_layer, donor_layer = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

    model = onnx.load(source)
    attn_to_conv(model, attn_layer, donor_layer)

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, destination, save_as_external_data=False)
    print(f"salvo em {destination}")


if __name__ == "__main__":
    main()
