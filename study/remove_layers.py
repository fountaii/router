"""Cirurgia arquitetural sem retreino: remove camadas inteiras do LFM2 do ONNX
fp32, religando o residual stream ao redor delas. Nao inventa peso nenhum --
so pula blocos. Ver study/latency_breakdown.txt para o porque (as 6 camadas de
atencao custam ~2.3x mais que as 10 de conv) e study/eval_routing.py para a
checagem de qualidade antes de aceitar qualquer variante.

Uso: python study/remove_layers.py <saida.onnx> <indice> [indice ...]
"""

import sys

import onnx


def layer_prefix(idx: int) -> str:
    return f"/lfm2/layers.{idx}/"


def layer_input_tensor(graph: onnx.GraphProto, idx: int) -> str:
    """Tensor do residual stream que entra na operator_norm da camada idx."""
    target = f"{layer_prefix(idx)}operator_norm/Pow"
    for n in graph.node:
        if n.name == target:
            return n.input[0]
    raise ValueError(f"operator_norm/Pow nao encontrada para a camada {idx}")


def layer_output_tensor(idx: int) -> str:
    """Tensor de saida da camada idx (residual apos a FFN), consumido pela
    proxima camada -- e o unico tensor que uma camada bem formada deveria
    exportar para fora de si mesma."""
    return f"{layer_prefix(idx)}Add_1_output_0"


def remove_layer(model: onnx.ModelProto, idx: int) -> None:
    graph = model.graph
    prefix = layer_prefix(idx)

    new_input = layer_input_tensor(graph, idx)
    old_output = layer_output_tensor(idx)

    candidate = [n for n in graph.node if n.name.startswith(prefix)]
    candidate_names = {n.name for n in candidate}
    candidate_outputs = {o for n in candidate for o in n.output}

    # 1) religa: quem consumia a saida da camada passa a consumir a entrada dela
    rewired = 0
    for n in graph.node:
        if n.name in candidate_names:
            continue
        for i, inp in enumerate(n.input):
            if inp == old_output:
                n.input[i] = new_input
                rewired += 1
    for out in graph.output:
        if out.name == old_output:
            raise ValueError(
                f"Camada {idx} produz uma saida do grafo ({old_output}) -- "
                "nao da para remover a ultima camada com este script."
            )
    if rewired == 0:
        raise ValueError(f"Nada consumia a saida da camada {idx}; grafo inesperado.")

    # 2) protege nos cujo output "escapa" do bloco para outra camada que
    # continua existindo (ex.: constantes de epsilon do RMSNorm que o export
    # deduplicou e hospedou dentro do namespace de uma camada, mas que outras
    # camadas tambem leem). Nao apagar esses -- sao baratos e inofensivos.
    consumed_elsewhere = set()
    for n in graph.node:
        if n.name in candidate_names:
            continue
        for inp in n.input:
            if inp in candidate_outputs:
                consumed_elsewhere.add(inp)
    protected = {n.name for n in candidate if set(n.output) & consumed_elsewhere}
    if protected:
        print(f"  [layer {idx}] preservando {len(protected)} no(s) compartilhado(s) "
              f"com outras camadas: {sorted(protected)}")

    to_remove = [n for n in candidate if n.name not in protected]
    remaining = [n for n in graph.node if n.name not in {n2.name for n2 in to_remove}]
    del graph.node[:]
    graph.node.extend(remaining)
    print(f"  [layer {idx}] removidos {len(to_remove)} nos, religados {rewired} "
          f"consumidor(es) de {new_input!r} -> pulando {old_output!r}")


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "uso: python study/remove_layers.py <origem.onnx> <saida.onnx> <indice> [indice ...]"
        )
    source, destination, *indices = sys.argv[1:]
    indices = sorted({int(i) for i in indices})

    model = onnx.load(source)
    for idx in indices:
        remove_layer(model, idx)

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, destination, save_as_external_data=False)
    print(f"salvo em {destination} (camadas removidas: {indices})")


if __name__ == "__main__":
    main()
