#!/usr/bin/env node
// Gera o modelo com conv float ao lado do modelo baixado do hub.
// Precisa de python com onnx instalado; o router cai no modelo original se faltar.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const hub = path.join(
  process.env.HOME ?? process.cwd(),
  ".cache/huggingface/hub/kucukkanat--LFM2.5-Encoder-350M-Prompt-Router-ONNX/onnx",
);

// --gpu gera o fp32 inteiro e os buckets fp16 de CUDA graph. As contrib ops int8
// do modelo quantizado sao so de CPU.
const gpu = process.argv.includes("--gpu");

const src = path.join(hub, "model_quantized.onnx");
const dst = path.join(hub, gpu ? "model_fp32.onnx" : "model_quantized_convfloat.onnx");

if (!existsSync(src)) {
  throw new Error(`Modelo do hub nao encontrado em ${src}. Rode o router uma vez para baixar.`);
}

const python = process.env.PYTHON ?? "python";
const args = ["study/dequant_conv.py", src, dst, ...(gpu ? ["--matmuls"] : [])];
const result = spawnSync(python, args, { stdio: "inherit" });

if (result.status !== 0) {
  throw new Error(`${python} study/dequant_conv.py falhou (${result.status ?? result.signal}).`);
}

if (gpu) {
  // As camadas de atencao 12 e 14 (das 6 do LFM2, indices 2/5/8/10/12/14) sao
  // as mais profundas e as mais redundantes: removidas do fp32, o ranking do
  // router nao muda (26/26 casos, cosseno de probs 0.9999, incl. categorias
  // quase-sinonimas de proposito) e o route() cai de 2.70 para 2.34ms p50.
  // As tres primeiras (2/5/8) sao criticas -- nao mexer. Ver
  // study/latency_findings.md ("Poda de camadas") e study/remove_layers.py.
  const PRUNED_LAYERS = [12, 14];
  const prunedDst = path.join(hub, "model_fp32_pruned.onnx");
  const pruneResult = spawnSync(
    python,
    ["study/remove_layers.py", dst, prunedDst, ...PRUNED_LAYERS.map(String)],
    { stdio: "inherit" },
  );
  if (pruneResult.status !== 0) {
    throw new Error(`${python} study/remove_layers.py falhou (${pruneResult.status ?? pruneResult.signal}).`);
  }

  // FFN 4608 -> 4096 por poda de magnitude (score por canal = soma das normas
  // de w1/w3/w2 que tocam aquele neuronio). Empilhado com a poda de camada
  // acima, o cosseno minimo contra a referencia cai de 0.9999 para 0.89 (26/26
  // ainda corretos nos 26 casos de teste) e ganha mais ~6% de latencia
  // (2.34 -> 2.20ms). Margem mais fina que a poda de camada sozinha -- ver
  // study/latency_findings.md ("Compressao do FFN") antes de estreitar mais.
  const FFN_WIDTH = 4096;
  const ffnDst = path.join(hub, "model_fp32_pruned_ffn.onnx");
  const ffnResult = spawnSync(
    python,
    ["study/prune_ffn.py", prunedDst, ffnDst, String(FFN_WIDTH)],
    { stdio: "inherit" },
  );
  if (ffnResult.status !== 0) {
    throw new Error(`${python} study/prune_ffn.py falhou (${ffnResult.status ?? ffnResult.signal}).`);
  }

  // As 3 camadas de atencao restantes (2, 5, 8 -- as criticas, que a poda de
  // camada acima nao tocou) trocam de "atencao completa com RoPE" para
  // "mesmos pesos Q/K/V/O, sem RoPE" -- destilado fim-a-fim (gradiente pela
  // rede inteira, loss na saida final, 1 epoca contra dataset sintetico,
  // early stopping) contra o professor original. Pesos em study/checkpoints/,
  // reproduz via study/distill_e2e.py --module rope_free_attn. Ganho: ~4%
  // (1.90 -> 1.82ms de session.run). Ver study/latency_findings.md
  // ("Atencao sem RoPE, destilada").
  const ropeFreeLayers = [2, 5, 8];
  const ropeFreeDst = path.join(hub, "model_fp32_pruned_ffn_ropefree.onnx");
  const ropeFreeArgs = ["study/export_rope_free.py", ffnDst, ropeFreeDst];
  for (const layer of ropeFreeLayers) {
    ropeFreeArgs.push("--layer", String(layer), "--checkpoint", `study/checkpoints/rope_free_layer${layer}.pt`);
  }
  const ropeFreeResult = spawnSync(python, ropeFreeArgs, { stdio: "inherit" });
  if (ropeFreeResult.status !== 0) {
    throw new Error(`${python} study/export_rope_free.py falhou (${ropeFreeResult.status ?? ropeFreeResult.signal}).`);
  }

  for (const sequenceLength of [32, 64, 128]) {
    const graphDst = path.join(hub, `model_fp16_fixed${sequenceLength}.onnx`);
    const graphResult = spawnSync(python, ["study/prepare_cuda_graph.py", ropeFreeDst, graphDst, String(sequenceLength)], {
      stdio: "inherit",
    });
    if (graphResult.status !== 0) {
      throw new Error(`${python} study/prepare_cuda_graph.py falhou (${graphResult.status ?? graphResult.signal}).`);
    }
  }
}

console.log(`Gerado ${dst}`);
