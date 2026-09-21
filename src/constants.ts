export const ROUTER_MODEL_ID="kucukkanat/LFM2.5-Encoder-350M-Prompt-Router-ONNX"
export const ROUTER_MODEL_FILE="onnx/model_quantized.onnx"
// gerado por `bun run build:model`: os ConvInteger depthwise viram Conv float (~20% mais rapido).
export const ROUTER_MODEL_FILE_FAST="onnx/model_quantized_convfloat.onnx"
export const ROUTER_SEQ_LEN=128
export const ROUTER_MAX_LANES=8
export const ROUTER_THREADS=6
// gerado por `bun run build:model -- --gpu`: fp32 inteiro, unico formato que os
// provedores de GPU aceitam sem partir o grafo.
export const ROUTER_MODEL_FILE_GPU="onnx/model_fp32.onnx"
// Gerados por `bun run build:model -- --gpu`. Cada bucket tem endereco e forma
// constantes, condicoes necessarias para o CUDA graph.
export const ROUTER_CUDA_GRAPH_BUCKETS=[32, 64, 128] as const
export const ROUTER_CUDA_GRAPH_MODEL_FILE=(sequenceLength: number) =>
    `onnx/model_fp16_fixed${sequenceLength}.onnx`
