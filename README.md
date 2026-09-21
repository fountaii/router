# router

A local, GPU-accelerated prompt router built on [LiquidAI's LFM2.5-Encoder-350M-Prompt-Router](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router), hand-optimized end to end: **2.70 ms → 2.20 ms p50** (~20% faster), **613 MB → 505 MB**, zero retraining of the base model, precision validated at every step. Every optimization attempted — the ones that worked and the ones that didn't — is written up in [`study/latency_findings.md`](study/latency_findings.md).

[**Project page**](https://eletroswing.github.io/router/) · [English](#english) · [Português](#português)

---

## English

### What this is

A prompt/intent router: given a short text and a list of categories, it returns which category the text belongs to, using cosine similarity between pooled embeddings from a bidirectional encoder (LFM2.5, 350M params). It runs on Node/Bun with a custom native ONNX Runtime binding (CPU + CUDA), no Python at inference time.

This repo is the record of pushing that router's latency down as far as it would go without retraining the base model — through ONNX graph surgery, magnitude-based pruning, and (once those hit a wall) local knowledge distillation of individual layers.

### Results

| | Before | After | |
|---|---:|---:|---|
| `route()` p50 | 2.70 ms | **2.20 ms** | ~19% faster |
| `session.run` p50 | 2.35 ms | **1.82 ms** | ~23% faster |
| Model size (fp16) | 613 MB | **505 MB** | ~18% smaller |
| Layers | 16 | **14** | 2 removed |
| FFN width | 4608 | **4096** | pruned by magnitude |
| Attention layers | 6, full RoPE | **3 without RoPE** (distilled), 1 full | |

Measured on an RTX 4090 Laptop GPU, CUDA EP, CUDA graph, fp16, bucket of 32 tokens. Routing accuracy validated on 26 cases (16 easy + 10 deliberately hard near-synonym categories): 25-26/26 correct depending on the exact combination, full breakdown in the findings doc.

### What actually worked (applied to the shipped model)

1. **Layer pruning** — removed 2 of the 6 attention layers (the 2 deepest, most redundant ones; the other 4 broke routing badly when removed). Zero-shot ONNX graph surgery, no training: 26/26 routing preserved, cosine ≥0.999.
2. **FFN pruning** — cut the SwiGLU intermediate width 4608→4096 by magnitude scoring (not naive truncation of the first N channels). Same validation, ~6% extra latency.
3. **Attention distilled without RoPE** — the 3 most critical attention layers (removing them broke routing outright) keep their original Q/K/V/O weights and QK-Norm, but drop rotary position embeddings, closing the gap with **1 epoch** of end-to-end distillation (loss on the router's final output, gradient through the whole network, `accelerate` + bf16) against a synthetic dataset in the router's own prompt format. ~4% extra latency, minimal quality cost.

### What was tried and discarded (documented so nobody repeats the work)

- **TensorRT**: even the only numerically-correct variant (fp32 + CUDA graph) was slower than the plain CUDA EP path (4.37 ms vs ~2.2 ms), plus a 1.42 GB engine. fp16 inside TensorRT collapsed precision (cosine 0.141).
- **INT8/INT4/FP8 weight-only quantization**: tested all three formats on the actual bottleneck (FFN weights). INT8 was *slower* than fp16 (dequant overhead beats the bandwidth saved at this batch size). INT4 broke precision. FP8 was slower too, precision fine — same root cause: generic `DequantizeLinear`→dense `MatMul` isn't a fused low-precision GEMM.
- **2:4 structured sparsity**: broke precision (cosine 0.28) — unstructured pruning without retraining, as expected.
- **Attention → conv, all the way**: replacing *every* remaining attention layer with gated short convolutions plateaued at a hard ceiling no amount of training moved (cosine stuck at ~0.13 across 40 epochs) — convolution only mixes local neighbors; at least one test case needs long-range, content-dependent mixing that only attention provides. RoPE-free attention (above) is the point that actually holds.
- Full detail, numbers, and the reasoning behind each call: [`study/latency_findings.md`](study/latency_findings.md).

### Next step: ternary quantization

The remaining floor is bandwidth to VRAM — reading ~500 MB of fp16 weight per forward pass. The one lever that hasn't been tried yet: **ternary weights (~1.58 bit, BitNet-style)**, which — unlike the INT8/INT4/FP8 attempts above — replaces the GEMM itself (accumulate ±1/0 instead of full multiply-add) rather than just shrinking storage and dequantizing back to dense. That's a real kernel-level change, not a drop-in ONNX Runtime op, so it's the next thing on the list rather than something finished here.

### Quickstart

```bash
bun install
bun run index.ts
```

### Reproducing the optimized model from scratch

```bash
bun run build:model -- --gpu
```

Runs the full pipeline: fp32 conversion → layer pruning → FFN pruning → RoPE-free attention distillation (using the checkpoints in `study/checkpoints/`, no training needed to reproduce) → fp16 CUDA-graph buckets (32/64/128 tokens). Needs Python with `onnx`, `onnxruntime`, `onnxsim`, `torch`, `transformers`, `accelerate` — see `study/*.py` for the individual steps and [`study/latency_findings.md`](study/latency_findings.md) for what each one does and why.

To retrain the distilled attention layers yourself (or try new ones): `study/distill_e2e.py`.

### Credits & license

Built on [`LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router) (LFM2 architecture, [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router/blob/main/LICENSE)) and its [ONNX export](https://huggingface.co/kucukkanat/LFM2.5-Encoder-350M-Prompt-Router-ONNX). The weights in `study/checkpoints/` are a local fine-tune of specific attention sub-layers of that same checkpoint (RoPE removed, QK-Norm and Q/K/V/O projections kept), released under the same license as the base model. See the LFM2 paper (Liquid AI, Dec 2025) for the architecture this all builds on.

---

## Português

### O que é isso

Um router de prompt/intenção: dado um texto curto e uma lista de categorias, retorna a qual categoria o texto pertence, usando similaridade de cosseno entre embeddings de um encoder bidirecional (LFM2.5, 350M parâmetros). Roda em Node/Bun com um binding nativo próprio do ONNX Runtime (CPU + CUDA), sem Python em tempo de inferência.

Este repositório é o registro de empurrar a latência desse router pro mais baixo que deu, sem retreinar o modelo base — via cirurgia de grafo ONNX, poda por magnitude, e (quando isso bateu num teto) destilação local de camadas específicas.

### Resultados

| | Antes | Depois | |
|---|---:|---:|---|
| `route()` p50 | 2.70 ms | **2.20 ms** | ~19% mais rápido |
| `session.run` p50 | 2.35 ms | **1.82 ms** | ~23% mais rápido |
| Tamanho do modelo (fp16) | 613 MB | **505 MB** | ~18% menor |
| Camadas | 16 | **14** | 2 removidas |
| Largura do FFN | 4608 | **4096** | podado por magnitude |
| Camadas de atenção | 6, RoPE completo | **3 sem RoPE** (destiladas), 1 completa | |

Medido numa RTX 4090 Laptop, CUDA EP, CUDA graph, fp16, bucket de 32 tokens. Precisão de roteamento validada em 26 casos (16 fáceis + 10 propositalmente difíceis, categorias quase-sinônimas): 25-26/26 corretos dependendo da combinação exata, detalhe completo no documento de achados.

### O que funcionou de verdade (aplicado no modelo em produção)

1. **Poda de camadas** — removidas 2 das 6 camadas de atenção (as 2 mais profundas e redundantes; as outras 4 quebravam o roteamento feio quando removidas). Cirurgia de grafo ONNX sem treino nenhum: 26/26 preservado, cosseno ≥0.999.
2. **Poda de FFN** — largura intermediária do SwiGLU cortada de 4608 pra 4096 por pontuação de magnitude (não truncamento cru dos primeiros N canais). Mesma validação, ~6% de latência a mais.
3. **Atenção destilada sem RoPE** — as 3 camadas de atenção mais críticas (removê-las quebrava o roteamento na hora) mantêm os pesos Q/K/V/O e o QK-Norm originais, só tiram o rotary position embedding, fechando a diferença com **1 época** de destilação fim-a-fim (loss na saída final do router, gradiente pela rede inteira, `accelerate` + bf16) contra um dataset sintético no formato real do router. ~4% de latência a mais, custo de qualidade mínimo.

### O que foi tentado e descartado (documentado pra ninguém repetir o trabalho)

- **TensorRT**: até a única variante numericamente correta (fp32 + CUDA graph) ficou mais lenta que o caminho CUDA EP comum (4.37 ms vs ~2.2 ms), além de uma engine de 1.42 GB. fp16 dentro do TensorRT quebrou a precisão (cosseno 0.141).
- **Quantização INT8/INT4/FP8 (peso-somente)**: testei os três formatos no gargalo real (peso do FFN). INT8 ficou *mais lento* que fp16 (overhead de desempacotar supera a banda economizada nesse tamanho de lote). INT4 quebrou precisão. FP8 também ficou mais lento, precisão ok — mesma causa raiz: `DequantizeLinear`→`MatMul` denso genérico não é um GEMM de baixa precisão fundido de verdade.
- **Esparsidade estruturada 2:4**: quebrou precisão (cosseno 0.28) — poda sem retreino, como esperado.
- **Attention → conv, até o fim**: substituir *toda* camada de atenção restante por convolução curta gated bateu num teto que nenhuma quantidade de treino moveu (cosseno travado em ~0.13 ao longo de 40 épocas) — convolução só mistura vizinhos locais; pelo menos um caso de teste precisa de mistura de longo alcance dependente de conteúdo, que só atenção entrega. A atenção sem RoPE (acima) é o ponto que realmente segura.
- Detalhe completo, números e o raciocínio por trás de cada decisão: [`study/latency_findings.md`](study/latency_findings.md).

### Próximo passo: quantização ternária

O piso que sobra é banda até a VRAM — ler ~500 MB de peso fp16 a cada forward. A alavanca que ainda não foi testada: **pesos ternários (~1.58 bit, estilo BitNet)**, que — diferente das tentativas INT8/INT4/FP8 acima — substitui o próprio GEMM (acumula ±1/0 em vez de multiplicação completa) em vez de só encolher o armazenamento e desempacotar de volta pra denso. Isso é mudança de nível de kernel de verdade, não um op de ONNX Runtime plug-and-play, então fica como o próximo item da lista em vez de algo terminado aqui.

### Começando

```bash
bun install
bun run index.ts
```

### Reproduzindo o modelo otimizado do zero

```bash
bun run build:model -- --gpu
```

Roda o pipeline completo: conversão fp32 → poda de camadas → poda de FFN → destilação da atenção sem RoPE (usando os checkpoints em `study/checkpoints/`, não precisa treinar de novo pra reproduzir) → buckets fp16 com CUDA graph (32/64/128 tokens). Precisa de Python com `onnx`, `onnxruntime`, `onnxsim`, `torch`, `transformers`, `accelerate` — ver `study/*.py` pras etapas individuais e [`study/latency_findings.md`](study/latency_findings.md) pro que cada uma faz e por quê.

Pra retreinar as camadas de atenção destiladas você mesmo (ou tentar outras novas): `study/distill_e2e.py`.

### Créditos e licença

Construído sobre [`LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router) (arquitetura LFM2, [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router/blob/main/LICENSE)) e a [exportação ONNX](https://huggingface.co/kucukkanat/LFM2.5-Encoder-350M-Prompt-Router-ONNX) dele. Os pesos em `study/checkpoints/` são um fine-tune local de sub-camadas específicas de atenção desse mesmo checkpoint (RoPE removido, QK-Norm e projeções Q/K/V/O mantidas), publicados sob a mesma licença do modelo base. Ver o paper do LFM2 (Liquid AI, dez/2025) pra arquitetura em que tudo isso se apoia.

---

## Technical reference / Referência técnica

<details>
<summary><strong>Native binding build, GPU runtime setup, detailed flags (click to expand)</strong></summary>

### Build somente do binding nativo

```bash
bun run build:native
```

Compila os fontes C++ de `src/native` e atualiza somente
`src/onnx/bin/<plataforma>/<arquitetura>/onnxruntime_binding.node` da máquina atual.
Reutiliza a DLL/shared library de `src/native/lib/<plataforma>/<arquitetura>`
(ou a existente em `src/onnx/bin`), sem recompilar ONNX Runtime. Cria a pasta
de saída e copia a biblioteca ao lado do `.node`, inclusive com `bin` vazio.
Os headers ONNX Runtime e Node-API, import libraries do Windows e licenças estão
locais em `src/native`. Depois de `bun install`, o build não precisa de rede.

Requisitos: Node.js e, no Windows, Visual Studio Build Tools com C++ e Windows SDK;
no Linux/macOS, compilador C++17 (`CXX` pode selecionar o executável).
Não faz cross-compilation. Os artefatos intermediários ficam em `.cache/native`.

O binding reduzido suporta CPU, modelos por caminho, tensores `float32`/`int64`,
nomes de entradas/saídas, opções de threads/memória/otimização, logging, profiling
e liberação da sessão. Foram removidos provedores GPU, IO binding, carregamento
por buffer, metadados de tipos/shapes e conversões de tipos não expostos pelo wrapper.
As bibliotecas dos provedores já presentes em `src/onnx/bin` foram preservadas.

Validação do binding (fixtures Identity locais, sem download do modelo):

```bash
bun test src/native/tests/binding.test.mjs
node --test src/native/tests/binding.test.mjs
```

### Modelo otimizado para CPU (opcional, ~20% mais rápido)

```bash
bun run build:model
```

Reescreve os `ConvInteger` depthwise do modelo baixado como `Conv` float, dobra o
encanamento de shape com `onnxsim` e salva `onnx/model_quantized_convfloat.onnx`
ao lado do original no cache do Hugging Face. O router usa esse arquivo
automaticamente quando ele existe e cai no modelo do hub quando não existe.

Precisa de python com `onnx` (`onnxruntime` para `--check`, `onnxsim` para os
últimos ~5%; sem ele o modelo sai igual, só um pouco mais lento). Não é usado em
tempo de execução.

### GPU (CUDA)

As contrib ops int8 (`MatMulIntegerToFloat`, `DynamicQuantizeMatMul`) só existem
na CPU. Pedir `cuda` com o modelo quantizado faz o ORT partir o grafo e ficar
*mais lento* que a CPU, então `fromHub` exige o `model_fp32.onnx` e falha com
mensagem clara se ele não existe.

#### CUDA graph (padrão em `--gpu`)

Esse caminho mantém buffers GPU estáveis e os reescreve com a API de driver da
NVIDIA antes de cada execução. O router completa os prompts com padding até o
menor bucket e continua a fazer o pooling apenas dos tokens reais. Cada bucket
mantém sua própria sessão, pois CUDA graph exige que forma e endereços não mudem.

Para comparar com o caminho fp32 comum, defina `ROUTER_CUDA_GRAPH=0`.

`index.ts` imprime p50/p95 de 100 chamadas de propósito — uma chamada isolada
depois do warmup ainda mede o caminho frio do JIT/clock da GPU subindo.

#### Runtime CUDA

O provider precisa do cuBLAS 13 e do cuDNN 9, que **não são embarcados** (~900 MB,
além dos 176 MB do próprio provider). Duas formas de fornecê-los:

```bash
# apontar para uma instalação existente (ex.: a que vem com o pytorch)
ROUTER_CUDA_PATH=".../site-packages/torch/lib" ROUTER_EP=cuda bun run index.ts

# ou copiar para src/onnx/bin/win32/x64 e não precisar de variável
cp .../cublas64_13.dll .../cublasLt64_13.dll .../cudnn*.dll src/onnx/bin/win32/x64/
```

São exigidos `cublas64_13.dll`, `cublasLt64_13.dll` e `cudnn64_9.dll` (esse último
puxa as demais `cudnn_*`). O cuDNN só é carregado no primeiro `Conv`, então a
checagem acontece na criação da sessão: sem ele a sessão subiria e quebraria só na
primeira inferência. Essas DLLs estão no `.gitignore`.

Só win32/x64 com placa NVIDIA; as outras plataformas seguem na CPU. Para conferir
a reescrita do modelo CPU:

```bash
python study/dequant_conv.py <modelo.onnx> <saida.onnx> --check
```

`win_delay_load_hook.cc` vem do node-gyp v12.1.0 (licença em
`src/native/LICENSE.node-gyp`) e permite carregar o mesmo addon no Node e no Bun.

</details>
