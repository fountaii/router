<div align="center">

# @fountaii/router

**Typed decisions in one forward pass, with a 1.58-bit model that runs fast on a plain CPU.**

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-fountaii%2Frouter-yellow)](https://huggingface.co/fountaii/router)
![Size](https://img.shields.io/badge/download-333%20MB-blue)
![Runtime](https://img.shields.io/badge/runtime-Bun%20%7C%20Node-black)
![CPU](https://img.shields.io/badge/CPU-AVX--VNNI%20%2F%20AVX2-green)


</div>

---

`@fountaii/router` answers questions about a piece of state (text or JSON) by choosing among options you
pass at call time. Each option can carry a written criterion, and the model can always **abstain**.
It is a ternary [BitNet b1.58](https://arxiv.org/abs/2402.17764) model (20 blocks, 700M-class) with a
choice head, and the whole decision is a single forward pass.

- **Fast on CPU.** A native runtime with hand-written ternary int8 kernels (AVX-VNNI / AVX2): about 55 ms
  for a news article and 200 ms for a long JSON decision on a laptop i9.
- **Uses the GPU by itself** (CUDA) when one is available.
- **Small.** The weights are -1, 0 or +1, stored in 2 bits: a 333 MB download, cached after the first
  load.
- **Dynamic options.** No fixed label set: choices, yes/no and ordinal scores, each with its own criteria.

## Install

```bash
bun add github:fountaii/router        # Bun: runs the TypeScript sources
npm install github:fountaii/router    # Node ≥ 22.13: compiled to dist/ on install (also pnpm, yarn)
```

The model downloads from [fountaii/router](https://huggingface.co/fountaii/router) on the first
`load()` (333 MB, about 15 s) into `~/.cache/bitnet-decision/`. Later loads are offline.

## Quick start

```ts
import { DecisionModel } from "@fountaii/router";

const model = await DecisionModel.load();  // CUDA if available, else the native CPU runtime

const decision = await model.decide({
  question: "What is this news article about?",
  state: "Chipmaker shares jump after record quarterly profit on server demand.",
  choices: ["World", "Sports", "Business", "Sci/Tech"],
});

console.log(decision.choice);         // "Sci/Tech"
console.log(decision.probabilities);  // { World: 0.02, Sports: 0.00, Business: 0.44, "Sci/Tech": 0.54 }
console.log(decision.abstain);        // 0.00

await model.release();
```

## Usage guide

### The input

`decide(input)` takes one decision:

| Field | Type | |
| --- | --- | --- |
| `question` | `string` | What to decide. Criteria go after `"\nCriteria: "` as JSON (see below). |
| `state` | `string` | What to decide about: free text, or a JSON document as a string. |
| `choices` | `string[]` | The options. Non-empty and unique; `"none"` is reserved for abstention. |
| `instructions` | `string?` | Standing instructions for this kind of decision. |
| `context` | `string?` | Extra context. |
| `history` | `string?` | Earlier turns or events. |

The whole prompt must fit in 2,048 tokens; longer inputs throw.

### Question types

The criteria tell the model what each option means, and they decide the question type:

```ts
// choice: an object {label: description}
await model.decide({
  question: "What should the assistant do next?\nCriteria: " + JSON.stringify({
    answer_directly: "The assistant can resolve this with what it already knows.",
    escalate_to_human: "A human agent must take over.",
    execute_refund: "Issue the refund the customer is entitled to.",
    request_information: "Ask for what is missing before acting.",
  }),
  state: JSON.stringify({ conversation: "I was charged twice for order 1182. Refund the duplicate.", plan: "pro" }),
  choices: ["answer_directly", "escalate_to_human", "execute_refund", "request_information"],
});

// score: a list of levels, lowest first; the choices are their indexes
await model.decide({
  question: "How severe is the impact?\nCriteria: " + JSON.stringify(["none", "low", "high", "critical"]),
  state: "Production database unreachable for every customer since 09:12.",
  choices: ["0", "1", "2", "3"],
});

// yes/no: the choices "false" and "true" (criteria optional)
await model.decide({
  question: "This alert reflects genuinely malicious activity.",
  state: JSON.stringify({ alert: "impossible travel", user: "j.doe", mfa: true, known_vpn: true }),
  choices: ["false", "true"],
});

// plain classification: no criteria at all
await model.decide({ question: "What emotion does this text express?", state: "I can't believe they cancelled it again.",
  choices: ["anger", "joy", "sadness", "surprise"] });
```

### The result

```ts
type Decision = {
  choice: string | null;                  // the most probable option, or null when abstention wins
  probabilities: Record<string, number>;  // one per choice
  abstain: number;                        // probability of abstaining
};
```

The probabilities plus `abstain` sum to 1, calibrated with temperature 1.2. A useful pattern is to act
only on confident answers, for example `decision.choice && decision.probabilities[decision.choice] > 0.8`,
and send the rest to a human.

`model.logits(input)` returns the raw scores (choices in your order, abstention last), and
`model.encode(input)` returns the token ids.

### Loading options

```ts
const model = await DecisionModel.load(dir?, { device, cpuRuntime, threads });
```

| Option | Default | |
| --- | --- | --- |
| `dir` | downloaded model | A local model folder instead of the Hugging Face download. |
| `device` | `"auto"` | `"auto"`, `"cpu"` or `"cuda"`. `"auto"` uses CUDA when its libraries are present. |
| `cpuRuntime` | `"auto"` | `"native"` (x86-64 with AVX2), `"onnx"` (ONNX Runtime) or `"auto"`. |
| `threads` | min(16, logical CPUs / 2) | CPU threads. |

`model.device` (`"cpu"` / `"cuda"`) and `model.runtime` (`"native-avx-vnni"`, `"native-avx2"`,
`"onnx-cpu"`, `"onnx-cuda"`) tell what was picked.

| Environment variable | |
| --- | --- |
| `DECISION_DEVICE`, `DECISION_CPU_RUNTIME`, `DECISION_THREADS` | Same as the options above. |
| `DECISION_MODEL_DIR` | Use a local model folder. |
| `DECISION_CACHE` | Download cache folder (default `~/.cache/bitnet-decision/`). |
| `HF_TOKEN` | Token for a private model repository. |
| `ROUTER_CUDA_PATH` | Folder with the CUDA 13 cuBLAS and cuDNN 9 DLLs (Windows). |
| `BITNET_EXACT_HEAD=1` | Native runtime: run the choice head in fp32 instead of int8 (about 15% slower, same answers on the parity set). |

### Good to know

- **Load once, reuse.** Loading takes 2–3 s on CPU, longer on CUDA. `decide` is async; calls on one
  model run one at a time.
- **Call `release()`** when done, to free the ~1 GB the CPU runtime keeps in memory.
- **Offline.** `ensureModel()` (exported) downloads ahead of time, for example in a Docker build.

## Performance

Official test sets:

| Benchmark | ONNX Runtime CPU | CUDA | PyTorch reference |
| --- | ---: | ---: | ---: |
| [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), 2,000 decisions | 78.75% | 78.55% | 78.75% |
| AG News, 7,600 | 94.22% | 94.21% | 94.17% |
| Emotion validation, 2,000 | 92.55% | 92.55% | — |

The native CPU runtime gives the reference's answer on all of `benchmark/parity.jsonl`. Runtimes differ
by a few answers on the full sets: BitNet rounds activations to int8, so tiny float differences flip
near-ties.

Latency per decision, one decision at a time (i9-14900HX laptop with desktop apps running, RTX 4090
Laptop; CPU timings vary by about ±30% with background load):

| | Emotion (~60 tokens) | AG News (~90) | Typed Decisions (~315) |
| --- | ---: | ---: | ---: |
| Native CPU runtime, 16 threads | ~35 ms | ~55 ms | ~200 ms |
| Native CPU runtime, 1 thread | ~0.2 s | ~0.35 s | ~1.7 s |
| ONNX Runtime CPU, 16 threads | ~75 ms | ~105 ms | ~370 ms |
| CUDA | ~25 ms | ~25 ms | ~40 ms |

## Platform support

| Platform | Native CPU | ONNX CPU | CUDA |
| --- | :---: | :---: | :---: |
| Windows x64 | ✅ prebuilt | ✅ prebuilt | ✅ with the CUDA runtime DLLs¹ |
| Linux / macOS x64 | `bun run build:native`² | `build:native`² | Linux, `build:native`² |
| ARM64 | — | `build:native`² | — |

¹ `onnxruntime_providers_cuda.dll` next to `onnxruntime.dll` in `src/onnx/bin/win32/x64/`, plus cuBLAS and
cuDNN there or in `ROUTER_CUDA_PATH`. Without them, `"auto"` uses the CPU.
² Needs a C++ compiler and the ONNX Runtime shared library in `src/native/lib/<platform>/<arch>/`.

## How it works

- **Prompt.** Instructions, question, history, context and state form a shared prefix. Each option line
  (`- label: criterion`) is its own branch that reads the prefix and itself, so options never see each
  other. The upper 10 blocks read the prefix bidirectionally. A small transformer head scores the last
  token of each option line, plus an abstention line.
- **Native runtime** (`src/native/bitnet_avx2.cc`):
  - Every BitLinear is BitNet's own arithmetic: per-token absmax int8 activations times ternary weights,
    on 6×16 register tiles with `vpdpbusd`, with the residual add and SwiGLU fused into the GEMM.
  - RMSNorm is fused with the activation quantization.
  - Attention computes only the keys a query may see.
  - The choice head runs on the same int8 kernel (7-bit per-channel weights); its last layer runs only
    on the option rows.
  - A work-sharing pool balances hybrid P/E cores.
- **Where the time goes.** About 75% of a forward is the ternary GEMM, whose kernel matches ONNX
  Runtime's own int8 GEMM (MLAS) at ~0.47 TOPS per core, the AVX-VNNI ceiling of this CPU. Each token
  costs ~1.1 G multiply-adds, so a 315-token decision is ~360 GOP. A single core cannot go much below
  ~0.8 s for it; more cores, a GPU, or fewer tokens are what make it faster.
- **Model files.** The ternary layers are stored 2 bits per weight, losslessly; embeddings and head are
  fp16. One `weights.bin` serves the native runtime and both ONNX graphs, which unpack it once at load.

## Development

```bash
bun install
bun run build                    # dist/ for Node (runs on install too)
bun test --timeout 300000 src/decision.test.ts   # or: node --test src/decision.test.ts
bun run demo
bun run evaluate                 # accuracy and latency on benchmark/
bun run build:native             # rebuild the native binding
python scripts/pack-model.py <exported> <packed>   # 2-bit / fp16 packing of an exported model
```

