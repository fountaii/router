// BitNet b1.58 decision model: one forward scores every option of a typed decision, plus abstention.
// Runtimes, picked automatically: CUDA through ONNX Runtime (model-gpu.onnx) when a GPU and its
// libraries are present; otherwise the native CPU runtime (src/native/bitnet_avx2.cc: ternary int8
// GEMMs, AVX-VNNI/AVX2) on x86-64 with AVX2; otherwise ONNX Runtime on CPU (model-cpu.onnx).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Tokenizer } from "tokenizers";

import { ensureCuda, ensureModel } from "./download.ts";
import { binding, cudaRuntimeDir } from "./onnx/binding.ts";
import { createSession, listSupportedBackends, Tensor, type Session } from "./onnx/index.ts";


export type DecisionInput = {
  question: string;
  state: string;
  choices: string[];
  instructions?: string;
  context?: string;
  history?: string;
};

export type Decision = {
  /** most probable choice, or null when abstention wins */
  choice: string | null;
  probabilities: Record<string, number>;
  abstain: number;
};

export type Device = "auto" | "cpu" | "cuda";

export type LoadOptions = {
  device?: Device;
  /** CPU runtime: "auto" (native when the CPU has AVX2, else ONNX Runtime), "native" or "onnx" */
  cpuRuntime?: "auto" | "native" | "onnx";
  /** CPU threads; defaults to half the logical CPUs, capped at 16 */
  threads?: number;
  /** longest prompt accepted, in tokens; default 16384 on the native runtime; ONNX Runtime (and CUDA) stops at
   * config.max_context (2048), its RoPE table. Attention reaches at most `window` positions back. */
  maxTokens?: number;
};

type Config = {
  bos_id: number; abstain: string; max_context: number; temperature: number; window?: number;
  architecture?: Record<string, number>; tensors?: Record<string, unknown>;
};

type Backend = {
  runtime: string;
  run(ids: Int32Array, positions: Int32Array, segments: Int32Array, markers: Int32Array): Promise<Float32Array>;
  release(): unknown;
};

const onnxBackend = (session: Session, runtime: string): Backend => {
  const int64 = (data: Int32Array) => Tensor({ type: "int64", data: BigInt64Array.from(data, BigInt) });
  return {
    runtime,
    run: async (ids, positions, segments, markers) => {
      const out = await session.run({
        input_ids: int64(ids), position_ids: int64(positions), segment_ids: int64(segments), markers: int64(markers),
      });
      return Object.values(out)[0]!.data as Float32Array;
    },
    release: () => session.release(),
  };
};

// Laya's descriptions for yes/no questions that come without criteria (decision_head.NOUL).
const NOUL: Record<string, string> = { false: "no, the statement does not hold", true: "yes, the statement holds" };

/** bitnet_large._flat: ASCII whitespace runs to one space, outer spaces stripped. */
export const flat = (value: string) => value.replace(/[ \t\r\n\f\v]+/g, " ").replace(/^ +| +$/g, "");

// Python's str() for JSON criteria values (decision_head encodes them with str()).
const pyStr = (value: unknown): string =>
  typeof value === "string" ? value : typeof value === "boolean" ? (value ? "True" : "False") : JSON.stringify(value);

// Python sorts strings by code point; JS compares UTF-16 units (differs only past U+FFFF).
const byCodePoint = (a: string, b: string) => {
  const x = Array.from(a, c => c.codePointAt(0)!), y = Array.from(b, c => c.codePointAt(0)!);
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i]! - y[i]!;
  return x.length - y.length;
};

/** decision_head.criteria: (question text, description per choice or null, question type). */
const criteria = (question: string, choices: string[]) => {
  const at = question.indexOf("\nCriteria: ");
  let text = question, table: Record<string, unknown> | null = null, levels = false;
  if (at >= 0) {
    text = question.slice(0, at);
    let parsed: unknown = null;
    try { parsed = JSON.parse(question.slice(at + "\nCriteria: ".length)); } catch { /* not mappable */ }
    levels = Array.isArray(parsed);
    if (levels) table = Object.fromEntries((parsed as unknown[]).map((v, i) => [String(i), v]));
    else if (parsed && typeof parsed === "object") table = parsed as Record<string, unknown>;
    if (!table || choices.some(c => !Object.hasOwn(table!, c))) [text, table, levels] = [question, null, false];
  }
  const noul = choices.length === 2 && choices.includes("false") && choices.includes("true");
  const type = levels ? "score" : noul ? "noul" : "choice";
  return { text, table: table ?? (type === "noul" ? NOUL : null), type };
};

export class DecisionModel {
  private readonly backend: Backend;
  private readonly tokenizer: Tokenizer;
  private readonly config: Config;
  readonly device: "cpu" | "cuda";

  /** longest prompt accepted, in tokens */
  readonly maxTokens: number;

  private constructor(backend: Backend, tokenizer: Tokenizer, config: Config, device: "cpu" | "cuda", maxTokens: number) {
    this.maxTokens = maxTokens;
    this.backend = backend;
    this.tokenizer = tokenizer;
    this.config = config;
    this.device = device;
  }

  /** "onnx-cuda", "native-avx-vnni", "native-avx2" or "onnx-cpu" */
  get runtime() {
    return this.backend.runtime;
  }

  /** dir: a local model directory; by default the files of fountaii/router, downloaded once to a cache
   * (DECISION_MODEL_DIR overrides it, DECISION_CACHE moves the cache). */
  static async load(dir?: string, { device = "auto", cpuRuntime = "auto", threads, maxTokens }: LoadOptions = {}): Promise<DecisionModel> {
    const envMax = process.env["DECISION_MAX_TOKENS"] ? Number(process.env["DECISION_MAX_TOKENS"]) : undefined;
    const limit = (fallback: number) => maxTokens ?? envMax ?? fallback;
    dir ??= process.env["DECISION_MODEL_DIR"] ?? await ensureModel();
    const config = JSON.parse(fs.readFileSync(path.join(dir, "config.json"), "utf8")) as Config;
    const tokenizer = Tokenizer.fromFile(path.join(dir, "tokenizer.json"));
    const wanted = (process.env["DECISION_DEVICE"] as Device | undefined) ?? device;
    if (wanted !== "cpu") await ensureCuda();  // first use with an NVIDIA GPU: ~700 MB of CUDA runtime
    const cuda = wanted !== "cpu" && cudaRuntimeDir() !== undefined &&
      listSupportedBackends().some(b => b.name === "cuda");
    if (wanted === "cuda" && !cuda) throw new Error("CUDA was requested but its provider or runtime libraries are missing.");
    if (cuda) {
      try {
        const session = await createSession({
          modelPath: path.join(dir, "model-gpu.onnx"),
          options: { executionProviders: ["cuda"], graphOptimizationLevel: "all" },
        });
        return new DecisionModel(onnxBackend(session, "onnx-cuda"), tokenizer, config, "cuda", Math.min(limit(config.max_context), config.max_context));
      } catch (error) {
        if (wanted === "cuda") throw error;  // auto: a GPU without enough memory or a driver falls back to CPU
      }
    }
    const nthreads = threads ?? defaultThreads();
    const wantedCpu = (process.env["DECISION_CPU_RUNTIME"] as LoadOptions["cpuRuntime"]) ?? cpuRuntime;
    const cpu = binding.bitnetCpu();
    if (wantedCpu !== "onnx" && cpu.supported && config.tensors && config.architecture) {
      const native = await binding.BitnetModel.load(path.join(dir, "weights.bin"), config.tensors,
        { ...config.architecture, max_context: limit(16384), window: config.window ?? config.max_context,
          exact_head: process.env["BITNET_EXACT_HEAD"] ? 1 : 0 }, nthreads);
      const backend = { runtime: `native-${cpu.kernel}`, run: native.run.bind(native), release: () => native.release() };
      return new DecisionModel(backend, tokenizer, config, "cpu", limit(16384));
    }
    if (wantedCpu === "native") throw new Error("The native CPU runtime needs an x86-64 CPU with AVX2 and FMA.");
    const session = await createSession({
      modelPath: path.join(dir, "model-cpu.onnx"),
      options: {
        intraOpNumThreads: nthreads,
        interOpNumThreads: 1,
        executionMode: "sequential",
        graphOptimizationLevel: "all",
        executionProviders: ["cpu"],
      },
    });
    return new DecisionModel(onnxBackend(session, "onnx-cpu"), tokenizer, config, "cpu", Math.min(limit(config.max_context), config.max_context));
  }

  /** decision_head.encode_case (head-3-branched): prompt ids, option markers in caller order
   * (abstention last) and the shared prefix length. Pieces are tokenized one by one. */
  async encode(input: DecisionInput) {
    const { choices } = input;
    const flatChoices = choices.map(flat);
    if (!choices.length || flatChoices.some(c => !c)) throw new Error("At least one nonempty choice is required");
    if (new Set(flatChoices).size !== flatChoices.length) throw new Error("Choices must be unique");
    if (flatChoices.includes(this.config.abstain)) throw new Error(`'${this.config.abstain}' is reserved for abstention`);
    const { text, table, type } = criteria(input.question, choices);
    const pieces = ([["I", input.instructions], [`Q (${type})`, text], ["H", input.history],
      ["C", input.context], ["T", input.state]] as const)
      .filter(([, v]) => v && flat(v)).map(([tag, v]) => `${tag}: ${flat(v!)}\n`);
    const first = pieces.length;
    const order = choices.map((_, i) => i).sort((a, b) => byCodePoint(flatChoices[a]!, flatChoices[b]!));
    for (const choice of [...order.map(i => choices[i]!), this.config.abstain]) {
      const about = table && choice !== this.config.abstain && Object.hasOwn(table, choice) ? table[choice] : undefined;
      pieces.push(about !== undefined && about !== null && about !== ""
        ? `- ${flat(choice)}: ${flat(pyStr(about))}\n` : `- ${flat(choice)}\n`);
    }
    const encoded = await Promise.all(pieces.map(p => this.tokenizer.encode(p, null, { addSpecialTokens: false })));
    const ids = [this.config.bos_id], physical: number[] = [];
    let prefix = 1;
    encoded.forEach((e, index) => {
      if (index === first) prefix = ids.length;
      ids.push(...e.getIds());
      if (index >= first) physical.push(ids.length - 1);
    });
    if (ids.length > this.maxTokens) {
      throw new Error(`Prompt needs ${ids.length} tokens and exceeds maxTokens (${this.maxTokens})`);
    }
    const markers = new Array<number>(physical.length);
    [...order, choices.length].forEach((original, index) => { markers[original] = physical[index]!; });
    return { ids, markers, prefix };
  }

  /** Raw logits for the choices, abstention last. */
  async logits(input: DecisionInput): Promise<Float32Array> {
    const { ids, markers, prefix } = await this.encode(input);
    const positions = new Int32Array(ids.length), segments = new Int32Array(ids.length);
    for (let i = 0; i < prefix; i++) positions[i] = i;
    let start = prefix, branch = 0;
    for (const marker of [...markers].sort((a, b) => a - b)) {  // branches in physical order
      branch++;
      for (let i = start; i <= marker; i++) {
        positions[i] = prefix + i - start;
        segments[i] = branch;
      }
      start = marker + 1;
    }
    return this.backend.run(Int32Array.from(ids), positions, segments, Int32Array.from(markers));
  }

  async decide(input: DecisionInput): Promise<Decision> {
    const logits = await this.logits(input);
    const scaled = Array.from(logits, v => v / this.config.temperature);
    const max = Math.max(...scaled);
    const exp = scaled.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    const p = exp.map(v => v / sum);
    const best = p.indexOf(Math.max(...p));
    return {
      choice: best === input.choices.length ? null : input.choices[best]!,
      probabilities: Object.fromEntries(input.choices.map((c, i) => [c, p[i]!])),
      abstain: p[input.choices.length]!,
    };
  }

  async release() {
    await this.backend.release();
  }
}

// int8 GEMM threads beyond the physical cores only contend (and hybrid E-cores are slower).
const defaultThreads = () => {
  const env = process.env["DECISION_THREADS"];
  if (env !== undefined) {
    const n = Number.parseInt(env, 10);
    if (!Number.isInteger(n) || n < 1) throw new Error("DECISION_THREADS must be a positive integer.");
    return n;
  }
  return Math.max(1, Math.min(16, Math.floor(os.availableParallelism() / 2)));
};
