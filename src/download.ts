// Downloads from the Hugging Face hub once: the model files into a local cache, and on Windows with an
// NVIDIA driver the CUDA runtime next to the bundled ONNX Runtime. Later loads are offline.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import { binDir } from "./onnx/binding.ts";

export const MODEL_REPO = "fountaii/router";
export const MODEL_FILES = ["config.json", "tokenizer.json", "model-cpu.onnx", "model-gpu.onnx", "weights.bin"];

const download = async (url: string, target: string) => {
  const token = process.env["HF_TOKEN"];
  const response = await fetch(url, { headers: token ? { authorization: `Bearer ${token}` } : {} });
  if (!response.ok || !response.body) throw new Error(`Download failed: ${url} (${response.status})`);
  const size = Number(response.headers.get("content-length") ?? 0);
  if (size > 50e6) console.error(`[@fountaii/router] downloading ${path.basename(target)} (${(size / 2 ** 20).toFixed(0)} MB)`);
  const partial = `${target}.partial`;
  await pipeline(Readable.fromWeb(response.body as import("node:stream/web").ReadableStream), fs.createWriteStream(partial));
  if (size && fs.statSync(partial).size !== size) throw new Error(`Incomplete download: ${url}`);
  fs.renameSync(partial, target);  // only complete files get the final name
};

/** The CUDA runtime this package's ONNX Runtime needs (Windows x64): its CUDA provider, cuBLAS and the
 * cuDNN parts the model uses, ~700 MB. */
export const CUDA_FILES = ["onnxruntime_providers_cuda.dll", "cublas64_13.dll", "cublasLt64_13.dll",
  "cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll"];

/** On Windows x64 with an NVIDIA driver, downloads the CUDA runtime next to onnxruntime.dll (once).
 * Returns whether CUDA can be used. */
export const ensureCuda = async (repo = MODEL_REPO, revision = "main"): Promise<boolean> => {
  if (process.platform !== "win32" || process.arch !== "x64") return false;
  const missing = CUDA_FILES.filter(f => !fs.existsSync(path.join(binDir, f)));
  if (!missing.length) return true;
  const driver = path.join(process.env["SystemRoot"] ?? "C:/Windows", "System32", "nvcuda.dll");
  if (!fs.existsSync(driver)) return false;  // no NVIDIA GPU driver: CPU
  for (const file of missing) {
    await download(`https://huggingface.co/${repo}/resolve/${revision}/${file}`, path.join(binDir, file));
  }
  process.env["PATH"] = [binDir, process.env["PATH"]].join(path.delimiter);
  return true;
};

/** Directory holding every MODEL_FILES of `repo`, downloading what is missing. */
export const ensureModel = async (repo = MODEL_REPO, revision = "main"): Promise<string> => {
  const dir = process.env["DECISION_CACHE"] ??
    path.join(os.homedir(), ".cache", "bitnet-decision", repo.replace("/", "--"), revision);
  const marker = path.join(dir, ".complete");
  if (fs.existsSync(marker) && MODEL_FILES.every(f => fs.existsSync(path.join(dir, f)))) return dir;
  fs.mkdirSync(dir, { recursive: true });
  for (const file of MODEL_FILES) {
    const target = path.join(dir, file);
    if (!fs.existsSync(target)) await download(`https://huggingface.co/${repo}/resolve/${revision}/${file}`, target);
  }
  fs.writeFileSync(marker, new Date().toISOString());
  return dir;
};
