// Downloads the model files from the Hugging Face hub once, into a local cache; later loads are offline.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export const MODEL_REPO = "fountaii/router";
export const MODEL_FILES = ["config.json", "tokenizer.json", "model-cpu.onnx", "model-gpu.onnx", "weights.bin"];

/** Directory holding every MODEL_FILES of `repo`, downloading what is missing. */
export const ensureModel = async (repo = MODEL_REPO, revision = "main"): Promise<string> => {
  const dir = process.env["DECISION_CACHE"] ??
    path.join(os.homedir(), ".cache", "bitnet-decision", repo.replace("/", "--"), revision);
  const marker = path.join(dir, ".complete");
  if (fs.existsSync(marker) && MODEL_FILES.every(f => fs.existsSync(path.join(dir, f)))) return dir;
  fs.mkdirSync(dir, { recursive: true });
  const token = process.env["HF_TOKEN"];
  const headers: Record<string, string> = token ? { authorization: `Bearer ${token}` } : {};
  for (const file of MODEL_FILES) {
    const target = path.join(dir, file);
    if (fs.existsSync(target)) continue;
    const url = `https://huggingface.co/${repo}/resolve/${revision}/${file}`;
    const response = await fetch(url, { headers });
    if (!response.ok || !response.body) throw new Error(`Download failed: ${url} (${response.status})`);
    const size = Number(response.headers.get("content-length") ?? 0);
    if (size > 50e6) console.error(`[@fountaii/router] downloading ${file} (${(size / 2 ** 20).toFixed(0)} MB) from ${repo}`);
    const partial = `${target}.partial`;
    await pipeline(Readable.fromWeb(response.body as import("node:stream/web").ReadableStream), fs.createWriteStream(partial));
    if (size && fs.statSync(partial).size !== size) throw new Error(`Incomplete download: ${file}`);
    fs.renameSync(partial, target);  // only complete files get the final name
  }
  fs.writeFileSync(marker, new Date().toISOString());
  return dir;
};
