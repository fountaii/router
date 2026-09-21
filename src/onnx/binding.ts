import path from "node:path";
import fs from "node:fs";
import { createRequire } from "node:module";
import type { OrtBinding } from "./types.ts"

const binDir = path.resolve(
  import.meta.dirname,
  `./bin/${process.platform}/${process.arch}`,
);

// O provider de CUDA precisa do cuBLAS e do cuDNN, que nao sao embarcados (~900
// MB). Procura ao lado do binding e nos diretorios de ROUTER_CUDA_PATH; aumentar
// o PATH aqui basta porque o ORT so carrega essas bibliotecas quando a sessao
// pede cuda, bem depois deste modulo.
// O cuDNN e carregado tarde, no primeiro Conv, entao as duas familias sao
// exigidas juntas: senao a sessao sobe e so quebra na primeira inferencia.
export const CUDA_RUNTIME = ["cublas64_13.dll", "cublasLt64_13.dll", "cudnn64_9.dll"];

export const cudaRuntimeDir = (): string | undefined => {
  if (process.platform !== "win32") return undefined;

  return [binDir, ...(process.env["ROUTER_CUDA_PATH"] ?? "").split(path.delimiter)]
    .find(dir => dir && CUDA_RUNTIME.every(lib => fs.existsSync(path.join(dir, lib))));
};

const cudaDir = cudaRuntimeDir();
if (cudaDir) {
  process.env["PATH"] = [cudaDir, process.env["PATH"]].join(path.delimiter);
}

export const binding = createRequire(import.meta.url)(
  path.join(binDir, "onnxruntime_binding.node"),
) as OrtBinding;
