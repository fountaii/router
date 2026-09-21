import { isMainThread } from "node:worker_threads";

import type {
  NativeSession,
  NativeTensorCtor,
  Session,
  SessionOptions,
  TensorConstructor,
  TensorObject,
} from "./types.ts"
import { TYPE_TO_ARRAY } from "./constants.ts";
import { CUDA_RUNTIME, binding, cudaRuntimeDir } from "./binding.ts";

export type { Session, SessionOptions, TensorData, TensorObject, TensorType } from "./types.ts";

export const Tensor = ({ type, data, dims }: TensorConstructor): TensorObject => {
  const ctor = TYPE_TO_ARRAY[type];
  if (!(data instanceof ctor)) {
    throw new TypeError(`A ${type} tensor's data must be a ${ctor.name}.`);
  }

  const shape = dims ?? [data.length];
  const size = shape.reduce((a, b) => a * b, 1);
  if (size !== data.length) {
    throw new Error(`Tensor's size(${size}) does not match data length(${data.length}).`);
  }

  return { data, type, dims: shape, size, location: "cpu" };
};

const NativeTensor = function (
  this: TensorObject,
  ...args: ConstructorParameters<NativeTensorCtor>
) {
  const [type, data, dims] = args;
  Object.assign(this, Tensor({ type, data, dims }));
} as unknown as NativeTensorCtor;

export const listSupportedBackends = binding.listSupportedBackends;

let ortInitialized = false;

const wrap = (session: NativeSession): Session => {
  const inputNames = session.inputMetadata.map(m => m.name);
  const outputNames = session.outputMetadata.map(m => m.name);

  return {
    inputNames,
    outputNames,

    run: async (feeds, options = {}) => {
      for (const name of inputNames) {
        if (feeds[name] === undefined) throw new Error(`input '${name}' is missing in 'feeds'.`);
      }
      const fetches = Object.fromEntries(outputNames.map(n => [n, null]));

      return new Promise((resolve, reject) => {
        setImmediate(() => {
          try {
            resolve(session.run(feeds, fetches, options));
          } catch (e) {
            reject(e);
          }
        });
      });
    },

    endProfiling: () => session.endProfiling(),

    release: async () => session.dispose(),
  };
};

export const createSession = async ({
  modelPath,
  options = {},
}: {
  modelPath: string,
  options?: SessionOptions
}): Promise<Session> => {
  if (!ortInitialized) {
    ortInitialized = true;
    // 3 = error. Em cuda o ORT avisa, a cada sessao, que mandou os nos de shape
    // para a CPU — o que e a decisao certa dele: medido, forcar esses nos para a
    // GPU com shape fixo nao acelera nada (5.74 contra 5.65 ms). ROUTER_LOG_LEVEL
    // baixa de novo quando for preciso investigar.
    const logLevel = Number.parseInt(process.env["ROUTER_LOG_LEVEL"] ?? "3", 10);
    if (!Number.isInteger(logLevel) || logLevel < 0 || logLevel > 4) {
      throw new Error("ROUTER_LOG_LEVEL must be an integer from 0 to 4.");
    }
    binding.initOrtOnce(logLevel, NativeTensor, isMainThread);
  }

  // so pergunta quando alguem pede um provider alem do cpu, que e o padrao do
  // ORT: listar os backends carrega as bibliotecas deles (176 MB no caso do cuda).
  const requested = (options.executionProviders ?? []).filter(ep => ep !== "cpu");
  if (options.enableCudaGraph && !requested.includes("cuda")) {
    throw new Error("enableCudaGraph requires the cuda execution provider.");
  }
  if (requested.length) {
    const supported = new Set(binding.listSupportedBackends().map(b => b.name));
    for (const ep of requested) {
      if (!supported.has(ep)) throw new Error(`Unsupported execution provider: ${ep}.`);
    }

    if (requested.includes("cuda") && !cudaRuntimeDir()) {
      throw new Error(
        `The cuda provider needs ${CUDA_RUNTIME.join(", ")} next to the binding ` +
        `or in a ROUTER_CUDA_PATH directory. The CUDA 13 and cuDNN 9 runtimes are not bundled.`,
      );
    }
  }

  // native loadModel blocks; defer so callers keep the event loop.
  return new Promise((resolve, reject) => {
    setImmediate(() => {
      try {
        const session = new binding.InferenceSession();
        session.loadModel(modelPath, options);
        resolve(wrap(session));
      } catch (e) {
        reject(e);
      }
    });
  });
};
