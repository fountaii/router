export type TensorType =
  | "float32"
  | "int64"

export type TensorData = Float32Array | BigInt64Array;

export interface SessionOptions {
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
  executionMode?: "sequential" | "parallel";
  graphOptimizationLevel?: "disabled" | "basic" | "extended" | "all";
  executionProviders?: string[];
  enableCpuMemArena?: boolean;
  enableMemPattern?: boolean;
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
  enableProfiling?: boolean;
  enableCudaGraph?: boolean;
  profileFilePrefix?: string;
  configEntries?: Record<string, string>;
}

type ValueMetadata = {
  name: string
}

export type TensorObject = {
  data: TensorData
  type: TensorType
  dims: number[]
  size: number
  // the binding rejects any input tensor without it: "Tensor.location must be a string."
  location: "cpu"
}

export type TensorConstructor = {
  type: TensorType,
  data: TensorData,
  dims?: number[]
}

/** what `binding.InferenceSession` hands back — synchronous and blocking. */
export type NativeSession = {
  loadModel(modelPath: string, options: SessionOptions): void;
  inputMetadata: ValueMetadata[]
  outputMetadata: ValueMetadata[]
  run(
    feeds: Record<string, TensorObject>,
    fetches: Record<string, TensorObject | null>,
    options: Record<string, unknown>,
  ): Record<string, TensorObject>;
  endProfiling(): void;
  dispose(): void;
}

/** what `createSession` hands back. */
export type Session = {
  inputNames: string[]
  outputNames: string[]
  run(
    feeds: Record<string, TensorObject>,
    options?: Record<string, unknown>,
  ): Promise<Record<string, TensorObject>>
  endProfiling(): void
  release(): Promise<void>
}

/** the binding builds output tensors with `new Ctor(type, data, dims)`. */
export type NativeTensorCtor = new (
  type: TensorType,
  data: TensorData,
  dims: number[],
) => TensorObject

/** native CPU runtime of the BitNet decision model (src/native/bitnet_avx2.cc) */
export type NativeBitnet = {
  run(ids: Int32Array, positions: Int32Array, segments: Int32Array, markers: Int32Array): Promise<Float32Array>;
  release(): void;
}

export type OrtBinding = {
  BitnetModel: {
    load(weightsPath: string, tensors: Record<string, unknown>, config: Record<string, number>,
      threads: number): Promise<NativeBitnet>;
  };
  bitnetCpu: () => { supported: boolean; kernel: "avx-vnni" | "avx2" | "none" };
  InferenceSession: new () => NativeSession;
  listSupportedBackends: () => Array<{
    name: string;
    bundled: boolean;
  }>;
  initOrtOnce: (
    logLevel: number,
    tensorConstructor: NativeTensorCtor,
    isMainThread: boolean,
  ) => void;
};
