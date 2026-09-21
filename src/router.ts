import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import * as ort from './onnx/index.ts';
import { Tokenizer } from 'tokenizers'

import { ROUTER_CUDA_GRAPH_BUCKETS, ROUTER_CUDA_GRAPH_MODEL_FILE, ROUTER_MAX_LANES, ROUTER_MODEL_FILE, ROUTER_MODEL_FILE_FAST, ROUTER_MODEL_FILE_GPU, ROUTER_MODEL_ID, ROUTER_SEQ_LEN, ROUTER_THREADS } from "./constants.ts";

type Head = {
    kind: string;
    normalize: boolean;
    scale: number;
    bias: number;
    activation: string;
    prefix_heading: string;
    proj_dim: number;
};

export const getTokenizer = async (payload?: {
    token?: string
}) => {
    const tokenizerPath = await hfHubDownload({
        repo: ROUTER_MODEL_ID,
        filename: "tokenizer.json",
        token: payload?.token
    })

    return Tokenizer.fromFile(tokenizerPath)
}

export const effectiveCpus = (): number => {
    try {
        const raw = fs
            .readFileSync("/sys/fs/cgroup/cpu.max", "utf8")
            .trim()
            .split(/\s+/)

        if (raw.length >= 2 && raw[0] !== "max") {
            const quota = Number.parseInt(raw[0]!, 10)
            const period = Number.parseInt(raw[1]!, 10)

            if (
                Number.isFinite(quota) &&
                Number.isFinite(period) &&
                period > 0
            ) {
                return Math.max(1, Math.round(quota / period))
            }
        }
    } catch {
    }

    try {
        const quota = Number.parseInt(
            fs.readFileSync(
                "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                "utf8",
            ).trim(),
            10,
        )

        const period = Number.parseInt(
            fs.readFileSync(
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
                "utf8",
            ).trim(),
            10,
        )

        if (
            Number.isFinite(quota) &&
            Number.isFinite(period) &&
            quota > 0 &&
            period > 0
        ) {
            return Math.max(1, Math.floor(quota / period))
        }
    } catch {
    }

    return Math.max(1, os.cpus().length)
}

export const inferenceThreads = (): number => {
    const value = process.env.ROUTER_THREADS;

    if (value === undefined) {
        return Math.min(ROUTER_THREADS, effectiveCpus());
    }

    const threads = Number.parseInt(value, 10);

    if (!Number.isInteger(threads) || threads < 1) {
        throw new Error("ROUTER_THREADS must be a positive integer.");
    }

    return threads;
}

const hfHubDownload = async (
    { repo, filename, token }:
        {
            repo: string,
            filename: string,
            token?: string,
        }
): Promise<string> => {
    const cacheDir = path.join(
        process.env.HOME ?? process.cwd(),
        ".cache",
        "huggingface",
        "hub",
    )

    const outputPath = path.join(
        cacheDir,
        repo.replace("/", "--"),
        filename,
    )

    if (fs.existsSync(outputPath)) {
        return outputPath
    }

    const url =
        `https://huggingface.co/${repo}/resolve/main/${filename}`

    const response = await fetch(url, {
        headers: token
            ? {
                Authorization: `Bearer ${token}`,
            }
            : undefined,
    })

    if (!response.ok) {
        throw new Error(
            `Failed to download ${repo}/${filename}: ` +
            `${response.status} ${response.statusText}`,
        )
    }

    if (!response.body) {
        throw new Error(
            `Response body is not available for ${repo}/${filename}.`,
        )
    }

    fs.mkdirSync(path.dirname(outputPath), {
        recursive: true,
    })

    const total = Number(
        response.headers.get("content-length"),
    )

    let downloaded = 0
    let lastDraw = 0

    const file = fs.createWriteStream(outputPath)
    const reader = response.body.getReader()

    const drawProgress = () => {
        const now = Date.now()

        if (now - lastDraw < 50 && downloaded < total) {
            return
        }

        lastDraw = now

        const width = 30

        if (total > 0) {
            const progress = Math.min(
                downloaded / total,
                1,
            )

            const filled = Math.round(
                progress * width,
            )

            const bar =
                "█".repeat(filled) +
                "░".repeat(width - filled)

            const percent = Math.floor(
                progress * 100,
            )

            const downloadedMb = (
                downloaded /
                1024 /
                1024
            ).toFixed(1)

            const totalMb = (
                total /
                1024 /
                1024
            ).toFixed(1)

            process.stdout.write(
                `\r${filename} ` +
                `[${bar}] ` +
                `${percent.toString().padStart(3)}% ` +
                `${downloadedMb}/${totalMb} MB`,
            )
        } else {
            const downloadedMb = (
                downloaded /
                1024 /
                1024
            ).toFixed(1)

            process.stdout.write(
                `\r${filename} ` +
                `[${"█".repeat(width)}] ` +
                `${downloadedMb} MB`,
            )
        }
    }

    try {
        while (true) {
            const { done, value } = await reader.read()

            if (done) {
                break
            }

            file.write(Buffer.from(value))

            downloaded += value.byteLength

            drawProgress()
        }

        await new Promise<void>((resolve, reject) => {
            file.end((error: unknown) => {
                if (error) {
                    reject(error)
                } else {
                    resolve()
                }
            })
        })

        if (total > 0) {
            const bar = "█".repeat(30)

            process.stdout.write(
                `\r${filename} ` +
                `[${bar}] ` +
                `100% ` +
                `${(downloaded / 1024 / 1024).toFixed(1)}/` +
                `${(total / 1024 / 1024).toFixed(1)} MB`,
            )
        }

        process.stdout.write("\n")

        return outputPath
    } catch (error) {
        file.destroy()

        try {
            fs.unlinkSync(outputPath)
        } catch {
        }

        process.stdout.write("\n")

        throw error
    }
}

export const poolNormalized = ({
    weights,
    proj,
    count,
    projDim,
}: {
    weights: Float32Array
    proj: Float32Array
    count: number
    projDim: number
}): Float32Array => {
    const output = new Float32Array(projDim)

    for (let t = 0; t < count; t++) {
        const weight = weights[t]!

        if (weight === 0) {
            continue
        }

        for (let p = 0; p < projDim; p++) {
            output[p]! += weight * proj[t * projDim + p]!
        }
    }

    let norm = 0

    for (let p = 0; p < projDim; p++) {
        norm += output[p]! * output[p]!
    }

    norm = Math.max(Math.sqrt(norm), 1e-12)

    for (let p = 0; p < projDim; p++) {
        output[p]! /= norm
    }

    return output
}

const fromHub = async (
    token?: string,
    numThreads?: number,
    executionProviders?: string[],
) => {
    const baseModelPath = await hfHubDownload(
        {
            repo: ROUTER_MODEL_ID,
            filename: ROUTER_MODEL_FILE,
            token: token,
        }
    );

    const onnxDir = path.dirname(path.dirname(baseModelPath))
    const variant = (file: string) => path.join(onnxDir, file)

    // as contrib ops int8 so existem na CPU: em cuda o modelo precisa ser o fp32,
    // senao o ORT parte o grafo e fica mais lento que a propria CPU.
    const wantsGpu = (executionProviders ?? []).some(ep => ep !== "cpu")
    // GPU usa CUDA graph por padrao quando os buckets gerados estao presentes.
    // ROUTER_CUDA_GRAPH=0 mantem o caminho fp32 normal para diagnostico.
    const cudaGraph = wantsGpu && process.env.ROUTER_CUDA_GRAPH !== "0"

    if (wantsGpu && !fs.existsSync(variant(ROUTER_MODEL_FILE_GPU))) {
        throw new Error(
            `${ROUTER_MODEL_FILE_GPU} is missing. Run 'bun run build:model -- --gpu' to generate it.`,
        )
    }

    const cudaGraphModelPaths = cudaGraph
        ? Object.fromEntries(ROUTER_CUDA_GRAPH_BUCKETS.map((bucket) => [bucket, variant(ROUTER_CUDA_GRAPH_MODEL_FILE(bucket))]))
        : undefined

    if (cudaGraphModelPaths && Object.values(cudaGraphModelPaths).some((modelPath) => !fs.existsSync(modelPath))) {
        throw new Error(
            "CUDA graph models are missing. Run 'bun run build:model -- --gpu' to generate the fp16 fixed-shape buckets.",
        )
    }

    // ponytail: usa o modelo com conv float quando ja foi gerado; senao roda o do hub.
    const modelPath = wantsGpu
        ? variant(ROUTER_MODEL_FILE_GPU)
        : fs.existsSync(variant(ROUTER_MODEL_FILE_FAST))
            ? variant(ROUTER_MODEL_FILE_FAST)
            : baseModelPath;

    let head: Head | undefined;

    const configPath = await hfHubDownload(
        {
            repo: ROUTER_MODEL_ID,
            filename: "config.json",
            token,
        }
    );

    const config = JSON.parse(
        fs.readFileSync(configPath, "utf8"),
    );

    head = config.head;

    return {
        modelPath,
        cudaGraphModelPaths,
        numThreads,
        head,
    };
}

const model = async (payload: {
    modelPath: string,
    cudaGraphModelPaths?: Record<number, string>,
    numThreads?: number,
    head?: Head,
    executionProviders?: string[]
}) => {
    const threads = payload.numThreads ? payload.numThreads : inferenceThreads()

    if (!payload.head || payload.head.kind !== "cosine") {
        throw new Error("ONNX requires the model config's cosine head.")
    }

    const scale = Number(payload.head.scale)
    const bias = Number(payload.head.bias)

    const sessionOptions = {
        intraOpNumThreads: threads,
        interOpNumThreads: 1,
        executionMode: "sequential" as const,
        graphOptimizationLevel: "all" as const,
        enableMemPattern: true,
        enableCpuMemArena: true,
        executionProviders: payload.executionProviders ?? ["cpu"],
    }

    const sessions = payload.cudaGraphModelPaths
        ? new Map(await Promise.all(Object.entries(payload.cudaGraphModelPaths).map(async ([bucket, modelPath]) => [
            Number(bucket),
                await ort.createSession({ modelPath, options: { ...sessionOptions, enableCudaGraph: true } }),
            ] as const)))
            : new Map([[0, await ort.createSession({ modelPath: payload.modelPath, options: sessionOptions })]])

    const run = async (payload: {
        inputIds: ort.TensorObject,
        attentionMask: ort.TensorObject,
        textPool: Float32Array,
        categoryPool: Float32Array,
    }) => {
        const count = payload.inputIds.dims[1]!

        const bucket = payload.inputIds.dims[1]!
        const session = sessions.get(sessions.has(bucket) ? bucket : 0)
        if (!session) {
            throw new Error(`No CUDA graph session for ${bucket} tokens.`)
        }

        const outputs = await session.run({
            input_ids: payload.inputIds,
            attention_mask: payload.attentionMask,
        })

        const tokenProj = outputs.token_proj!
        const ruleProj = outputs.rule_proj!

        const projDim = tokenProj.dims[tokenProj.dims.length - 1]!

        const query = poolNormalized({
            weights: payload.textPool,
            proj: tokenProj.data as Float32Array,
            count,
            projDim,
        })

        const lanes = payload.categoryPool.length / ROUTER_SEQ_LEN
        const logits = new Float32Array(lanes)

        for (let l = 0; l < lanes; l++) {
            const lane = poolNormalized({
                weights: payload.categoryPool.subarray(
                    l * ROUTER_SEQ_LEN,
                    (l + 1) * ROUTER_SEQ_LEN,
                ),
                proj: ruleProj.data as Float32Array,
                count,
                projDim,
            })

            let sum = 0

            for (let p = 0; p < projDim; p++) {
                sum += lane[p]! * query[p]!
            }

            logits[l] = sum * scale + bias
        }

        return {
            output: logits,
        }
    }

    return Object.assign(run, {
        cudaGraphBuckets: payload.cudaGraphModelPaths ? [...ROUTER_CUDA_GRAPH_BUCKETS] : undefined,
    })
}

const prepareFromHub = async (
    { numThreads, token, executionProviders }: {
        numThreads?: number,
        token?: string,
        executionProviders?: string[],
    }
) => {
    const hub = await fromHub(token, numThreads, executionProviders)

    return model({ ...hub, executionProviders })
}


const prepareFromFile = async (
    { configPath, modelPath, numThreads, executionProviders }: {
        modelPath: string,
        configPath: string,
        numThreads?: number,
        executionProviders?: string[],
    }
) => {
    const config = JSON.parse(
        fs.readFileSync(configPath, "utf8"),
    );

    const head = config.head;

    return model({
        modelPath,
        head,
        numThreads,
        executionProviders
    })
}


export const Lfm2Router = {
    fromHub: prepareFromHub,
    fromFile: prepareFromFile
}

export const buildPrefix = (cats: string[]): string => {
    return `Categories:\n${cats.map((c) => `- ${c}`).join("\n")}\n\nText:\n`
}

export const catRanges = (cats: string[]): Array<[number, number]> => {
    const ranges: Array<[number, number]> = []

    let pos = "Categories:\n".length

    for (const cat of cats) {
        const start = pos + 2

        ranges.push([
            start,
            start + cat.length,
        ])

        pos = start + cat.length + 1
    }

    return ranges
}

export const buildInputs = async (
    text: string,
    cats: string[],
    tok: Tokenizer
) => {
    if (cats.length < 1 || cats.length > ROUTER_MAX_LANES) {
        throw new Error(
            `Provide between 1 and ${ROUTER_MAX_LANES} categories.`,
        )
    }

    if (!text.trim() || cats.some((cat) => !cat.trim())) {
        throw new Error(
            "Prompt and categories must not be empty.",
        )
    }

    const prefix = buildPrefix(cats)

    const encoding = await tok.encode(prefix + text)

    const ids = encoding.getIds()
    const offsets = encoding.getOffsets()

    const count = ids.length

    if (count > ROUTER_SEQ_LEN) {
        throw new Error(
            `Prompt and categories use ${count} tokens; ` +
            `the router allows at most ${ROUTER_SEQ_LEN}. ` +
            "Shorten the prompt or categories.",
        )
    }

    const inputIds = BigInt64Array.from(ids, BigInt)
    const attentionMask = new BigInt64Array(count).fill(1n)

    // ponytail: os pools ficam em stride ROUTER_SEQ_LEN -- poolNormalized e
    // "lanes = categoryPool.length / ROUTER_SEQ_LEN" dependem disso.
    const textPool = new Float32Array(ROUTER_SEQ_LEN)

    const categoryPool = new Float32Array(
        ROUTER_MAX_LANES * ROUTER_SEQ_LEN,
    )

    const meanPool = (
        row: Float32Array,
        indices: number[],
    ) => {
        if (indices.length === 0) {
            throw new Error(
                "Prompt and each category must contain at least one token.",
            )
        }

        const value = 1 / indices.length

        for (const index of indices) {
            row[index] = value
        }
    }

    const textIndices: number[] = []

    for (let i = 0; i < offsets.length; i++) {
        const [a, b] = offsets[i]!

        if (
            b! > prefix.length &&
            a !== b
        ) {
            textIndices.push(i)
        }
    }

    meanPool(textPool, textIndices)

    const ranges = catRanges(cats)

    for (let r = 0; r < ranges.length; r++) {
        const [start, end] = ranges[r]!

        const indices: number[] = []

        for (let i = 0; i < offsets.length; i++) {
            const [a, b] = offsets[i]!

            if (
                a! < end &&
                b! > start &&
                a !== b
            ) {
                indices.push(i)
            }
        }

        const row = categoryPool.subarray(
            r * ROUTER_SEQ_LEN,
            (r + 1) * ROUTER_SEQ_LEN,
        )

        meanPool(row, indices)
    }

    return {
        inputIds,
        attentionMask,
        textPool,
        categoryPool,
        count,
    }
}

export const route = async (
    { model, tok, text, cats }: {
        model: (payload: {
            inputIds: ort.TensorObject
            attentionMask: ort.TensorObject
            textPool: Float32Array
            categoryPool: Float32Array
        }) => Promise<{ output: Float32Array }>,
        tok: Tokenizer,
        text: string,
        cats: string[]
    }
) => {
    const feed = await buildInputs(text, cats, tok)
    const cudaGraphBuckets = (model as typeof model & { cudaGraphBuckets?: readonly number[] }).cudaGraphBuckets
    const bucket = cudaGraphBuckets?.find((value) => value >= feed.count)
    if (cudaGraphBuckets && !bucket) {
        throw new Error(`No CUDA graph bucket can hold ${feed.count} tokens.`)
    }

    const sequenceLength = bucket ?? feed.count
    const inputIds = sequenceLength === feed.count
        ? feed.inputIds
        : (() => {
            const padded = new BigInt64Array(sequenceLength)
            padded.set(feed.inputIds)
            return padded
        })()
    const attentionMask = sequenceLength === feed.count
        ? feed.attentionMask
        : (() => {
            const padded = new BigInt64Array(sequenceLength)
            padded.set(feed.attentionMask)
            return padded
        })()

    const outputs = await model({
        inputIds: ort.Tensor({
            type: "int64",
            data: inputIds,
            dims: [1, sequenceLength],
        }),
        attentionMask: ort.Tensor({
            type: "int64",
            data: attentionMask,
            dims: [1, sequenceLength],
        }),
        textPool: feed.textPool,
        categoryPool: feed.categoryPool,
    })

    const logits = Array.from(
        outputs.output.slice(0, cats.length),
        Number,
    )

    if (logits.some((x) => !Number.isFinite(x))) {
        throw new Error(
            "The model returned non-finite routing logits.",
        )
    }

    const max = Math.max(...logits)

    const ex = logits.map((x) =>
        Math.exp(x - max),
    )

    const total = ex.reduce(
        (sum, x) => sum + x,
        0,
    )

    const probs = ex.map(
        (x) => x / total,
    )

    const topIndex = probs.indexOf(
        Math.max(...probs),
    )

    return {
        probs,
        topIndex,
        count: feed.count,
    }
}
