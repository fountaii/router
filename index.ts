import { inferenceThreads, Lfm2Router, route, getTokenizer } from "./src/router.ts"

async function main() {
    const HF_TOKEN = process.env["HF_TOKEN"]
    const CPUS = inferenceThreads()

    console.time("tokenizer")
    const TOK = await getTokenizer()
    console.timeEnd("tokenizer")

    // ROUTER_EP=cuda usa a GPU; precisa de `bun run build:model -- --gpu` antes.
    const executionProviders = (process.env["ROUTER_EP"] ?? "cuda").split(",")

    console.time("model")
    const model = await Lfm2Router.fromHub({
        numThreads: CPUS,
        token: HF_TOKEN,
        executionProviders
    })
    console.timeEnd("model")
    console.log("[boot] providers:", executionProviders.join(","))

    console.time("warmup")
    const warmup = await route({
        model,
        tok: TOK,
        text: "Set a timer.",
        cats: ["Simple tool use", "Creative writing"]
    })
    console.timeEnd("warmup")

    // Uma chamada isolada mede o caminho frio (JIT, clocks da GPU) e da ~1 ms a
    // mais que o regime. O p50 e o numero que vale.
    const samples: number[] = []
    let hot = warmup
    for (let i = 0; i < 100; i++) {
        const start = performance.now()
        hot = await route({
            model,
            tok: TOK,
            text: "Set a timer.",
            cats: ["Simple tool use", "Creative writing"]
        })
        samples.push(performance.now() - start)
    }
    samples.sort((a, b) => a - b)
    console.log(`route: p50 ${samples[50]!.toFixed(2)}ms p95 ${samples[95]!.toFixed(2)}ms`)

    console.log("[boot] router ready")
    console.log(hot)
}

main()
