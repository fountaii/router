// Accuracy and latency of the Node runtime on benchmark files (one case at a time, as in production).
//   node scripts/evaluate.ts benchmark/typed-decisions-test.jsonl benchmark/ag-news-test.jsonl
//   options: --device auto|cpu|cuda  --runtime auto|native|onnx (CPU)  --threads N  --limit N
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";

import { DecisionModel, type Device } from "../src/decision.ts";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    device: { type: "string", default: "auto" }, runtime: { type: "string", default: "auto" },
    threads: { type: "string" }, limit: { type: "string", default: "0" },
  },
});
const model = await DecisionModel.load(undefined, {
  device: values.device as Device,
  cpuRuntime: values.runtime as "auto" | "native" | "onnx",
  threads: values.threads ? Number(values.threads) : undefined,
});
const limit = Number(values.limit);
console.log(`device ${model.device}, runtime ${model.runtime}${values.threads ? `, ${values.threads} threads` : ""}`);

for (const file of positionals) {
  let rows = fs.readFileSync(file, "utf8").trim().split("\n").map(l => JSON.parse(l));
  if (limit) rows = rows.slice(0, limit);
  await model.decide(rows[0]);  // warm-up
  const times: number[] = [];
  let correct = 0;
  for (const row of rows) {
    const start = performance.now();
    const { choice } = await model.decide(row);
    times.push(performance.now() - start);
    correct += Number(choice === row.choices[row.target]);
  }
  times.sort((a, b) => a - b);
  const at = (q: number) => times[Math.min(times.length - 1, Math.floor(q * times.length))]!.toFixed(1);
  console.log(JSON.stringify({
    data: path.basename(file), correct, total: rows.length, accuracy: +(correct / rows.length).toFixed(4),
    p50_ms: +at(0.5), p95_ms: +at(0.95), mean_ms: +(times.reduce((a, b) => a + b, 0) / times.length).toFixed(1),
  }));
}
await model.release();
