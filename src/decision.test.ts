// Node runtime against the Python reference recorded in benchmark/parity.jsonl (60 benchmark cases):
// the same token ids and markers as the training code's encoder, and the same logits as ONNX Runtime in
// Python. Run: node --test src/decision.test.ts
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";

import { DecisionModel, flat } from "./decision.ts";

test("flat matches bitnet_large._flat", () => {
  assert.equal(flat("  a\t\nb  c \r\n"), "a b c");
  assert.equal(flat(" x "), " x ");  // ASCII whitespace only
});

test("encoding and logits match the Python reference", async () => {
  const cases = fs.readFileSync(path.join(import.meta.dirname, "../benchmark/parity.jsonl"), "utf8").trim().split("\n").map(l => JSON.parse(l));
  const model = await DecisionModel.load(undefined, { device: "cpu", cpuRuntime: "onnx" });
  let worst = 0;
  for (const { row, input_ids, markers, logits } of cases) {
    const input = { question: row.question, state: row.state, choices: row.choices,
      instructions: row.instructions, context: row.context, history: row.history };
    const encoded = await model.encode(input);
    assert.deepEqual(encoded.ids, input_ids);
    assert.deepEqual(encoded.markers, markers);
    const out = await model.logits(input);
    out.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - logits[i])); });
    assert.equal(out.indexOf(Math.max(...out)), logits.indexOf(Math.max(...logits)));
  }
  assert.ok(worst < 5e-3, `max logit difference ${worst}`);  // fp16 embeddings and head weights
  await model.release();
});

test("the native CPU runtime gives the reference's answers", async t => {
  const { binding } = await import("./onnx/binding.ts");
  if (!binding.bitnetCpu().supported) return t.skip("CPU without AVX2/FMA");
  const cases = fs.readFileSync(path.join(import.meta.dirname, "../benchmark/parity.jsonl"), "utf8").trim().split("\n").map(l => JSON.parse(l));
  const model = await DecisionModel.load(undefined, { device: "cpu", cpuRuntime: "native" });
  for (const { row, logits } of cases) {
    const out = await model.logits({ question: row.question, state: row.state, choices: row.choices,
      instructions: row.instructions, context: row.context, history: row.history });
    // int8 rounding of activations amplifies float summation-order differences: compare the answers
    assert.equal(out.indexOf(Math.max(...out)), logits.indexOf(Math.max(...logits)));
  }
  await model.release();
});
