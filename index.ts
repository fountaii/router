// Demo: load the BitNet decision model (downloaded once from fountaii/router; CUDA when available,
// else the native CPU runtime) and decide two cases.
//   node index.ts            DECISION_DEVICE=cpu node index.ts
import { DecisionModel } from "./src/decision.ts";

const model = await DecisionModel.load();
console.log(`device: ${model.device}, runtime: ${model.runtime}`);

const cases = [
  {
    question: "What is this news article about?",
    state: "Chipmaker shares jump after record quarterly profit. Intel said on Tuesday that demand for server processors drove revenue above analysts' forecasts.",
    choices: ["World", "Sports", "Business", "Sci/Tech"],
  },
  {
    question: "What should the assistant do next with this conversation?\nCriteria: " + JSON.stringify({
      answer_directly: "The assistant can resolve this itself with information it already has.",
      escalate_to_human: "A human agent must take over.",
      execute_refund: "Issue the refund the customer is entitled to.",
      request_information: "Ask the customer for what is missing before acting.",
    }),
    state: JSON.stringify({
      conversation: "Customer: I was charged twice for order 1182 last week. Please refund the duplicate today.",
      account: { plan: "pro", tenure_months: 26, open_tickets: 0 },
    }),
    choices: ["answer_directly", "escalate_to_human", "execute_refund", "request_information"],
  },
];

for (const input of cases) {
  const start = performance.now();
  const decision = await model.decide(input);
  console.log(`${(performance.now() - start).toFixed(1)} ms`, decision);
}
await model.release();
