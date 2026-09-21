"""Substitui o self_attn de UMA camada por um Lfm2ShortConv treinado (nao
inicializado do zero -- warm-start com os pesos de uma conv vizinha) pra
imitar a saida da atencao original via MSE, com o resto do modelo congelado.
E "opcao C" discutida: nao e mais cirurgia sem treino, e destilacao local
pequena (so o modulo novo tem gradiente).

Dataset sintetico: combinacoes de categorias + textos cobrindo varios
dominios de roteamento, no mesmo formato que buildPrefix/catRanges do
router.ts (`Categories:\\n- ...\\n\\nText:\\n<texto>`).

Uso: python study/distill_layer.py <camada> [--epochs N] [--lr LR]
"""

import argparse
import itertools
import random

import torch
import torch.nn as nn
import transformers

MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"

CATEGORIES = [
    "Simple tool use", "Creative writing", "Weather lookup", "Code generation",
    "Translation", "Math", "Restaurant reservation", "Customer support",
    "Summarization", "Image generation", "Account management", "Small talk",
    "Unit conversion", "Storytelling", "Flight search", "Recipe suggestion",
    "Account security", "Account billing", "Calendar management", "Task management",
    "Arithmetic", "Science explanation", "History explanation",
    "Document summarization", "Document translation", "Legal advice",
    "Medical advice", "Financial planning", "Shopping assistant", "News lookup",
    "Video editing", "Podcast transcription", "Resume review", "Interview prep",
    "Language learning", "Fitness coaching", "Meal planning", "Travel itinerary",
    "Home automation", "Car maintenance", "Pet care advice", "Gardening tips",
    "Music recommendation", "Movie recommendation", "Book recommendation",
    "Sports scores", "Stock price lookup", "Cryptocurrency lookup",
    "Data analysis", "SQL query generation", "Regex generation",
    "API documentation lookup", "Bug report triage", "Code review",
    "Meeting notes summarization", "Email drafting", "Social media post drafting",
    "Resume writing", "Cover letter writing", "Grammar correction",
    "Sentiment analysis", "Entity extraction", "Keyword extraction",
    "Product comparison", "Price tracking", "Order tracking",
    "Insurance claims", "Tax advice", "Real estate search",
    "Job search", "Salary negotiation advice", "Team scheduling",
    "Inventory management", "Supply chain lookup", "Customer feedback analysis",
    "Voice assistant control", "Smart home control", "Alarm and reminders",
    "Note taking", "File search", "Contact lookup",
]

TEXTS = [
    "Set a timer for 5 minutes.", "Write a short poem about the ocean at night.",
    "What's the weather in Tokyo right now?", "Write a Python function that sorts a list.",
    "Translate 'good morning' into French.", "What is the derivative of x^2 + 3x?",
    "Book me a table for two tonight at 8pm.", "My order hasn't arrived and it's been two weeks.",
    "Summarize this article in three bullet points.", "Generate an image of a cat riding a bicycle.",
    "Cancel my subscription immediately.", "How's your day going?",
    "Convert 100 fahrenheit to celsius.", "Tell me a bedtime story about a dragon.",
    "Find flights from Sao Paulo to Lisbon next Tuesday.", "Give me a recipe for chocolate chip cookies.",
    "Reset my password, I can't log in.", "Why was I charged twice this month?",
    "Schedule a meeting with the design team for Thursday.", "Add 'review the Q3 report' to my to-do list.",
    "What's 15% of 240?", "How many kilometers is 50 miles?",
    "Explain how photosynthesis works.", "Explain the causes of World War I.",
    "Summarize the key points of this contract.", "Translate this contract into Spanish.",
    "Can I sue my landlord for not returning my deposit?", "I've had a headache for three days, what could it be?",
    "How should I invest 10000 dollars for retirement?", "Find me a red dress under 50 dollars.",
    "What happened in the news today about the election?", "Remind me to call mom tomorrow at noon.",
    "Play some jazz music.", "What's the capital of Mongolia?",
    "Debug this JavaScript code, it's throwing a null reference error.",
    "Write a haiku about autumn leaves.", "Convert this CSV to JSON.",
    "What's the exchange rate between USD and EUR?", "Draft an email declining a job offer politely.",
    "How do I fix a leaking faucet?",
    "Trim the last 10 seconds off this video clip.", "Transcribe this podcast episode for me.",
    "Review my resume for a data scientist position.", "Give me common interview questions for a PM role.",
    "Teach me 10 useful phrases in Japanese.", "Build me a 4-day workout split for strength.",
    "Plan a week of vegetarian dinners for a family of four.", "Plan a 5-day itinerary for Kyoto.",
    "Turn off the living room lights at 10pm.", "When is my car's next oil change due?",
    "What should I feed a puppy that's 8 weeks old?", "Why are my tomato plants turning yellow?",
    "Recommend some albums similar to Kind of Blue.", "Recommend a movie like Inception.",
    "Recommend a sci-fi novel for a beginner.", "What was the score of last night's Lakers game?",
    "What's the current price of Apple stock?", "What's the price of Bitcoin right now?",
    "Find the average revenue per row in this spreadsheet.", "Write a SQL query to find duplicate emails.",
    "Write a regex that matches US phone numbers.", "How do I authenticate with the Stripe API?",
    "This bug only happens on Safari, triage it for me.", "Review this pull request for security issues.",
    "Summarize the key decisions from today's standup.", "Draft a follow-up email after a sales call.",
    "Write a LinkedIn post announcing our product launch.", "Rewrite my resume bullet points to be more concise.",
    "Write a cover letter for a marketing internship.", "Fix the grammar in this paragraph.",
    "Is this product review positive or negative?", "Extract all company names mentioned in this article.",
    "Pull out the five most important keywords from this text.", "Compare the iPhone 15 and the Galaxy S24.",
    "Alert me if this laptop drops below 800 dollars.", "Where is my package right now?",
    "Can I file a claim for water damage from a burst pipe?", "Can I deduct my home office on my taxes?",
    "Find 2-bedroom apartments for rent near downtown.", "Find me remote software engineering jobs.",
    "How do I ask for a 15% raise?", "Schedule the design review with the whole team next week.",
    "How many units of SKU 4821 are left in the warehouse?", "When will the next shipment from our supplier arrive?",
    "Summarize the negative feedback from this week's surveys.", "Turn on 'do not disturb' mode.",
    "Set the thermostat to 68 degrees.", "Wake me up at 6:30am tomorrow.",
    "Save this idea: launch a referral program next quarter.", "Find the quarterly report I saved last month.",
    "What's John's phone number?",
]


def build_prefix(cats: list[str]) -> str:
    return "Categories:\n" + "\n".join(f"- {c}" for c in cats) + "\n\nText:\n"


def synthetic_dataset(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        n_cats = rng.randint(1, 6)
        cats = rng.sample(CATEGORIES, n_cats)
        text = rng.choice(TEXTS)
        samples.append(build_prefix(cats) + text)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", type=int)
    ap.add_argument("--donor", type=int, default=None, help="camada conv pra warm-start (padrao: vizinha mais proxima)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.lfm2.layers
    target = layers[args.layer]
    if not target.is_attention_layer:
        raise SystemExit(f"camada {args.layer} ja e conv, nada pra destilar")

    donor_idx = args.donor
    if donor_idx is None:
        for delta in range(1, len(layers)):
            for cand in (args.layer - delta, args.layer + delta):
                if 0 <= cand < len(layers) and not layers[cand].is_attention_layer:
                    donor_idx = cand
                    break
            if donor_idx is not None:
                break
    print(f"camada alvo: {args.layer} (attn)  |  doadora pro warm-start: {donor_idx} (conv)")

    student = type(layers[donor_idx].conv)(model.config, layer_idx=args.layer).to(device)
    student.load_state_dict(layers[donor_idx].conv.state_dict())
    for p in student.parameters():
        p.requires_grad_(True)

    # captura (input, output) do self_attn da camada alvo via hook
    captured = {}
    def hook(_module, args, kwargs, output):
        captured["input"] = (args[0] if args else kwargs["hidden_states"]).detach()
        captured["output"] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = target.self_attn.register_forward_hook(hook, with_kwargs=True)

    texts = synthetic_dataset(args.n_samples)
    print(f"dataset sintetico: {len(texts)} amostras")

    pairs = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tok(text, return_tensors="pt").to(device)
            model.lfm2(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
            pairs.append((captured["input"].clone(), captured["output"].clone()))
            if (i + 1) % 100 == 0:
                print(f"  capturado {i + 1}/{len(texts)}")
    handle.remove()

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print("treinando...")
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        total = 0.0
        for x, y in pairs:
            opt.zero_grad()
            pred = student(hidden_states=x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            cos_sum, n = 0.0, 0
            with torch.no_grad():
                for x, y in pairs[:50]:
                    pred = student(hidden_states=x)
                    cos = torch.nn.functional.cosine_similarity(pred.flatten(), y.flatten(), dim=0)
                    cos_sum += cos.item(); n += 1
            print(f"  epoch {epoch:3d}  mse {total / len(pairs):.6f}  cosseno(pred,teacher) {cos_sum / n:.6f}")

    out_path = args.out or f".cache/perf/distilled_layer{args.layer}_from{donor_idx}.pt"
    torch.save(student.state_dict(), out_path)
    print(f"salvo: {out_path}")


if __name__ == "__main__":
    main()
