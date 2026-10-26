---
title: EG2 Navigator
emoji: 🧭
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Frozen EmbeddingGemma 2 + 10M navigator, token-by-token
---

# EG2 Navigator: a chatbot that generates with a frozen embedding model

Can a **frozen text-embedding model** be the "reader" of a language model, so that only a tiny
generator has to be trained? This repo answers that on a laptop GPU. A small trainable
**navigator** (10.5M params) does next-token prediction while cross-attending to the per-token
hidden states of a **frozen [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
encoder** (271M params). After about 1h45m of training on an RTX 4060 Laptop GPU it holds
simple, friendly multi-turn conversations.

```
User: Hi!                                     Bot: Hello! How can I help you today?
User: What should I cook for dinner tonight?  Bot: I'd recommend cooking. What are your favorite dishes?
User: Thanks, bye!                            Bot: You're welcome! I hope you enjoy your dinner.
```
More (including failures) in [results/demo_transcripts.md](results/demo_transcripts.md).

## How it works

```
 context so far ──► EmbeddingGemma 2 (frozen, bidirectional) ──► per-token states  "the map" (768-d)
                                                                        │ cross-attention
 start, t1 … t15 ─► Navigator (4-layer causal transformer, d=384) ◄─────┘
                         │
                         └─► W_out(h) · E[token]ᵀ  ──►  next token
                             (E = the encoder's own frozen 512-d token-embedding table)
```

* **Map:** the frozen encoder reads the whole conversation and returns per-token hidden states.
* **Steps:** the navigator generates **K = 16** tokens, self-attending causally to the tokens it
  has produced so far in that chunk and cross-attending to the map. A recency embedding tells it
  how far each map position is from the end of the context.
* **Re-localize:** after 16 tokens the encoder re-reads the full context. This gives one encoder
  pass per chunk, which is how a bidirectional encoder can drive left-to-right generation.
* **Tied output:** logits are dot products with the encoder's frozen token-embedding table,
  restricted to tokens that occur in the training data (about 44.5k of 262k).
* Chat format: `User: …\nAssistant: …\n`. Loss is computed on assistant turns only, and
  generation stops at the newline.

Only the navigator is trained; the encoder never receives gradients.

## Results

| | Stage 1: social chat | Stage 2: assistant fine-tune |
|---|---|---|
| Data | [SODA](https://huggingface.co/datasets/allenai/soda) (turns ≤ 30 words) + [DailyDialog](https://huggingface.co/datasets/OpenRL/daily_dialog) ×3 | [everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) ×4 + [Empathetic Dialogues](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm) |
| Train sequences | 203,511 | 28,571 |
| Steps × batch | ~22.4k × 32 | 3,000 × 32 |
| LR | 3e-4, cosine | 1e-4, cosine, warm start |
| Time (RTX 4060 Laptop) | ~91 min (0.237 s/step) | ~10 min |
| Val CE | 3.50 → **2.406** | 3.85 → **3.498** (everyday 3.55, empathetic 3.44) |

Val CE is not comparable across stages: the validation sets differ, and assistant replies carry
more content than small talk.

**What the stages do.** Stage 1 teaches fluent turn-taking, but SODA is role-play between two
named characters, so the bot sometimes casts itself into your story ("You're going to miss me
too, Annalisa"). Stage 2 replaces that with a consistent assistant/listener voice: greetings,
follow-up questions, supportive replies.

**Limitations.** It has very little world knowledge. "Why is the sky blue?" gets an on-topic but
meaningless answer about "light" and "colors". It deflects "Who are you?", and it can lose the
thread after 2–3 turns. The frozen encoder understands the question, but a 10M-parameter
generator has little to answer with. Use **temp 0.5** for the most coherent replies.

<details><summary>Validation curves</summary>

Stage 1 (steps 1k–7k from the first run; the run was then interrupted and warm-started from the
step-7k checkpoint at lr 2.4e-4, close to the original cosine schedule at that point, with fresh
optimizer state):

| step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 9k | 11k | 13k | 15k | 17k | 19k | 21k | 22.4k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val CE | 3.503 | 3.211 | 3.062 | 2.962 | 2.887 | 2.812 | 2.761 | 2.712 | 2.664 | 2.603 | 2.524 | 2.465 | 2.430 | 2.408 | 2.406 |

Stage 2:

| step | 250 | 500 | 750 | 1000 | 1500 | 2000 | 2500 | 2750 | 3000 |
|---|---|---|---|---|---|---|---|---|---|
| val CE | 3.848 | 3.744 | 3.684 | 3.625 | 3.559 | 3.516 | 3.500 | 3.498 | 3.498 |

Full logs are in [logs/](logs/).
</details>

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build of PyTorch
pip install -r requirements.txt

python eg2_autoregressive_decoding.py --chat --ckpt checkpoints/chat.pt --temp 0.5   # interactive
python chat_test.py checkpoints/chat.pt                                              # scripted test
```
```python
from eg2_autoregressive_decoding import ChatBot
bot = ChatBot("checkpoints/chat.pt", temp=0.5)
bot.say("Hi!"); bot.reset()
```
The encoder (about 0.6 GB) downloads from Hugging Face on first use. Inference works on CPU but is
faster on a GPU.

## Web demo: GitHub Pages frontend + Hugging Face Space backend

```
 GitHub Pages (docs/index.html)  ──POST /chat──►  HF Space (Dockerfile → server.py, FastAPI, CPU)
   chat UI + live token panel   ◄── SSE stream ──  one event per token: {"t": text, "p": prob, "c": chunk}
```
The page streams each reply into the chat and shows every sampled token, colored by model
confidence (hover for the probability). The underline color changes with each 16-token chunk, and
**⟳** marks every point where the frozen encoder re-reads the conversation. A playback-delay slider
slows the stream down so you can watch.

**Run locally**
```bash
uvicorn server:app --port 7860                       # backend
python -m http.server 8000 --directory docs          # frontend
# open http://127.0.0.1:8000/?api=http://127.0.0.1:7860
```

**1. Deploy the backend (Hugging Face Space, free CPU tier).** The YAML header at the top of this
README is the Space config (`sdk: docker`). Create a Docker Space named `eg2-navigator` (the
"Blank" template), then upload:
```bash
pip install -U huggingface_hub
hf auth login
hf upload <hf-user>/eg2-navigator . . --repo-type space \
    --exclude "checkpoints/archive/*" "logs/*" "results/*" "docs/*"
```
The Space builds the image, bakes the encoder in, and serves at
`https://<hf-user>-eg2-navigator.hf.space`. Check `/health`. `chat.pt` (40 MB) is stored with
Xet/LFS automatically. The Space must be **public** for the page to reach it.

**2. Deploy the frontend (GitHub Pages).** Set `window.EG2_API` in `docs/config.js` to your Space
URL. Push the repo, then go to Settings → Pages → Deploy from a branch → `main` / `/docs`.

**Notes**
* A free Space sleeps after about 48 h idle. The page shows "Waking the backend…" and retries
  until it is up (about 1 min).
* `ALLOWED_ORIGINS` (a Space variable) restricts CORS to your Pages origin, e.g.
  `https://<gh-user>.github.io`. The default is `*`.
* Requests are capped at 20 turns and 160 new tokens, and the server runs one generation at a
  time.

## Reproduce training

```bash
# smoke test (~2 min) to measure seconds/step
python eg2_autoregressive_decoding.py --data dialogue --daily 3 --steps 300 --eval_every 300 \
    --max_len 384 --n_stories 20000 --max_turn_words 30 --aux_weight 0 --ckpt test.pt

# stage 1 (~90 min on an RTX 4060 Laptop; STEPS = 5400 / seconds_per_step)
python eg2_autoregressive_decoding.py --data dialogue --daily 3 --steps 23143 --max_len 384 \
    --n_stories 300000 --max_turn_words 30 --aux_weight 0 --ckpt soda.pt

# stage 2 (~10 min)
python eg2_autoregressive_decoding.py --data chatmix --init soda.pt --steps 3000 --lr 1e-4 \
    --max_len 384 --eval_every 250 --aux_weight 0 --ckpt chat.pt
```
Notes:
* A checkpoint is written only when val CE improves, so `--ckpt` always holds the best model.
* Peak usage is about 5.5 GB VRAM and 6–8 GB system RAM (during SODA tokenization).
* In `chatmix`, val CE is the equal-weight mean of the everyday and empathetic sets, and both are
  printed.
* `--aux_weight 0` disables the optional "where will the chunk land" auxiliary loss, which halves
  encoder compute.

## Repository layout

```
eg2_autoregressive_decoding.py   model, data loaders, training, generation, ChatBot
server.py                        FastAPI backend: streams tokens over SSE (Hugging Face Space)
Dockerfile, requirements-server.txt   Space image (CPU PyTorch, encoder baked in)
docs/index.html, docs/config.js  static frontend for GitHub Pages
chat_test.py                     scripted 5-turn chat test
checkpoints/chat.pt              final model (navigator weights + vocab + args, ~40 MB)
logs/                            training logs for the smoke test, stage 1 (two parts) and stage 2
results/demo_transcripts.md      sample conversations from both stages
```

## Licenses and data

The encoder, `google/embeddinggemma-2`, is Apache-2.0. The training datasets have their own
licenses, and some (DailyDialog, Empathetic Dialogues) are **non-commercial**, so
`checkpoints/chat.pt` is for research and demo use only. See each dataset card for terms.
