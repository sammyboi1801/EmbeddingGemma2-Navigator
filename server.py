"""FastAPI backend for the EG2 Navigator web demo (Hugging Face Space, Docker SDK).

POST /chat    {"messages": [{"role": "user"|"assistant", "content": str}, ...],
               "temperature": 0.5, "top_k": 40, "max_new_tokens": 96}
              -> text/event-stream, one event per sampled token:
                 data: {"t": "<token text>", "p": <prob>, "c": <chunk index>}
                 ... and a final  data: {"done": true, "text": ..., "tokens": n, ...}
GET  /health  -> model info; the frontend polls it to wake a sleeping Space

Local:  uvicorn server:app --port 7860
Env:    EG2_CKPT (checkpoint path), ALLOWED_ORIGINS (comma-separated, default "*")
"""
import json
import os
import threading
import time
from typing import Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from eg2_autoregressive_decoding import ChatBot, _ids, encode

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("EG2_CKPT", os.path.join(HERE, "checkpoints", "chat.pt"))
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_TURNS, MAX_NEW = 20, 160  # public endpoint: bound the work per request

bot = ChatBot(CKPT)  # frozen encoder + navigator, loaded once
tok, a = bot.tok, bot.a
lock = threading.Lock()  # one generation at a time (the free CPU Space has 2 vCPUs)

app = FastAPI(title="EG2 Navigator", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=100)
    temperature: float = Field(0.5, ge=0.05, le=2.0)
    top_k: int = Field(40, ge=0, le=1000)
    max_new_tokens: int = Field(96, ge=1, le=MAX_NEW)


def _clean(text):
    return " ".join(text.split())  # one line per turn: "\n" ends a turn


def prompt_ids(messages):
    """Same format the model was trained on: 'User: ...\\nAssistant: ...\\n'."""
    ids = _ids(tok, "", bos=True)
    for m in messages:
        ids += _ids(tok, f"{'User' if m.role == 'user' else 'Assistant'}: {_clean(m.content)}\n")
    return ids + _ids(tok, "Assistant:")


@torch.no_grad()
def stream_tokens(ids, temp, topk, max_new):
    """Yield (token_id, prob, chunk) one token at a time. Every K tokens the frozen
    encoder re-reads the whole context (a new chunk); in between, only the navigator runs."""
    new, chunk = [], 0
    while len(new) < max_new:
        ctx = (ids + new)[-a.max_len:]
        t = torch.tensor([ctx], device=bot.device)
        mem = encode(bot.enc, t, torch.ones_like(t), a.enc_layer)  # re-read the map
        mem_len = torch.tensor([len(ctx)], device=bot.device)
        steps = []
        for _ in range(a.K):
            s = torch.tensor([steps], dtype=torch.long, device=bot.device)
            logits = bot.nav(mem, None, mem_len, s)[0][0, -1].float() / temp
            if topk > 0:
                kth = torch.topk(logits, min(topk, logits.numel())).values[-1]
                logits[logits < kth] = -float("inf")
            probs = F.softmax(logits, -1)
            j = torch.multinomial(probs, 1).item()
            tid = bot.sub_ids[j].item()
            if tid == tok.eos_token_id or "\n" in tok.decode([tid]):
                return  # end of the assistant's turn
            steps.append(tid)
            yield tid, probs[j].item(), chunk
            if len(new) + len(steps) >= max_new:
                return
        new += steps
        chunk += 1


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/")
def root():
    return {"name": "EG2 Navigator backend", "chat": "POST /chat", "health": "GET /health", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "device": bot.device, "chunk_tokens": a.K, "max_new_tokens": MAX_NEW,
            "navigator_params_m": round(sum(p.numel() for p in bot.nav.parameters()) / 1e6, 1)}


@app.post("/chat")
def chat(req: ChatRequest):
    msgs = req.messages[-MAX_TURNS:]
    if msgs[-1].role != "user" or not msgs[-1].content.strip():
        raise HTTPException(400, "the last message must be a non-empty user message")
    ids = prompt_ids(msgs)

    def events():
        with lock:  # released even if the client disconnects mid-stream
            t0, new, prev, chunks = time.time(), [], "", 1
            for tid, p, c in stream_tokens(ids, req.temperature, req.top_k, req.max_new_tokens):
                new.append(tid)
                text = tok.decode(new, skip_special_tokens=True)  # incremental decode keeps spaces right
                yield _sse({"t": text[len(prev):], "p": round(p, 4), "c": c})
                prev, chunks = text, c + 1
            yield _sse({"done": True, "text": prev.strip(), "tokens": len(new),
                        "encoder_passes": chunks, "seconds": round(time.time() - t0, 3)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
