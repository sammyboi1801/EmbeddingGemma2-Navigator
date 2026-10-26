"""
eg2_autoregressive_decoding.py: next-token generation with a FROZEN EmbeddingGemma 2
encoder plus a tiny trainable "navigator" network (see README.md for the chat recipe).

The idea, as navigation in a high-dimensional space:
  * Map: EmbeddingGemma 2 (frozen) reads the story so far and returns per-token
    hidden states, i.e. where the story currently sits in its 768-d space.
  * Steps: the navigator (a tiny causal transformer, ~10M params) takes K steps
    from there. It predicts the next K tokens, cross-attending to the frozen map
    and self-attending (causally) to the steps it has already taken.
  * Each step points into the encoder's own frozen token-embedding table:
    logits = W_out(h) . E[token]. Predicting a token = pointing at a region of
    the space the encoder already organized.
  * Re-localize: every `refresh` tokens the encoder re-reads the whole story and
    the map is rebuilt. This also sidesteps the bidirectional-encoder problem:
    one encoder pass per chunk, not per token.
  * Optional aux loss ("where will I land"): at the end of each chunk the
    navigator also predicts the encoder's mean-pooled embedding of the story
    *after* the chunk. Set --aux_weight 0 to disable (halves encoder compute).

Usage (full chat recipe and results in README.md)
  # stage 1: SODA + DailyDialog social chat
  python eg2_autoregressive_decoding.py --data dialogue --daily 3 --steps 23000 \
      --max_len 384 --n_stories 300000 --max_turn_words 30 --aux_weight 0 --ckpt soda.pt
  # stage 2: assistant / empathetic fine-tune
  python eg2_autoregressive_decoding.py --data chatmix --init soda.pt --steps 3000 \
      --lr 1e-4 --max_len 384 --eval_every 250 --aux_weight 0 --ckpt chat.pt
  # chat
  python eg2_autoregressive_decoding.py --chat --ckpt checkpoints/chat.pt --temp 0.5
  #   or: from eg2_autoregressive_decoding import ChatBot
  #       bot = ChatBot("checkpoints/chat.pt", temp=0.5); bot.say("hi!")
  # stories (the original TinyStories setup)
  python eg2_autoregressive_decoding.py --steps 20000 --ckpt navigator.pt
"""

import argparse
import math
import random
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "google/embeddinggemma-2"

# Windows: when stdout is redirected to a log file it defaults to cp1252 and block
# buffering, so a sampled emoji/CJK token crashes training and the log lags behind.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except AttributeError:
        pass

# bf16 only on Ampere or newer (A100, L4, RTX 30/40). T4 / V100 fall back to fp32;
# fp16 is avoided because Gemma-family activations can overflow in it.
USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
ENC_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32


# ----------------------------------------------------------------- encoder
def load_encoder(device):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    try:  # text-only: drop the vision and audio towers
        model = AutoModel.from_pretrained(
            MODEL_ID, vision_config=None, audio_config=None, dtype=ENC_DTYPE
        )
    except Exception as e:
        print(f"[warn] text-only load failed ({e}); loading the full model")
        model = AutoModel.from_pretrained(MODEL_ID, dtype=ENC_DTYPE)

    # If the top-level model is a multimodal wrapper, use only its text tower.
    text = model
    for parent in (model, getattr(model, "model", None)):
        if parent is None:
            continue
        for name in ("text_model", "language_model", "text_encoder"):
            sub = getattr(parent, name, None)
            if isinstance(sub, nn.Module):
                text = sub
                break
        if text is not model:
            break

    text = text.to(device).eval()
    for p in text.parameters():
        p.requires_grad_(False)
    print(f"[encoder] {type(text).__name__}, "
          f"{sum(p.numel() for p in text.parameters()) / 1e6:.0f}M frozen params")
    return tok, text


@torch.no_grad()
def encode(enc, ids, mask, layer):
    out = enc(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    h = hs[layer] if hs is not None else out.last_hidden_state
    return h.float()


def hidden_width(enc, layer, device):
    t = torch.ones((1, 4), dtype=torch.long, device=device)
    return encode(enc, t, torch.ones_like(t), layer).shape[-1]


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


# --------------------------------------------------------------- navigator
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mem, mem_pad, causal):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross_attn(h, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class Navigator(nn.Module):
    def __init__(self, E_full, sub_ids, K, D_hid, d=384, layers=4, heads=6, recency=64):
        super().__init__()
        D = E_full.shape[1]  # token-table width (can differ from hidden width D_hid)
        self.K = K
        self.E = E_full.detach()  # frozen encoder token table (plain tensor, not saved)
        self.register_buffer("E_out", E_full[sub_ids].detach().float(), persistent=False)
        self.bias = nn.Parameter(torch.zeros(len(sub_ids)))  # per-token frequency prior

        self.inp = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))      # steps taken
        self.mem_in = nn.Sequential(nn.Linear(D_hid, d), nn.LayerNorm(d))  # frozen map
        self.recency = nn.Embedding(recency, d)  # distance from end of prefix
        self.start = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, K, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)   # point into the encoder's token space
        self.aux = nn.Linear(d, D_hid)   # predict where the chunk lands
        self.register_buffer(
            "causal", torch.triu(torch.ones(K, K, dtype=torch.bool), 1), persistent=False
        )

    def forward(self, mem, mem_pad, mem_len, steps):
        """mem: (B,T,D) frozen prefix states; steps: (B,S) token ids taken so far
        in this chunk, S in [0, K-1]. Returns logits (B,S+1,Vsub), aux vec (B,D)."""
        B, T, _ = mem.shape
        dist = mem_len[:, None] - 1 - torch.arange(T, device=mem.device)[None]
        dist = dist.clamp(0, self.recency.num_embeddings - 1)
        m = self.mem_in(mem) + self.recency(dist)

        x = self.start.expand(B, 1, -1)
        if steps.shape[1] > 0:
            x = torch.cat([x, self.inp(self.E[steps].float())], dim=1)
        L = x.shape[1]
        x = x + self.pos[:, :L]
        for blk in self.blocks:
            x = blk(x, m, mem_pad, self.causal[:L, :L])
        h = self.norm(x)
        return self.out(h) @ self.E_out.T + self.bias, self.aux(h[:, -1])


# -------------------------------------------------------------------- data
# Every example is (ids, mask, starts):
#   mask[i] = 1 if token i is trained on (None = all tokens, i.e. stories)
#   starts  = positions where an assistant reply's text begins (dialogue only)

def load_stories(tok, split, n, max_len, min_len):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split)
    texts = [t for t in list(ds.select(range(min(n, len(ds))))["text"]) if isinstance(t, str) and t]
    bos, eos = tok.bos_token_id, tok.eos_token_id
    out = []
    for ids in tok(texts, add_special_tokens=False)["input_ids"]:
        ids = ([bos] if bos is not None else []) + ids[: max_len - 2] + ([eos] if eos is not None else [])
        if len(ids) >= min_len:
            out.append((ids, None, None))
    return out


def pack_conversations(tok, convs, max_len, min_len):
    """convs: list of conversations, each a list of (is_assistant, text).
    Rendered as 'User: ...\nAssistant: ...\n', loss only on Assistant turns."""
    turns, owner = [], []
    for i, conv in enumerate(convs):
        for is_asst, text in conv:
            # collapse internal newlines: "\n" is the end-of-turn marker the chat stops on
            turns.append(f"{'Assistant' if is_asst else 'User'}: {' '.join(str(text).split())}\n")
            owner.append((i, is_asst))
    turn_ids = tok(turns, add_special_tokens=False)["input_ids"]
    plen = len(tok("Assistant:", add_special_tokens=False)["input_ids"])
    bos, eos = tok.bos_token_id, tok.eos_token_id

    out = []
    cur, ids, mask, starts = None, [], [], []

    def flush():
        if not ids:
            return
        full = ([bos] if bos is not None else []) + ids + ([eos] if eos is not None else [])
        m = ([0] if bos is not None else []) + mask + ([0] if eos is not None else [])
        off = 1 if bos is not None else 0
        full, m = full[:max_len], m[:max_len]
        st = [x + off for x in starts if x + off < len(full)]
        if len(full) >= min_len and sum(m) > 0:
            out.append((full, m, st))

    for (i, is_asst), t in zip(owner, turn_ids):
        if i != cur:
            flush()
            cur, ids, mask, starts = i, [], [], []
        if is_asst:
            starts.append(len(ids) + plen)  # first token after "Assistant:"
        ids += t
        mask += [1 if is_asst else 0] * len(t)
    flush()
    return out


def _detok(t):
    """DailyDialog text is pre-tokenized: 'Say , Jim , how about it ? '."""
    t = str(t).replace("’", "'").replace("‘", "'")
    t = re.sub(r"\s+([,.!?;:%)])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"(\w) ' (s|t|re|ve|ll|d|m)\b", lambda m: f"{m[1]}'{m[2].lower()}", t, flags=re.I)
    t = re.sub(r"([a-z])([.!?])([A-Z])", r"\1\2 \3", t)  # "right.But" -> "right. But"
    return t.strip()


def _daily_convs(split, max_turn_words):
    """DailyDialog (parquet mirror): 11k human-written everyday two-person chats.
    Speakers alternate; the first one is the User."""
    from datasets import load_dataset
    ds = load_dataset("OpenRL/daily_dialog", split=split)
    convs = []
    for d in list(ds["dialog"]):
        d = [_detok(t) for t in d if str(t).strip()]
        if len(d) < 2:
            continue
        if max_turn_words and max(len(t.split()) for t in d) > max_turn_words:
            continue
        convs.append([(i % 2 == 1, t) for i, t in enumerate(d)])
    return convs


def load_dialogues(tok, split, n, max_len, min_len, max_turn_words=0, daily=0):
    """SODA two-person social dialogues. The first speaker is the User.
    daily > 0 mixes in DailyDialog, repeated `daily` times (train only)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/soda", split=split)
    ds = ds.select(range(min(n, len(ds))))
    convs = []
    for d, s in zip(list(ds["dialogue"]), list(ds["speakers"])):
        if not d or not s or len(d) != len(s) or len(set(s)) != 2:
            continue
        if max_turn_words and max(len(str(t).split()) for t in d) > max_turn_words:
            continue  # keep only snappy, chat-like dialogues
        convs.append([(who != s[0], text) for text, who in zip(d, s)])
    if daily > 0:
        dd = _daily_convs(split, max_turn_words)
        print(f"[data] dailydialog: {len(dd)} conversations x{daily if split == 'train' else 1}")
        convs += dd * (daily if split == "train" else 1)
    return pack_conversations(tok, convs, max_len, min_len)


def _role_convs(name, split, column):
    """Datasets stored as lists of {"role", "content"} messages."""
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    convs = []
    for msgs in list(ds[column]):
        conv = [(m["role"] == "assistant", m["content"]) for m in msgs
                if m.get("role") in ("user", "assistant") and m.get("content")]
        if conv:
            convs.append(conv)
    return convs


def load_everyday(tok, split, max_len, min_len):
    """HuggingFaceTB/everyday-conversations-llama3.1-2k: short User <-> AI assistant
    chats about everyday topics and simple science."""
    convs = _role_convs("HuggingFaceTB/everyday-conversations-llama3.1-2k", split, "messages")
    return pack_conversations(tok, convs, max_len, min_len)


def load_chat_mix(tok, split, max_len, min_len, everyday_repeat=4):
    """everyday-conversations (repeated, it is only 2.2k chats) + Empathetic Dialogues
    (19.5k short, warm user <-> listener chats). More data than everyday alone, so the
    fine-tune overfits later."""
    everyday = _role_convs("HuggingFaceTB/everyday-conversations-llama3.1-2k",
                           "train_sft" if split == "train" else "test_sft", "messages")
    empathetic = _role_convs("Estwld/empathetic_dialogues_llm", split, "conversations")
    reps = everyday_repeat if split == "train" else 1
    return pack_conversations(tok, everyday * reps + empathetic, max_len, min_len)


def pad(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    lens = torch.tensor([len(s) for s in seqs])
    return ids.to(device), mask.to(device), lens.to(device)


def make_batch(seqs, B, K, min_prefix, pad_id, device, rng):
    pre, chunk, cmask, full = [], [], [], []
    for ids, m, starts in rng.sample(seqs, B):
        if m is None:  # stories: any cut point
            b = rng.randint(min_prefix, len(ids) - K)
        else:          # dialogue: start inside an assistant reply
            hi = len(ids) - 1
            good = [x for x in starts if min_prefix <= x <= hi]
            if good and rng.random() < 0.5:
                b = rng.choice(good)  # the exact case used when chatting
            else:
                cand = [i for i in range(min_prefix, hi + 1) if m[i]]
                b = rng.choice(cand) if cand else min(min_prefix, hi)
        c = ids[b : b + K]
        cm = [1] * len(c) if m is None else m[b : b + K]
        npad = K - len(c)
        pre.append(ids[:b]); full.append(ids[: b + K])
        chunk.append(c + [pad_id] * npad); cmask.append(cm + [0] * npad)
    return (pad(pre, pad_id, device), torch.tensor(chunk, device=device),
            torch.tensor(cmask, device=device, dtype=torch.float), pad(full, pad_id, device))


# ------------------------------------------------------------------- train
def step_loss(enc, nav, batch, full2sub, a, report_pos=False):
    (pid, pmask, plen), chunk, cmask, (fid, fmask, _) = batch
    mem = encode(enc, pid, pmask, a.enc_layer)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
        logits, z = nav(mem, pmask == 0, plen, chunk[:, :-1])
    tok_ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             full2sub[chunk].reshape(-1), reduction="none").view(chunk.shape)
    ce = (tok_ce * cmask).sum() / cmask.sum().clamp(min=1)
    aux = torch.zeros((), device=ce.device)
    if a.aux_weight > 0:
        target = F.normalize(mean_pool(encode(enc, fid, fmask, -1), fmask), dim=-1)
        aux = 1 - F.cosine_similarity(z.float(), target, dim=-1).mean()
    pos = (tok_ce * cmask).sum(0) / cmask.sum(0).clamp(min=1) if report_pos else None
    return ce, aux, pos


def build_navigator(enc, sub_ids, a, device, init=None):
    E = enc.get_input_embeddings().weight
    D_hid = hidden_width(enc, a.enc_layer, device)
    nav = Navigator(E, sub_ids, a.K, D_hid, a.d, a.layers, a.heads).to(device)
    if init is not None:  # warm start; vocab may have grown, so remap the bias
        sd = dict(init["nav"])
        old_bias = sd.pop("bias").to(device)
        nav.load_state_dict(sd, strict=False)
        where = {t: j for j, t in enumerate(sub_ids.tolist())}
        with torch.no_grad():
            for j, t in enumerate(init["sub_ids"].tolist()):
                if t in where:
                    nav.bias[where[t]] = old_bias[j]
    return nav, E, D_hid


def save_ckpt(ck, path):
    try:
        torch.save(ck, path)
    except Exception as e:  # e.g. Drive not mounted: keep going, save locally
        print(f"  [warn] could not save to {path} ({e}); saving to navigator.pt instead")
        torch.save(ck, "navigator.pt")


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = None
    if a.init:
        init = torch.load(a.init, map_location="cpu")
        for k in ("K", "d", "layers", "heads", "enc_layer"):
            setattr(a, k, init["args"][k])  # architecture must match the checkpoint
        print(f"[init] warm start from {a.init}")

    tok, enc = load_encoder(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    val_parts = None  # optional {name: seqs}: report val ce per source, average them equally
    if a.data == "dialogue":
        train_seqs = load_dialogues(tok, "train", a.n_stories, a.max_len, a.min_prefix + 2,
                                    a.max_turn_words, a.daily)
        val_seqs = load_dialogues(tok, "validation", 2000, a.max_len, a.min_prefix + 2,
                                  a.max_turn_words, a.daily)
        sample_prompt = "User: Hi! How was your day?\n"
    elif a.data == "everyday":
        train_seqs = load_everyday(tok, "train_sft", a.max_len, a.min_prefix + 2)
        val_seqs = load_everyday(tok, "test_sft", a.max_len, a.min_prefix + 2)
        sample_prompt = "User: Hi! How was your day?\n"
    elif a.data == "chatmix":
        train_seqs = load_chat_mix(tok, "train", a.max_len, a.min_prefix + 2)
        # val kept per source: pooled, it would be ~96% empathetic (119 vs 2770 chats)
        val_parts = {
            "everyday": load_everyday(tok, "test_sft", a.max_len, a.min_prefix + 2),
            "empathetic": pack_conversations(
                tok, _role_convs("Estwld/empathetic_dialogues_llm", "valid", "conversations"),
                a.max_len, a.min_prefix + 2),
        }
        val_seqs = [s for p in val_parts.values() for s in p]
        sample_prompt = "User: Hi! How was your day?\n"
    else:
        min_len = a.min_prefix + a.K
        train_seqs = load_stories(tok, "train", a.n_stories, a.max_len, min_len)
        val_seqs = load_stories(tok, "validation", 2000, a.max_len, min_len)
        sample_prompt = None
    print(f"[data] {a.data}: {len(train_seqs)} train / {len(val_seqs)} val")

    # Output vocab = tokens in the data (plus any from the warm-start checkpoint).
    used = {t for s, _, _ in train_seqs for t in s}
    if init is not None:
        used |= set(init["sub_ids"].tolist())
    sub_ids = torch.tensor(sorted(used), device=device)
    V = enc.get_input_embeddings().weight.shape[0]
    full2sub = torch.zeros(V, dtype=torch.long, device=device)
    full2sub[sub_ids] = torch.arange(len(sub_ids), device=device)

    nav, E, D_hid = build_navigator(enc, sub_ids, a, device, init)
    print(f"[vocab] {len(sub_ids)} of {V} tokens | [dims] table {E.shape[1]}-d, hidden {D_hid}-d")
    print(f"[navigator] {sum(p.numel() for p in nav.parameters()) / 1e6:.1f}M trainable params")

    opt = torch.optim.AdamW(nav.parameters(), lr=a.lr, weight_decay=0.05)
    warm = 500 if init is None else 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))
    )
    rng, vrng = random.Random(0), random.Random(1)
    parts = val_parts or {"all": val_seqs}
    val_batches = {k: [make_batch(p, a.batch, a.K, a.min_prefix, pad_id, device, vrng)
                       for _ in range(20 // len(parts))] for k, p in parts.items()}

    t0, best = time.time(), float("inf")
    for step in range(1, a.steps + 1):
        nav.train()
        batch = make_batch(train_seqs, a.batch, a.K, a.min_prefix, pad_id, device, rng)
        ce, aux, _ = step_loss(enc, nav, batch, full2sub, a)
        loss = ce + a.aux_weight * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nav.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            print(f"step {step:6d} | ce {ce.item():.3f} (ppl {math.exp(ce.item()):.1f}) "
                  f"| aux {aux.item():.3f} | {time.time() - t0:.0f}s")
        if step % a.eval_every == 0 or step == a.steps:
            nav.eval()
            with torch.no_grad():
                res = {k: [step_loss(enc, nav, b, full2sub, a, report_pos=a.pos_ce) for b in bs]
                       for k, bs in val_batches.items()}
            per = {k: sum(r[0].item() for r in rs) / len(rs) for k, rs in res.items()}
            v = sum(per.values()) / len(per)
            split = " | " + " | ".join(f"{k} {x:.3f}" for k, x in per.items()) if len(per) > 1 else ""
            print(f"  [val] ce {v:.3f} (ppl {math.exp(v):.1f}){split}")
            if a.pos_ce:
                pos = torch.stack([r[2] for rs in res.values() for r in rs]).mean(0)
                print("  [pos ce]", " ".join(f"{x:.2f}" for x in pos.tolist()))
            if sample_prompt is None:
                print("  [sample]", generate(enc, nav, tok, sub_ids, "Once upon a time", a, device, 80))
            else:
                ids = _ids(tok, sample_prompt, bos=True) + _ids(tok, "Assistant:")
                reply = continue_ids(enc, nav, sub_ids, ids, a, device, 60, stop=_chat_stop(tok))
                print("  [sample] User: Hi! How was your day? | Assistant:",
                      tok.decode(reply, skip_special_tokens=True).split("\n")[0].strip())
            # keep the best-val checkpoint, so stopping after val ce rises loses nothing
            if v < best:
                best = v
                save_ckpt({"nav": nav.state_dict(), "sub_ids": sub_ids.cpu(), "args": vars(a)}, a.ckpt)
                print(f"  [ckpt] saved {a.ckpt} (best val ce {v:.3f})")


# ---------------------------------------------------------------- generate
def _ids(tok, text, bos=False):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ([tok.bos_token_id] if bos and tok.bos_token_id is not None else []) + ids


def _chat_stop(tok):
    return lambda new: new[-1] == tok.eos_token_id or "\n" in tok.decode(new)


@torch.no_grad()
def continue_ids(enc, nav, sub_ids, ids, a, device, max_new, stop=None):
    """Sample up to max_new tokens after `ids`, re-encoding every `refresh` tokens."""
    nav.eval()
    refresh = min(a.refresh or a.K, a.K)
    new = []
    while len(new) < max_new:
        ctx = (ids + new)[-a.max_len:]
        t = torch.tensor([ctx], device=device)
        mem = encode(enc, t, torch.ones_like(t), a.enc_layer)   # re-localize on the map
        mem_len = torch.tensor([len(ctx)], device=device)
        steps, done = [], False
        for _ in range(refresh):
            s = torch.tensor([steps], dtype=torch.long, device=device)
            logits, _ = nav(mem, None, mem_len, s)
            logits = logits[0, -1].float() / a.temp
            if a.topk > 0:
                kth = torch.topk(logits, a.topk).values[-1]
                logits[logits < kth] = -float("inf")
            steps.append(sub_ids[torch.multinomial(F.softmax(logits, -1), 1)].item())
            if stop is not None and stop(new + steps):
                done = True
                break
        new += steps
        if done:
            break
    return new


def generate(enc, nav, tok, sub_ids, prompt, a, device, n_tokens):
    new = continue_ids(enc, nav, sub_ids, _ids(tok, prompt, bos=True), a, device, n_tokens,
                       stop=lambda s: s[-1] == tok.eos_token_id)
    return tok.decode(new, skip_special_tokens=True).replace("\n", " ")


class ChatBot:
    """Notebook chat:  bot = ChatBot("chat.pt");  bot.say("hello");  bot.reset()"""

    def __init__(self, ckpt, temp=0.7, topk=40, refresh=0, max_new=60):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(ckpt, map_location="cpu")
        self.a = argparse.Namespace(**ck["args"])
        self.a.temp, self.a.topk, self.a.refresh = temp, topk, refresh
        self.tok, self.enc = load_encoder(self.device)
        self.sub_ids = ck["sub_ids"].to(self.device)
        self.nav, _, _ = build_navigator(self.enc, self.sub_ids, self.a, self.device)
        self.nav.load_state_dict(ck["nav"])
        self.nav.eval()
        self.max_new = max_new
        self.reset()

    def reset(self):
        self.history = _ids(self.tok, "", bos=True)

    def say(self, msg):
        self.history += _ids(self.tok, f"User: {msg.strip()}\n")
        prompt = self.history + _ids(self.tok, "Assistant:")
        new = continue_ids(self.enc, self.nav, self.sub_ids, prompt, self.a, self.device,
                           self.max_new, stop=_chat_stop(self.tok))
        reply = self.tok.decode(new, skip_special_tokens=True).split("\n")[0].strip()
        self.history += _ids(self.tok, f"Assistant: {reply}\n")
        return reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["stories", "dialogue", "everyday", "chatmix"], default="stories")
    p.add_argument("--init", default="", help="warm-start from this checkpoint")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--K", type=int, default=16, help="navigator steps per encoder refresh")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_stories", type=int, default=200000, help="number of training examples")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--max_turn_words", type=int, default=0,
                   help="dialogue: drop conversations with any turn longer than this (0 = keep all)")
    p.add_argument("--daily", type=int, default=0,
                   help="dialogue: mix in DailyDialog repeated N times (0 = SODA only)")
    p.add_argument("--min_prefix", type=int, default=4)
    p.add_argument("--enc_layer", type=int, default=-1, help="encoder layer to read; try middle layers")
    p.add_argument("--aux_weight", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--pos_ce", action="store_true", help="print val loss per chunk position")
    p.add_argument("--ckpt", default="navigator.pt")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--chat", action="store_true", help="terminal chat (use ChatBot in notebooks)")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--n_tokens", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--refresh", type=int, default=0, help="re-encode every N tokens (<=K; 0 = K)")
    a = p.parse_args()

    if a.chat:
        bot = ChatBot(a.ckpt, a.temp, a.topk, a.refresh)
        print("chat ready: type a message, 'reset' to clear, 'quit' to exit")
        while True:
            msg = input("you: ").strip()
            if msg == "quit":
                break
            if msg == "reset":
                bot.reset(); continue
            print("bot:", bot.say(msg))
        return
    if not a.generate:
        return train(a)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu")
    for k in ("K", "d", "layers", "heads", "enc_layer", "max_len"):
        setattr(a, k, ck["args"][k])
    tok, enc = load_encoder(device)
    sub_ids = ck["sub_ids"].to(device)
    nav, _, _ = build_navigator(enc, sub_ids, a, device)
    nav.load_state_dict(ck["nav"])
    print(generate(enc, nav, tok, sub_ids, a.prompt, a, device, a.n_tokens))


if __name__ == "__main__":
    main()