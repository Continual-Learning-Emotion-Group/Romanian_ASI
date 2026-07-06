"""Extract contextual hidden states at the emotion-word position (runs on GPU box).

Standalone (no repo imports) so it can be scp'd alone. Reads the windowed input
JSONL produced by select.py, feeds each `text` (unmasked) through the model,
locates the emotion-word subword tokens via char offsets, and mean-pools their
hidden states across a set of evenly-spaced layers.

Output: <out>.npz  with arrays  layer_<idx> : float16 [N, H]  + `valid` mask,
        <out>.meta.jsonl  one metadata row per input row (order preserved).

Usage:
    python extract.py --input tier1_ro_input.jsonl --out tier1_ro_hidden \
        --model Qwen/Qwen3.5-4B --batch-size 16 --num-save-layers 13
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_layers(n_states: int, k: int) -> list[int]:
    """k evenly-spaced layer indices over [0, n_states-1] inclusive of ends."""
    if k >= n_states:
        return list(range(n_states))
    return sorted({round(i * (n_states - 1) / (k - 1)) for i in range(k)})


def word_token_indices(offsets, w_start: int, w_end: int) -> list[int]:
    idx = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:            # special/pad token -> (0,0)
            continue
        if a < w_end and b > w_start:   # overlaps the word span
            idx.append(i)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-save-layers", type=int, default=13)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    print(f"loaded {len(rows)} rows", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype,
        attn_implementation="sdpa",
    ).eval().cuda()
    print("model loaded", flush=True)

    save_layers = None
    per_layer: dict[int, list] = {}
    valid: list[bool] = []
    meta: list[dict] = []
    t0 = time.time()

    for bstart in range(0, len(rows), args.batch_size):
        batch = rows[bstart:bstart + args.batch_size]
        enc = tok([r["text"] for r in batch], return_offsets_mapping=True,
                  padding=True, truncation=True, max_length=args.max_tokens,
                  return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                       # tuple (L+1) of [B,T,H]

        if save_layers is None:
            save_layers = pick_layers(len(hs), args.num_save_layers)
            per_layer = {li: [] for li in save_layers}
            print(f"{len(hs)} hidden states; saving layers {save_layers}", flush=True)

        for j, r in enumerate(batch):
            tokidx = word_token_indices(offsets[j], r["word_start"], r["word_end"])
            ok = len(tokidx) > 0
            valid.append(ok)
            meta.append({"id": r["id"], "plutchik": r["plutchik"], "valence": r.get("valence", 0),
                         "surface": r["surface"], "source": r.get("source"),
                         "lemma": r.get("lemma"),
                         "seed_word_normalized": r.get("seed_word_normalized"),
                         "n_word_tokens": len(tokidx)})
            ti = torch.tensor(tokidx if ok else [0], device=hs[0].device)
            for li in save_layers:
                vec = hs[li][j].index_select(0, ti).mean(0).float().cpu().numpy()
                per_layer[li].append(vec.astype(np.float16))

        done = bstart + len(batch)
        if done % (args.batch_size * 20) == 0 or done == len(rows):
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(rows)}  ({rate:.1f} rows/s)", flush=True)

    arrays = {f"layer_{li}": np.stack(per_layer[li]) for li in save_layers}
    arrays["valid"] = np.array(valid)
    arrays["save_layers"] = np.array(save_layers)
    np.savez_compressed(args.out + ".npz", **arrays)
    with open(args.out + ".meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    n_ok = int(np.sum(valid))
    print(f"done: {n_ok}/{len(rows)} valid | layers {save_layers} | "
          f"dim {arrays[f'layer_{save_layers[0]}'].shape} | wrote {args.out}.npz", flush=True)


if __name__ == "__main__":
    main()
