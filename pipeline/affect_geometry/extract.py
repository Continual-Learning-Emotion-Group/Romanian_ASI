"""Extract all-layer target representations and retain lemma-level sufficient statistics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def token_indices(offsets, start: int, end: int):
    return [index for index, (left, right) in enumerate(offsets)
            if right > left and left < end and right > start]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-metadata", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", default=None,
                        help="optional PEFT LoRA adapter directory, merged into the base weights")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.manifest, encoding="utf-8")]
    lemmas = sorted({row["lemma"] for row in rows})
    lemma_index = {lemma: index for index, lemma in enumerate(lemmas)}
    lemma_emotion = {lemma: next((row["basic_emotion"] for row in rows
                                  if row["lemma"] == lemma), None) for lemma in lemmas}

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.eval().cuda()

    sums = counts = layers = hidden_size = None
    invalid = 0
    started = time.time()
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start:batch_start + args.batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch],
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=args.maximum_tokens,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            # logits_to_keep=1: we only need hidden states; skipping the full
            # vocab projection saves ~5 GB peak memory on 40 GB cards.
            output = model(**encoded, output_hidden_states=True, use_cache=False,
                           logits_to_keep=1)
        hidden = output.hidden_states
        if sums is None:
            layers = np.arange(len(hidden), dtype=np.int16)
            hidden_size = int(hidden[0].shape[-1])
            sums = np.zeros((len(layers), len(lemmas), hidden_size), dtype=np.float64)
            counts = np.zeros(len(lemmas), dtype=np.int32)
        for batch_index, row in enumerate(batch):
            indices = token_indices(offsets[batch_index], row["target_start"], row["target_end"])
            if not indices:
                invalid += 1
                continue
            lemma_id = lemma_index[row["lemma"]]
            index_tensor = torch.tensor(indices, device=hidden[0].device)
            for layer_index, state in enumerate(hidden):
                vector = state[batch_index].index_select(0, index_tensor).mean(0).float().cpu().numpy()
                sums[layer_index, lemma_id] += vector
            counts[lemma_id] += 1
        done = batch_start + len(batch)
        if done == len(rows) or done % (args.batch_size * 50) == 0:
            print(f"{done}/{len(rows)} ({done / (time.time() - started):.1f} rows/s)", flush=True)

    if np.any(counts == 0):
        missing = [lemma for lemma, count in zip(lemmas, counts) if count == 0]
        raise RuntimeError(f"No valid representations for lemmas: {missing}")
    centroids = (sums / counts[None, :, None]).astype(np.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        centroids=centroids,
        counts=counts,
        layers=layers,
        lemmas=np.asarray(lemmas),
        basic_emotions=np.asarray([lemma_emotion[lemma] or "" for lemma in lemmas]),
    )
    metadata = {
        "manifest": str(args.manifest),
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "maximum_tokens": args.maximum_tokens,
        "rows": len(rows),
        "valid_rows": int(counts.sum()),
        "invalid_rows": invalid,
        "lemmas": len(lemmas),
        "layers": layers.tolist(),
        "hidden_size": hidden_size,
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
    }
    Path(args.run_metadata).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

