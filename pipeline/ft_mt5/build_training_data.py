"""Convert split JSONL rows into mT5 span-corruption (input, target) pairs.

Input row (from pipeline/data/splits/{train,val,test}.jsonl):
    {
      "id": "...",
      "text": "...",
      "masked_text": "...Sunt [MASK]. ...",
      "seed_word": "mulțumit",
      ...
    }

Output (JSONL):
    {"id": ..., "input": "...Sunt <extra_id_0>. ...", "target": "<extra_id_0> mulțumit <extra_id_1>"}

The `input` is truncated sentence-wise to fit 512 mT5 tokens, preserving the
sentence containing the span. Target format matches mT5's pretraining span-
corruption: `<extra_id_0> <word> <extra_id_1>` marks the end of the first span.

Usage:
    python -m pipeline.ft_mt5.build_training_data \\
        --input  pipeline/data/splits/train.jsonl \\
        --output pipeline/data/splits/train.mt5.jsonl
    python -m pipeline.ft_mt5.build_training_data --max 100 --out /tmp/tiny.jsonl
"""

import argparse
import json
from pathlib import Path

from pipeline.ft_mt5.truncate import truncate_to_max_tokens

MT5_MASK = "<extra_id_0>"
MT5_MASK_END = "<extra_id_1>"


def build_example(row: dict, tokenizer, max_input_tokens: int = 512) -> dict:
    masked_text = row["masked_text"].replace("[MASK]", MT5_MASK)
    truncated = truncate_to_max_tokens(
        masked_text, MT5_MASK, tokenizer, max_tokens=max_input_tokens,
    )
    if MT5_MASK not in truncated:
        raise ValueError(f"Mask token lost during truncation for id={row['id']}")
    target = f"{MT5_MASK} {row['seed_word']} {MT5_MASK_END}"
    return {
        "id": row["id"],
        "input": truncated,
        "target": target,
        "seed_word": row["seed_word"],
        "seed_word_normalized": row["seed_word_normalized"],
        "source": row.get("source", ""),
        "pattern_category": row.get("pattern_category", ""),
        "emotion_category": row.get("emotion_category", []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, default="google/mt5-large")
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max", type=int, default=None,
                        help="Optional cap on number of rows (for smoke tests)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_truncated = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.max and n_in >= args.max:
                break
            n_in += 1
            row = json.loads(line)
            try:
                ex = build_example(row, tokenizer, args.max_input_tokens)
            except ValueError as e:
                print(f"  skip: {e}")
                continue
            if ex["input"] != row["masked_text"].replace("[MASK]", MT5_MASK):
                n_truncated += 1
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_out += 1
            if n_in % 5000 == 0:
                print(f"  processed {n_in} rows ({n_truncated} truncated)")

    print(f"Wrote {n_out}/{n_in} examples to {args.output} "
          f"({n_truncated} truncated to {args.max_input_tokens} tokens)")


if __name__ == "__main__":
    main()
