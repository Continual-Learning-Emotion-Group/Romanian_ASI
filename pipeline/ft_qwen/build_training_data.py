"""Convert split JSONL rows into Qwen chat-format training tensors.

Sibling to `pipeline/ft_mt5/build_training_data.py`. The mT5 version writes
out a JSONL of (input, target) text pairs; here we go straight from the raw
row to (input_ids, attention_mask, labels) because the chat template + loss
masking are coupled and there's no useful intermediate text representation.

Reuses `pipeline/ft_mt5/truncate.py` (`truncate_to_max_tokens`) so the
sentence-level truncation policy stays identical across both runs.
"""

from __future__ import annotations

from pipeline.ft_mt5.truncate import truncate_to_max_tokens
from pipeline.ft_qwen.prompts import tokenize_with_loss_mask

# Qwen tokenizes "[MASK]" as a literal multi-token sequence — there's no
# special sentinel like mT5's <extra_id_0>. We pass it as the marker to
# preserve so truncation never drops the mask sentence.
QWEN_MASK = "[MASK]"


def build_example(
    row: dict,
    tokenizer,
    max_input_tokens: int = 1024,
    max_target_tokens: int = 16,
) -> dict:
    """Return a tokenized training example ready for the data collator.

    Output fields:
        input_ids, attention_mask, labels  (lists, no padding)
        seed_word, seed_word_normalized, source, pattern_category, emotion_category
            — preserved as metadata for downstream analysis but stripped before
              the tensors hit the model (Trainer drops unknown columns).
    """
    masked_text = truncate_to_max_tokens(
        row["masked_text"], QWEN_MASK, tokenizer, max_tokens=max_input_tokens,
    )
    if QWEN_MASK not in masked_text:
        raise ValueError(f"Mask token lost during truncation for id={row['id']}")

    encoded = tokenize_with_loss_mask(
        masked_text, row["seed_word"], tokenizer,
        max_input_tokens=max_input_tokens,
        max_target_tokens=max_target_tokens,
    )

    return {
        **encoded,
        "id": row["id"],
        "seed_word": row["seed_word"],
        "seed_word_normalized": row["seed_word_normalized"],
        "source": row.get("source", ""),
        "pattern_category": row.get("pattern_category", ""),
        "emotion_category": row.get("emotion_category", []),
    }
