"""Chat-template builder + loss-masking tokenization for Qwen3 SFT.

The mt5 run feeds the model `<extra_id_0>` and trains on
`<extra_id_0> word <extra_id_1>`. Qwen3 is a decoder-only chat model, so we
adapt the same data to its chat template and mask the prompt tokens out of the
loss so only the assistant's single-word answer is supervised.

Sequence structure produced by `apply_chat_template(..., enable_thinking=False)`:

    <|im_start|>system
    {SYSTEM}<|im_end|>
    <|im_start|>user
    {masked_text}<|im_end|>
    <|im_start|>assistant
    {seed_word}<|im_end|>

The label tensor copies `input_ids` for the assistant span ({seed_word}<|im_end|>)
and is set to -100 everywhere else, so cross-entropy only flows through the
target word and the EOS marker.

CLI smoke test:

    python -m pipeline.ft_qwen.prompts --sanity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT_RO = (
    "Ești un model care identifică stări afective. "
    "Pentru o propoziție cu `[MASK]`, răspunde cu un singur cuvânt — "
    "cuvântul care completează `[MASK]`. Fără explicații, fără punctuație."
)


def build_messages(masked_text: str, seed_word: str | None = None) -> list[dict]:
    """Build the Qwen chat-template messages list.

    If `seed_word` is None, return only system + user (for inference).
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_RO},
        {"role": "user",   "content": masked_text},
    ]
    if seed_word is not None:
        msgs.append({"role": "assistant", "content": seed_word})
    return msgs


def tokenize_with_loss_mask(
    masked_text: str,
    seed_word: str,
    tokenizer,
    max_input_tokens: int = 1024,
    max_target_tokens: int = 16,
) -> dict:
    """Tokenize a (masked_text, seed_word) pair into Qwen chat format with
    prompt tokens masked out of the loss.

    Returns {"input_ids": [...], "attention_mask": [...], "labels": [...]}.
    """
    # Prompt-only prefix: system + user, ending with the assistant header
    # (`<|im_start|>assistant\n`) so the next tokens are the target word.
    prefix_msgs = build_messages(masked_text, seed_word=None)
    prefix_text = tokenizer.apply_chat_template(
        prefix_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Full sequence: same prompt + assistant turn with the target word.
    full_msgs = build_messages(masked_text, seed_word=seed_word)
    full_text = tokenizer.apply_chat_template(
        full_msgs,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Sanity: the chat template should produce the prefix as a strict prefix
    # of the full sequence. If not, the assistant turn was rendered differently
    # than the generation-prompt header — bail loudly so we notice in tests.
    if not full_text.startswith(prefix_text):
        raise RuntimeError(
            "Chat template produced an assistant turn that is not a strict "
            "extension of the generation-prompt prefix. This breaks loss masking."
        )

    # Tokenize without padding (collator handles padding to longest in batch).
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens + max_target_tokens,
    )["input_ids"]
    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
    )["input_ids"]

    # Defense in depth: prefix must be a strict prefix of full at the token level.
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError(
            "Tokenized prefix is not a prefix of tokenized full sequence."
        )

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _sanity_check(model_name: str = "Qwen/Qwen3.5-4B") -> None:
    """Print one fully-tokenized example and assert the masking is correct.

    Reads the first row of pipeline/data/splits/train.jsonl in the current
    repo. Does not download model weights — only the tokenizer.
    """
    from transformers import AutoTokenizer

    splits = Path(__file__).resolve().parent.parent / "data" / "splits"
    train_path = splits / "train.jsonl"
    with open(train_path) as f:
        row = json.loads(f.readline())

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    out = tokenize_with_loss_mask(
        row["masked_text"], row["seed_word"], tokenizer,
    )
    input_ids = out["input_ids"]
    labels = out["labels"]

    n_total = len(input_ids)
    n_supervised = sum(1 for x in labels if x != -100)

    print(f"\n--- Row id: {row['id']}")
    print(f"  seed_word: {row['seed_word']!r}")
    print(f"  total tokens: {n_total}")
    print(f"  supervised tokens (labels != -100): {n_supervised}")
    print(f"  prompt tokens (labels == -100): {n_total - n_supervised}")

    supervised_text = tokenizer.decode(
        [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    )
    print(f"  supervised decoded: {supervised_text!r}")

    last20 = tokenizer.decode(input_ids[-20:])
    print(f"  last 20 tokens decoded: {last20!r}")

    assert "[MASK]" in tokenizer.decode(input_ids), \
        "Expected [MASK] preserved verbatim in user turn"
    assert "<think>" not in tokenizer.decode(input_ids), \
        "Did not expect <think> tokens (enable_thinking=False)"
    assert n_supervised >= 1, "Expected at least one supervised token (the target word)"
    assert n_supervised < n_total, "Expected at least some prompt tokens to be masked out"

    print("\nOK: prompt is masked out, target word is supervised, no <think> block.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true",
                        help="Print one tokenized example with mask diagnostics")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    if args.sanity:
        _sanity_check(args.model)
    else:
        parser.print_help()
