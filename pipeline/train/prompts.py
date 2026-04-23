"""Chat-template prompt building and loss-masking collation for SFT.

For each training example we:
  1. Render the full conversation (system + user + assistant) via
     `tokenizer.apply_chat_template(..., enable_thinking=False)`.
  2. Render the prompt-only prefix (system + user + `<|im_start|>assistant\\n`)
     the same way with `add_generation_prompt=True`.
  3. Tokenize both. `labels` = `input_ids.clone()`; set labels[:prompt_len] to
     -100 so loss is only computed on the assistant turn + `<|im_end|>`.

The assistant turn is a single word (the emotion label), so gradients are
concentrated where they matter. Disabling `enable_thinking` keeps `<think>`
tokens out of the target entirely.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

SYSTEM_PROMPT = (
    "You are an affective-state identifier. Given a sentence with one or more "
    "[MASK] positions, output the distinct affective expressions that fill "
    "them, in order of first appearance, separated by a single space. "
    "Expressions may be single words or short idiomatic phrases. "
    "No explanation, no punctuation, no repeats."
)

MAX_INPUT_CHARS = 2200  # ≈700 Qwen tokens; chat template adds ~50 on top


def _truncate_around_mask(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Keep a window centered on [MASK] so we never lose the masked token.

    With Qwen3.5's 248k vocab the softmax memory scales linearly with sequence
    length, so we bound the user content to a predictable length. The long-tail
    of FULG passages reaches ~10k tokens; without this, batches with one long
    sample OOM during cross-entropy.
    """
    if len(text) <= max_chars:
        return text
    pos = text.find("[MASK]")
    if pos == -1:  # no mask — should not happen in this dataset
        return text[:max_chars]
    half = max_chars // 2
    start = max(0, pos - half)
    end = min(len(text), pos + half)
    out = text[start:end]
    if start > 0:
        out = "..." + out
    if end < len(text):
        out = out + "..."
    return out


def build_messages(input_text: str, label: str | None = None) -> list[dict]:
    """System + user (+ optional assistant) messages."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _truncate_around_mask(input_text)},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": label})
    return messages


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    """Apply chat template with `enable_thinking=False` if the template supports it."""
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_example(
    tokenizer,
    input_text: str,
    label: str,
    max_length: int = 1024,
) -> dict[str, list[int]]:
    """Tokenize one example and return input_ids / attention_mask / labels.

    Tokens corresponding to the prompt prefix are masked (-100) so loss is
    only taken on the assistant turn (`label` + `<|im_end|>`).
    """
    full_text = _apply_chat_template(
        tokenizer, build_messages(input_text, label), add_generation_prompt=False
    )
    prompt_text = _apply_chat_template(
        tokenizer, build_messages(input_text, label=None), add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True,
                         max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True,
                           max_length=max_length)["input_ids"]

    # The prompt text must be a prefix of the full text at the token level.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokenization is not a prefix of the full conversation. "
            "Chat template may be inconsistent between the two renderings."
        )

    labels = list(full_ids)
    for i in range(len(prompt_ids)):
        labels[i] = -100

    attention_mask = [1] * len(full_ids)
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class LossMaskingCollator:
    """Right-pad `input_ids`, `attention_mask`, `labels` to the longest in batch.

    `labels` are padded with -100 so padding tokens don't contribute to loss.
    """
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _sanity(model_name: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cases = [
        ("RO single-mask single-word",
         "Eram [MASK] de vedenia ei.",
         "uluita"),
        ("EN multi-mask distinct labels (matched)",
         "He didn't make me feel [MASK] nor [MASK].",
         "loved disgusted"),
        ("EN multi-mask repeated label (unique-set)",
         "I feel [MASK]. Or rather, I am [MASK].",
         "unfit"),
        ("FA single-mask phrase label",
         "این روزها آنقدر [MASK] که دیگر نمی‌دانم.",
         "بغضم سنگین شده"),
    ]

    think_ids = {
        tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")
        if tok.convert_tokens_to_ids(t) != tok.unk_token_id
    }

    for name, example_input, example_label in cases:
        print(f"\n=== {name} ===")
        print(f"input: {example_input!r}")
        print(f"target: {example_label!r}")

        encoded = encode_example(tok, example_input, example_label)
        loss_ids = [i for i, lab in zip(encoded["input_ids"], encoded["labels"]) if lab != -100]
        loss_text = tok.decode(loss_ids)

        n_loss = sum(1 for lab in encoded["labels"] if lab != -100)
        print(f"tokens with loss: {n_loss}/{len(encoded['labels'])}")
        print(f"loss span decoded: {loss_text!r}")

        for i, (input_id, lab) in enumerate(zip(encoded["input_ids"], encoded["labels"])):
            if lab != -100 and input_id in think_ids:
                raise AssertionError(f"{name}: loss taken on thinking token at position {i}")

        # The label text must appear (with whitespace tolerance) inside the
        # loss span. We compare on collapsed whitespace since tokenizers may
        # emit the label as "loved disgusted" or " loved disgusted".
        lab_norm = " ".join(example_label.split())
        span_norm = " ".join(loss_text.split())
        assert lab_norm in span_norm, \
            f"{name}: expected {lab_norm!r} inside {span_norm!r}"
        decoded_tokens = [tok.decode([i]) for i in loss_ids]
        assert "<|im_end|>" in decoded_tokens, \
            f"{name}: expected <|im_end|> in loss span, got {decoded_tokens!r}"
        print(f"✓ loss spans only the assistant turn")

    print("\nAll sanity cases passed ✓")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="Print a tokenized example")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    if args.sanity:
        _sanity(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
