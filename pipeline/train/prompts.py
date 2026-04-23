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
import argparse
from dataclasses import dataclass
from typing import Any

import torch

SYSTEM_PROMPT = (
    "You are an affective-state identifier. Given a sentence with [MASK], "
    "respond with only the single word that fills the mask — no explanation, "
    "no punctuation."
)


def build_messages(input_text: str, label: str | None = None) -> list[dict]:
    """System + user (+ optional assistant) messages."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
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

    example_input = "Eram [MASK] de vedenia ei."
    example_label = "uluita"
    encoded = encode_example(tok, example_input, example_label)

    print("=== Full rendered prompt ===")
    print(_apply_chat_template(tok, build_messages(example_input, example_label),
                               add_generation_prompt=False))
    print()

    decoded_tokens = [tok.decode([t]) for t in encoded["input_ids"]]
    print("=== Per-token label mask (IDs with loss on them) ===")
    for i, (tok_str, lab) in enumerate(zip(decoded_tokens, encoded["labels"])):
        mark = "LOSS" if lab != -100 else "    "
        print(f"  {i:4d} {mark} {tok_str!r}")

    n_loss = sum(1 for lab in encoded["labels"] if lab != -100)
    n_total = len(encoded["labels"])
    print(f"\nTokens with loss: {n_loss}/{n_total}")

    # Qwen3's non-thinking mode still emits an empty `<think>\n\n</think>\n\n`
    # stub in the assistant prefix; that is by design. What we must verify is
    # that (a) no loss is taken on any thinking-related token and (b) the loss
    # spans only the label + `<|im_end|>` (+ trailing newline).
    think_ids = {
        tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")
        if tok.convert_tokens_to_ids(t) != tok.unk_token_id
    }
    for i, (input_id, lab) in enumerate(zip(encoded["input_ids"], encoded["labels"])):
        if lab != -100 and input_id in think_ids:
            raise AssertionError(f"loss taken on thinking token at position {i}")

    loss_ids = [i for i, lab in zip(encoded["input_ids"], encoded["labels"]) if lab != -100]
    loss_text = tok.decode(loss_ids)
    print(f"Loss span decoded: {loss_text!r}")
    assert example_label in loss_text, \
        f"expected {example_label!r} inside loss span, got {loss_text!r}"
    assert "<|im_end|>" in [tok.decode([i]) for i in loss_ids], \
        f"expected <|im_end|> token in loss span, got ids {loss_ids}"
    print("Loss spans only the assistant turn ✓")


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
