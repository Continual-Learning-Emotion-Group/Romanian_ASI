"""Fine-tune Qwen3.5-4B on Romanian ASI (sibling of pipeline/ft_mt5/train.py).

Reads pipeline/data/splits/{train,val}.jsonl, builds Qwen chat-template
sequences with prompt-tokens masked out of the loss, then runs HF Trainer
(causal LM head, AdamW, BF16 + Flash Attention 2 + gradient checkpointing).

Usage (single A100 on tigerfish):

    export HF_HOME=/local/nlp/$USER/ro_asi_ft/hf_cache
    export CUDA_VISIBLE_DEVICES=1
    python -m pipeline.ft_qwen.train

Smoke test (laptop-friendly: small model, no FA2, no bf16):

    python -m pipeline.ft_qwen.train \\
        --model Qwen/Qwen3.5-0.5B \\
        --max-train-samples 50 --max-val-samples 10 \\
        --num-train-epochs 1 --per-device-train-batch-size 1 \\
        --gradient-accumulation-steps 1 \\
        --bf16 false --attn-implementation eager \\
        --output-dir /tmp/qwen-smoke
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pipeline.ft_qwen.build_training_data import build_example
from pipeline.ft_qwen.config import TrainConfig


def _str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "y"}


def parse_cli(defaults: TrainConfig) -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model_name_or_path", default=defaults.model_name_or_path)
    p.add_argument("--attn-implementation", default=defaults.attn_implementation,
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--train-file", type=Path, default=defaults.train_file)
    p.add_argument("--val-file",   type=Path, default=defaults.val_file)
    p.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    p.add_argument("--num-train-epochs", type=int, default=defaults.num_train_epochs)
    p.add_argument("--per-device-train-batch-size", type=int,
                   default=defaults.per_device_train_batch_size)
    p.add_argument("--per-device-eval-batch-size", type=int,
                   default=defaults.per_device_eval_batch_size)
    p.add_argument("--gradient-accumulation-steps", type=int,
                   default=defaults.gradient_accumulation_steps)
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--lr-scheduler-type", default=defaults.lr_scheduler_type)
    p.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    p.add_argument("--max-input-tokens",  type=int, default=defaults.max_input_tokens)
    p.add_argument("--max-target-tokens", type=int, default=defaults.max_target_tokens)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples",   type=int, default=None)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Optional cap on training steps (overrides epochs)")
    p.add_argument("--bf16", type=_str2bool, default=defaults.bf16)
    p.add_argument("--gradient-checkpointing", type=_str2bool,
                   default=defaults.gradient_checkpointing)
    p.add_argument("--report-to", default=defaults.report_to)
    p.add_argument("--run-name", default=defaults.run_name)
    p.add_argument("--seed", type=int, default=defaults.seed)
    args = p.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k in asdict(defaults)}
    cfg = TrainConfig(**{**asdict(defaults), **overrides})
    cfg.max_steps = args.max_steps  # CLI-only field
    return cfg


def load_jsonl(path: Path, max_rows: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


class CausalCollator:
    """Pad input_ids / attention_mask / labels to the longest sequence in the batch.

    DataCollatorForLanguageModeling(mlm=False) would overwrite our labels by
    copying input_ids — we already built labels with the prompt masked, so we
    pad them ourselves. Right-pad (Qwen training convention).
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn_masks, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attn_masks.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from datasets import Dataset

    cfg = parse_cli(TrainConfig())
    set_seed(cfg.seed)

    if cfg.tf32:
        torch.set_float32_matmul_precision("high")

    print(f"Model:  {cfg.model_name_or_path}")
    print(f"Train:  {cfg.train_file}")
    print(f"Val:    {cfg.val_file}")
    print(f"Output: {cfg.output_dir}")
    print(f"Attn:   {cfg.attn_implementation}")

    # ------------------------------------------------------------------
    # Tokenizer & model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token_id is None:
        # Qwen3 normally ships with a pad token; guard for forks that don't.
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"attn_implementation": cfg.attn_implementation}
    if cfg.bf16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False  # incompatible with grad checkpointing

    # ------------------------------------------------------------------
    # Build datasets: read split rows, run build_example (truncation +
    # chat template + loss masking).
    # ------------------------------------------------------------------
    train_rows = load_jsonl(cfg.train_file, cfg.max_train_samples)
    val_rows   = load_jsonl(cfg.val_file,   cfg.max_val_samples)
    print(f"Loaded {len(train_rows)} train / {len(val_rows)} val rows")

    def preprocess(batch: dict) -> dict:
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(batch["id"])):
            row = {k: batch[k][i] for k in batch}
            ex = build_example(
                row, tokenizer,
                max_input_tokens=cfg.max_input_tokens,
                max_target_tokens=cfg.max_target_tokens,
            )
            out["input_ids"].append(ex["input_ids"])
            out["attention_mask"].append(ex["attention_mask"])
            out["labels"].append(ex["labels"])
        return out

    train_ds = Dataset.from_list(train_rows)
    val_ds   = Dataset.from_list(val_rows)
    train_ds = train_ds.map(
        preprocess, batched=True, batch_size=64,
        num_proc=cfg.preprocess_num_workers,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    val_ds = val_ds.map(
        preprocess, batched=True, batch_size=64,
        num_proc=cfg.preprocess_num_workers,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)

    # ------------------------------------------------------------------
    # TrainingArguments — AdamW (Trainer default), cosine LR, FA2 via
    # the model load above, BF16, gradient checkpointing.
    # ------------------------------------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=getattr(cfg, "max_steps", -1),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        tf32=cfg.tf32 if torch.cuda.is_available() else None,
        gradient_checkpointing=False,  # already enabled on the model above
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_result = trainer.train()
    trainer.save_model(str(cfg.output_dir / "best"))
    tokenizer.save_pretrained(str(cfg.output_dir / "best"))
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)
    print(f"\nFinal eval_loss: {eval_metrics.get('eval_loss'):.4f}")
    print(f"Best model saved to {cfg.output_dir / 'best'}")


if __name__ == "__main__":
    main()
