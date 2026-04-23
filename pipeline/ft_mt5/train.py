"""Fine-tune mT5-large on Romanian ASI using the MASIVE recipe.

Reads pipeline/data/splits/{train,val}.jsonl, applies single-span masking
(<extra_id_0>) and sentence-level truncation, then runs HF Seq2SeqTrainer.

Usage (single A6000 on seahorse):

    export HF_HOME=/local/nlp/$USER/ro_asi_ft/hf_cache
    export CUDA_VISIBLE_DEVICES=0
    python -m pipeline.ft_mt5.train

Smoke test (laptop-friendly sizes):

    python -m pipeline.ft_mt5.train \\
        --model google/mt5-small --max-train-samples 200 --max-val-samples 40 \\
        --num-train-epochs 1 --per-device-train-batch-size 2 --bf16 false \\
        --output-dir /tmp/mt5-smoke
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from pipeline.ft_mt5.build_training_data import build_example
from pipeline.ft_mt5.config import TrainConfig


def _str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "y"}


def parse_cli(defaults: TrainConfig) -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model_name_or_path", default=defaults.model_name_or_path)
    p.add_argument("--train-file", type=Path, default=defaults.train_file)
    p.add_argument("--val-file",   type=Path, default=defaults.val_file)
    p.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    p.add_argument("--num-train-epochs", type=int, default=defaults.num_train_epochs)
    p.add_argument("--per-device-train-batch-size", type=int,
                   default=defaults.per_device_train_batch_size)
    p.add_argument("--per-device-eval-batch-size", type=int,
                   default=defaults.per_device_eval_batch_size)
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--max-input-tokens",  type=int, default=defaults.max_input_tokens)
    p.add_argument("--max-target-tokens", type=int, default=defaults.max_target_tokens)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples",   type=int, default=None)
    p.add_argument("--bf16", type=_str2bool, default=defaults.bf16)
    p.add_argument("--seed", type=int, default=defaults.seed)
    args = p.parse_args()
    cfg = TrainConfig(**{**asdict(defaults), **vars(args)})
    return cfg


def load_jsonl(path: Path, max_rows: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def main():
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )
    from transformers.optimization import Adafactor, get_linear_schedule_with_warmup
    from datasets import Dataset

    cfg = parse_cli(TrainConfig())
    set_seed(cfg.seed)

    if cfg.tf32:
        torch.set_float32_matmul_precision("high")

    print(f"Model:  {cfg.model_name_or_path}")
    print(f"Train:  {cfg.train_file}")
    print(f"Val:    {cfg.val_file}")
    print(f"Output: {cfg.output_dir}")

    # ------------------------------------------------------------------
    # Tokenizer & model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)

    # ------------------------------------------------------------------
    # Build datasets: read split rows, run build_example (truncation +
    # <extra_id_0> substitution), then tokenize input/target.
    # ------------------------------------------------------------------
    train_rows = load_jsonl(cfg.train_file, cfg.max_train_samples)
    val_rows   = load_jsonl(cfg.val_file,   cfg.max_val_samples)
    print(f"Loaded {len(train_rows)} train / {len(val_rows)} val rows")

    def preprocess(batch: dict) -> dict:
        inputs, targets = [], []
        for i in range(len(batch["id"])):
            row = {k: batch[k][i] for k in batch}
            ex = build_example(row, tokenizer, cfg.max_input_tokens)
            inputs.append(ex["input"])
            targets.append(ex["target"])
        model_inputs = tokenizer(
            inputs, max_length=cfg.max_input_tokens,
            truncation=True, padding=False,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=cfg.max_target_tokens,
            truncation=True, padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

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

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest",
        label_pad_token_id=-100,
    )

    # ------------------------------------------------------------------
    # Optimizer + scheduler (Adafactor + linear) — MASIVE Appendix D
    # ------------------------------------------------------------------
    # HF recommends these kwargs when using a fixed external LR with Adafactor.
    optimizer = Adafactor(
        model.parameters(),
        lr=cfg.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=cfg.weight_decay,
    )
    # Num training steps: computed by trainer; we wrap scheduler in a lambda
    # so Trainer can build it once it knows total steps.
    steps_per_epoch_est = max(
        1,
        len(train_ds) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps),
    )
    total_steps = steps_per_epoch_est * cfg.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        tf32=cfg.tf32 if torch.cuda.is_available() else None,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        predict_with_generate=False,  # eval loss only; we use eval_generative.py for acc
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
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
