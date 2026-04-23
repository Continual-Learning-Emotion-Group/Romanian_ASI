"""Day-by-day sequential multilingual SFT for Qwen3.5-4B.

Runs N "days" of training, one per language, in the order given by
`--language-order` (default: en,es,fa,hi,ro). Each day:
  1. Loads the model (base on day 1, previous day's `final/` afterwards).
  2. Trains for `num_train_epochs` epochs on `train_{lang}` / `val_{lang}`
     from the DatasetDict produced by
     `prepare_data.py --per-language-splits`.
  3. Saves the end-of-day model to `runs_root/day{N}_{lang}/final/`.
  4. Evaluates on ALL 5 language test sets via `pipeline.train.eval_sft`
     (subprocess so vLLM can grab the full GPU after we release training memory).

`load_best_model_at_end` MUST be false in the YAML — we want the end-of-day
weights as the next day's init, and disabling this also sidesteps the
Qwen3.5 multimodal `model.language_model.*` checkpoint-key bug seen in
pipeline/ft_qwen/RUN_LOG.md.

Usage:
    python -m pipeline.train.train_day_by_day \
        --config pipeline/train/configs/qwen3_5_4b_day_by_day.yaml \
        --language-order en,es,fa,hi,ro

Smoke test:
    python -m pipeline.train.train_day_by_day \
        --config pipeline/train/configs/qwen3_5_4b_day_by_day.yaml \
        --max-steps-per-day 2 --skip-eval
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from pipeline.train.prompts import LossMaskingCollator, encode_example


CONFIG_ONLY_KEYS = {
    "model_name_or_path",
    "dataset_dir",
    "max_seq_length",
    "final_dir",
    "attn_implementation",
    "runs_root",
}

VALID_LANGUAGES = {"ro", "en", "es", "fa", "hi"}


def load_yaml_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _split_config_only_overrides(extra_cli: list[str]) -> tuple[list[str], dict]:
    """Pull `--<config_only_key> <value>` overrides out of extra_cli so they
    don't reach HfArgumentParser (which only knows TrainingArguments fields)."""
    overrides: dict = {}
    remaining: list[str] = []
    i = 0
    while i < len(extra_cli):
        tok = extra_cli[i]
        matched = False
        for key in CONFIG_ONLY_KEYS:
            if tok == f"--{key}" and i + 1 < len(extra_cli):
                overrides[key] = extra_cli[i + 1]
                i += 2
                matched = True
                break
        if not matched:
            remaining.append(tok)
            i += 1
    return remaining, overrides


def build_training_args(config: dict, extra_cli: list[str]) -> tuple[TrainingArguments, dict]:
    """Same plumbing as pipeline/train/train.py: split config-only keys from
    HF TrainingArguments keys, then let HfArgumentParser parse the rest.

    CLI overrides for config-only keys (e.g. `--runs_root /tmp/...`) are
    pulled out of extra_cli before the HF parser sees them.
    """
    extra_cli, cli_config_overrides = _split_config_only_overrides(extra_cli)
    config_only = {k: config[k] for k in CONFIG_ONLY_KEYS if k in config}
    config_only.update(cli_config_overrides)
    hf_args = {k: v for k, v in config.items() if k not in CONFIG_ONLY_KEYS}

    import json as _json
    parser = HfArgumentParser(TrainingArguments)
    args_list: list[str] = []
    for k, v in hf_args.items():
        if isinstance(v, bool):
            args_list += [f"--{k}", str(v)]
        elif isinstance(v, dict):
            args_list += [f"--{k}", _json.dumps(v)]
        else:
            args_list += [f"--{k}", str(v)]
    args_list += extra_cli

    (training_args,) = parser.parse_args_into_dataclasses(args_list)
    return training_args, config_only


def tokenize_split(ds, tokenizer, max_length: int):
    def _map(example):
        return encode_example(tokenizer, example["input"], example["label"],
                              max_length=max_length)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in keep_cols])


def parse_language_order(raw: str) -> list[str]:
    order = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in order if x not in VALID_LANGUAGES]
    if bad:
        raise SystemExit(f"Unknown language(s) in --language-order: {bad}. "
                         f"Valid: {sorted(VALID_LANGUAGES)}")
    if len(set(order)) != len(order):
        raise SystemExit(f"--language-order has duplicates: {order}")
    return order


def run_one_day(
    day_idx: int,
    lang: str,
    prev_ckpt: str | None,
    cfg: dict,
    base_training_args: TrainingArguments,
    raw,
    max_steps_per_day: int,
) -> Path:
    """Train one day and return the path to the saved end-of-day model dir."""
    day_tag = f"day{day_idx}_{lang}"
    day_dir = Path(cfg["runs_root"]) / day_tag
    day_dir.mkdir(parents=True, exist_ok=True)
    final_dir = day_dir / "final"

    init_from = prev_ckpt or cfg["model_name_or_path"]
    print(f"\n{'=' * 60}\n[{day_tag}] init from: {init_from}\n{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"], trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        init_from,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if base_training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    train_key = f"train_{lang}"
    val_key = f"val_{lang}"
    if train_key not in raw or val_key not in raw:
        raise SystemExit(
            f"DatasetDict missing {train_key}/{val_key}. "
            f"Did you run prepare_data.py with --per-language-splits?"
        )
    train_ds = tokenize_split(raw[train_key], tokenizer, cfg.get("max_seq_length", 1024))
    val_ds = tokenize_split(raw[val_key], tokenizer, cfg.get("max_seq_length", 1024))

    targs = replace(
        base_training_args,
        output_dir=str(day_dir / "trainer"),
        run_name=f"{base_training_args.run_name}-{day_tag}",
        max_steps=max_steps_per_day if max_steps_per_day > 0 else base_training_args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=LossMaskingCollator(pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[{day_tag}] saved end-of-day model → {final_dir}")

    # Free disk: drop the Trainer's mid-training checkpoint dir now that we
    # have the clean `final/` snapshot. Each Qwen3.5-4B checkpoint is ~14 GB,
    # so leaving Trainer's dir around doubles per-day disk pressure.
    trainer_dir = day_dir / "trainer"
    if trainer_dir.exists():
        shutil.rmtree(trainer_dir, ignore_errors=True)
        print(f"[{day_tag}] removed mid-training trainer dir")

    # Release training-side GPU memory before vLLM eval.
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return final_dir


def run_eval(checkpoint: Path, dataset_dir: str, day_tag: str) -> bool:
    """Per-day eval on all 5 language test sets via pipeline.train.eval_sft.

    Returns True on success. Non-fatal on failure: we log the error and let
    the outer loop continue to the next day's training. The checkpoint is
    preserved (cleanup is gated on eval success) so we can re-run eval later
    via `python -m pipeline.train.eval_sft --checkpoint <day>/final ...`.
    """
    cmd = [
        sys.executable, "-m", "pipeline.train.eval_sft",
        "--checkpoint", str(checkpoint),
        "--dataset-dir", dataset_dir,
        "--tag", day_tag,
        "--backend", "vllm",
        "--no-similarity",
    ]
    print(f"[{day_tag}] launching eval: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{day_tag}] WARNING: eval failed (exit {e.returncode}). "
              f"Training continues; checkpoint preserved for later re-eval.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language-order", default="en,es,fa,hi,ro",
                        help="Comma-separated permutation of {ro,en,es,fa,hi}")
    parser.add_argument("--max-steps-per-day", type=int, default=-1,
                        help="Cap steps per day for dry-runs. -1 = use config.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip per-day vLLM eval (smoke-test mode)")
    parser.add_argument("--start-from", type=int, default=1,
                        help="1-indexed day to start from. Resumes from that day's "
                             "predecessor's final/ checkpoint.")
    parser.add_argument("--keep-checkpoints", choices=["last", "all"], default="last",
                        help="last: only keep the most recent day's final/ "
                             "(deletes prev day's after the new day finishes "
                             "training+eval). all: keep every day's final/ (~14 GB each).")
    args, extra_cli = parser.parse_known_args()

    config = load_yaml_config(args.config)
    base_training_args, cfg = build_training_args(config, extra_cli)
    # cfg already has CLI overrides for CONFIG_ONLY_KEYS merged in; only fall
    # back to output_dir if neither YAML nor CLI provided runs_root.
    cfg.setdefault("runs_root", base_training_args.output_dir)

    order = parse_language_order(args.language_order)
    print(f"Language order: {' → '.join(order)}")
    print(f"Runs root: {cfg['runs_root']}")
    print(f"Dataset dir: {cfg['dataset_dir']}")

    raw = load_from_disk(cfg["dataset_dir"])
    missing = [k for lang in order for k in (f"train_{lang}", f"val_{lang}", f"test_{lang}")
               if k not in raw]
    if missing:
        raise SystemExit(f"DatasetDict missing splits: {missing}. "
                         f"Run prepare_data.py with --per-language-splits.")

    # Resume: if --start-from > 1, point prev_ckpt at day(start-1)'s final/.
    prev_ckpt: str | None = None
    if args.start_from > 1:
        prev_lang = order[args.start_from - 2]
        prev_ckpt = str(Path(cfg["runs_root"]) / f"day{args.start_from - 1}_{prev_lang}" / "final")
        if not Path(prev_ckpt).is_dir():
            raise SystemExit(f"--start-from={args.start_from} but {prev_ckpt} doesn't exist.")
        print(f"Resuming: prev checkpoint = {prev_ckpt}")

    for day_idx, lang in enumerate(order, start=1):
        if day_idx < args.start_from:
            continue
        final_dir = run_one_day(
            day_idx=day_idx, lang=lang, prev_ckpt=prev_ckpt,
            cfg=cfg, base_training_args=base_training_args, raw=raw,
            max_steps_per_day=args.max_steps_per_day,
        )
        eval_ok = True
        if not args.skip_eval:
            eval_ok = run_eval(final_dir, cfg["dataset_dir"], f"day{day_idx}_{lang}")

        # Disk hygiene: drop the previous day's full directory now that the
        # new day's final/ exists, has been used to init this day, AND we've
        # finished eval for this day. The metrics JSONs live under
        # pipeline/data/eval_results/ and are independent of the checkpoint.
        # Skip cleanup if THIS day's eval failed — we want to keep the prev
        # day's checkpoint chain intact so eval can be retried later.
        if args.keep_checkpoints == "last" and prev_ckpt is not None and eval_ok:
            prev_day_dir = Path(prev_ckpt).parent
            if prev_day_dir.exists():
                shutil.rmtree(prev_day_dir, ignore_errors=True)
                print(f"[day{day_idx}_{lang}] removed previous day's dir: {prev_day_dir}")

        prev_ckpt = str(final_dir)

    print(f"\nAll {len(order)} days complete. Final ckpt: {prev_ckpt}")


if __name__ == "__main__":
    main()
