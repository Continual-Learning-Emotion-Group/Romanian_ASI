"""Entry point for Romanian ASI multilingual SFT on Qwen3.5-4B.

Reads a YAML config (see `configs/qwen3_5_4b_full_ft.yaml`), loads the
precomputed DatasetDict built by `prepare_data.py`, tokenizes with chat
template + loss masking from `prompts.py`, and trains with HF Trainer +
DeepSpeed ZeRO-3.

Usage (local 1-step dry run):
    python -m pipeline.train.train \\
        --config pipeline/train/configs/qwen3_5_4b_full_ft.yaml \\
        --max_steps 2 --per_device_train_batch_size 2

Usage (full multi-GPU run via torchrun — see run/piranha_launch.sh):
    torchrun --nproc_per_node=4 -m pipeline.train.train \\
        --config pipeline/train/configs/qwen3_5_4b_full_ft.yaml
"""
import argparse
import os
import shutil
import sys
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
}


def load_yaml_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_training_args(config: dict, extra_cli: list[str]) -> tuple[TrainingArguments, dict]:
    """Separate training-args keys from config-only keys, then let HF parse.

    `extra_cli` is forwarded so CLI flags like `--max_steps 2` override the YAML.
    """
    config_only = {k: config[k] for k in CONFIG_ONLY_KEYS if k in config}
    hf_args = {k: v for k, v in config.items() if k not in CONFIG_ONLY_KEYS}

    parser = HfArgumentParser(TrainingArguments)
    args_list: list[str] = []
    for k, v in hf_args.items():
        if isinstance(v, bool):
            if v:
                args_list.append(f"--{k}")
        elif isinstance(v, dict):
            args_list += [f"--{k}", str(v)]
        else:
            args_list += [f"--{k}", str(v)]
    args_list += extra_cli

    (training_args,) = parser.parse_args_into_dataclasses(args_list)
    return training_args, config_only


def tokenize_split(ds, tokenizer, max_length: int):
    def _map(example):
        return encode_example(tokenizer, example["input"], example["label"],
                              max_length=max_length)
    # `batched=False` keeps the logic simple; 5000 rows is tiny.
    keep_cols = {"input_ids", "attention_mask", "labels"}
    return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in keep_cols])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, extra_cli = parser.parse_known_args()

    config = load_yaml_config(args.config)
    training_args, cfg = build_training_args(config, extra_cli)

    model_name = cfg["model_name_or_path"]
    dataset_dir = cfg["dataset_dir"]
    max_length = cfg.get("max_seq_length", 1024)
    final_dir = cfg.get("final_dir")

    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        print(f"Model:        {model_name}")
        print(f"Dataset dir:  {dataset_dir}")
        print(f"Output dir:   {training_args.output_dir}")
        print(f"Final dir:    {final_dir}")
        print(f"World size:   {int(os.environ.get('WORLD_SIZE', '1'))}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required with gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    raw = load_from_disk(dataset_dir)
    train_ds = tokenize_split(raw["train"], tokenizer, max_length)
    val_ds = tokenize_split(raw["val"], tokenizer, max_length)

    if is_main:
        ex = train_ds[0]
        print(f"First train example: "
              f"input_ids len={len(ex['input_ids'])}, "
              f"loss tokens={sum(1 for x in ex['labels'] if x != -100)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=LossMaskingCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer=tokenizer,
    )

    trainer.train()

    if is_main and final_dir:
        Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir or training_args.output_dir)
    if is_main:
        tokenizer.save_pretrained(final_dir or training_args.output_dir)
        print(f"Saved final model → {final_dir or training_args.output_dir}")


if __name__ == "__main__":
    main()
