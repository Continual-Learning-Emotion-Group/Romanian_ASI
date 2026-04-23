"""Hyperparameters for Qwen3.5-4B fine-tuning on Romanian ASI.

Mirrors `pipeline/ft_mt5/config.py`. Data and seed match exactly so the two
runs are comparable on the same splits. Optimizer/LR/precision are adapted for
a 5x larger decoder-only causal LM:

  * AdamW (Qwen-family default) instead of Adafactor (mT5 pretraining default).
  * lr 2e-5 (standard 5x drop for larger causal LMs vs mT5's 1e-4).
  * Cosine LR with 5% warmup instead of linear-no-warmup.
  * SDPA attention (dispatches to Flash Attention 2 kernels on A100 + bf16).
  * max_input_tokens 1024 because Qwen BPE is denser than mT5 sentencepiece.
  * max_target_tokens 16 because the target is a single Romanian word.
"""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "pipeline" / "data"
SPLITS_DIR = DATA_DIR / "splits"


@dataclass
class TrainConfig:
    # Model ---------------------------------------------------------------
    model_name_or_path: str = "Qwen/Qwen3.5-4B"
    # `sdpa` dispatches to Flash Attention 2 kernels under the hood on A100
    # + bf16 + small head dims, and avoids the standalone `flash-attn` package
    # (which needs nvcc + ninja + a 15-min build from source).
    attn_implementation: str = "sdpa"

    # Data ----------------------------------------------------------------
    train_file: Path = SPLITS_DIR / "train.jsonl"
    val_file:   Path = SPLITS_DIR / "val.jsonl"
    max_input_tokens:  int = 1024
    max_target_tokens: int = 16

    # Optimization (adapted from MASIVE Appendix D for a 4B causal LM) ----
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8                  # eff. batch 32 on 1 GPU
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # Precision -----------------------------------------------------------
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True

    # Logging & checkpointing ---------------------------------------------
    output_dir: Path = Path("runs/qwen3.5-4b-ro-asi")
    logging_steps: int = 20
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 1                             # tigerfish /local is tight
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Dataloader ----------------------------------------------------------
    dataloader_num_workers: int = 4
    preprocess_num_workers: int = 8

    # Reproducibility -----------------------------------------------------
    seed: int = 42

    # CLI-only (not hyperparameters) --------------------------------------
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    report_to: str = "none"
    run_name: str = "qwen3.5-4b-ro-asi-v1"
