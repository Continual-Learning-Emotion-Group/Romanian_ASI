"""Hyperparameters for mT5 fine-tuning on Romanian ASI.

Values follow MASIVE Appendix D (Deas et al. 2024) verbatim except where the
lab GPU guide or our single-span masking choice forces a deviation. Any
deviation from MASIVE is annotated.
"""

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "pipeline" / "data"
SPLITS_DIR = DATA_DIR / "splits"


@dataclass
class TrainConfig:
    # Model ---------------------------------------------------------------
    model_name_or_path: str = "google/mt5-large"          # 1.2B params

    # Data ----------------------------------------------------------------
    train_file: Path = SPLITS_DIR / "train.jsonl"
    val_file:   Path = SPLITS_DIR / "val.jsonl"
    max_input_tokens:  int = 512                          # MASIVE
    max_target_tokens: int = 32                           # MASIVE

    # Optimization (MASIVE Appendix D) ------------------------------------
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 4                  # MASIVE = 4
    per_device_eval_batch_size: int = 16                  # larger for eval (no grads)
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0                             # MASIVE: HF defaults, no warmup

    # Precision (lab GPU guide) -------------------------------------------
    bf16: bool = True                                     # A6000 supports bf16
    tf32: bool = True                                     # matmul TF32 on

    # Logging & checkpointing ---------------------------------------------
    output_dir: Path = Path("runs/mt5-large-ro-asi")
    logging_steps: int = 50
    eval_steps: int = 2000
    save_steps: int = 2000
    save_total_limit: int = 3
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Dataloader ----------------------------------------------------------
    dataloader_num_workers: int = 4
    preprocess_num_workers: int = 8

    # Reproducibility -----------------------------------------------------
    seed: int = 42
