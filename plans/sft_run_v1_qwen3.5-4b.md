# Romanian ASI — Multilingual SFT Run #1 on Qwen3.5-4B-Instruct

## Context

Goal: first supervised fine-tuning experiment on the presentation splits. Train Qwen3.5-4B's post-trained (instruct) model on all 5 language train sets (ro/en/es/fa/hi, 5×1000=5000 rows) scrambled together, evaluate on each language's 1000-row test set. Purpose is to see what happens with very little data before committing to the curriculum-by-language setup. Work on `piranha.cs.columbia.edu` (4× A100-40GB, NVLink, ~347 GB free on `/local`). Follow the NLP Lab GPU Guide: everything under `/local/nlp/aij2115/`, BF16 + Flash Attention 2 + TF32, DeepSpeed ZeRO-3 allowed (NVLink present).

Prior decisions in this conversation:
- Model: `Qwen/Qwen3.5-4B` (post-trained; no `-Instruct` suffix in 3.5 naming).
- Do **not** modify or patch the `day-in-the-life` repo. `continual_learning_loop` is VLM-only and doesn't orchestrate text-only single-task SFT, and the user wants day-in-the-life left untouched. This run is written directly against HF `Trainer` inside `Romanian_ASI`. We'll revisit integration for the follow-up curriculum-by-language run.
- Data: presentation CSVs (1000/250/1000 per language). ro/en/es/fa were sampled by Parmida from `pipeline/data/benchmark_ro_asi_clean.jsonl` (verified earlier in conversation). hi was converted from `presentation_data/hindi_samples_final/` by `scripts/convert_hindi_to_presentation_format.py` (whole-token masking, label = inflected form as it appears in text). Hindi label space = 153 inflected forms with some source-data noise (short stems like `डर`/fear substring-matching loanwords like `किंडरगार्टेन`/kindergarten) — known and accepted as training noise.
- Prompt: chat template with `enable_thinking=False`, assistant turn = single label word + `<|im_end|>`, loss masked on prompt tokens.
- Evaluation: reuse `pipeline/eval/` (MASIVE-style acc@k, MRR, sim@k) for comparable numbers against the existing zero-shot results.

## Repo & VM layout

**Romanian_ASI lives on piranha under `/local/nlp/aij2115/`** (per lab guide — not `$HOME`, which is slow NAS). GitHub is the source of truth; we push from piranha. `day-in-the-life` is **not cloned on piranha for this run** — it's unused.

```
/local/nlp/aij2115/
├── code/
│   └── Romanian_ASI/          # git clone git@github.com:lolismek/Romanian_ASI.git
├── venv/                      # project-specific Python env (lab guide)
├── cache/                     # HF_HOME, TRANSFORMERS_CACHE, TORCH_HOME
├── data/asi_multilingual/     # HF DatasetDict(train,val,test), save_to_disk
└── runs/
    ├── checkpoints/           # Trainer output, save_total_limit=1
    ├── final/                 # final best checkpoint (kept)
    └── logs/                  # wandb offline, tensorboard, per-step JSONL
```

**All source code lives in `Romanian_ASI`.** Self-contained — no external-repo patches.

New files in `Romanian_ASI`:

```
pipeline/train/
├── __init__.py
├── prepare_data.py          # CSV → HF DatasetDict(train, val, test) with language column
├── prompts.py               # chat-template builder, enable_thinking=False, loss-masking collator
├── train.py                 # entry point: loads model, dataset, runs HF Trainer directly
├── configs/
│   ├── qwen3_5_4b_full_ft.yaml     # HF TrainingArguments overrides
│   └── deepspeed_zero3.json        # DeepSpeed config
└── eval_sft.py              # wraps pipeline.eval.eval_generative for the SFT checkpoint
run/
└── piranha_launch.sh        # env setup + torchrun command
```

## Critical design points

### Data (`pipeline/train/prepare_data.py`)
- Read `presentation_data/presentation_{ro,en,es,fa,hi}/{train,val,test}.csv`.
- Add `language` column (`ro|en|es|fa|hi`).
- Concatenate all train sets; `shuffle(seed=42)` → 5000 rows. This is the "scramble" per user instruction.
- Keep val/test per-language (Dataset per language) + a combined Dataset for overall metrics.
- `labels` column contains a JSON list like `['uluita']` — take first element as the target word.
- Save once via `DatasetDict.save_to_disk("/local/nlp/aij2115/data/asi_multilingual/")`.
- Hindi prep is handled out-of-band by `scripts/convert_hindi_to_presentation_format.py`; `prepare_data.py` consumes the already-normalized `presentation_hi/` CSVs just like the other 4 languages.

### Prompt format (`pipeline/train/prompts.py`)
- System: short Romanian/English bilingual instruction, e.g. `"You are an affective-state identifier. Given a sentence with [MASK], respond with only the single word that fills the mask — no explanation, no punctuation."`
- User: raw `input` column (already contains `[MASK]`).
- Assistant: `labels[0]`.
- Apply via `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)`.
- **Loss masking** — custom collator:
  1. Tokenize full conversation.
  2. Tokenize the prompt-only prefix (up to and including `<|im_start|>assistant\n`).
  3. Set labels for prefix tokens to `-100`, keep real IDs for `{label}<|im_end|>`.
- Sanity check: print one tokenized sample at startup, verify `<think>` tokens are absent.

### Model loading (inside `pipeline/train/train.py`)
- `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")`.
- `AutoTokenizer.from_pretrained(..., padding_side="right")` (left-padding only needed at inference).
- Enable gradient checkpointing with `use_reentrant=False` so it composes with ZeRO-3.
- No `device_map` under DeepSpeed — DeepSpeed owns placement.

### Training config (`pipeline/train/configs/qwen3_5_4b_full_ft.yaml`)
Overrides merged into HF `TrainingArguments`:
- `bf16: true`, `tf32: true`
- `gradient_checkpointing: true`, `gradient_checkpointing_kwargs: {use_reentrant: false}`
- `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 2` → effective 32 with 4 GPUs
- `per_device_eval_batch_size: 8`
- `num_train_epochs: 3`
- `learning_rate: 2.0e-5`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.05`
- `eval_strategy: steps`, `eval_steps: 25`
- `save_strategy: steps`, `save_steps: 25`, `save_total_limit: 1`, `load_best_model_at_end: true`, `metric_for_best_model: eval_loss`
- `logging_steps: 5`
- `report_to: wandb`, `run_name: qwen3.5-4b-asi-multi-v1` (wandb offline so we're not leaking; user can sync later)
- `deepspeed: configs/deepspeed_zero3.json`
- `seed: 42`, `data_seed: 42`

DeepSpeed ZeRO-3 config: stage 3, bf16 enabled, reduce_bucket_size auto, `offload_param: {device: none}`, `offload_optimizer: {device: none}` — plenty of VRAM across 4 GPUs, no need to offload to CPU.

Estimated training time: 5000 rows × 3 epochs / 32 ≈ 470 steps. At ~2 s/step on 4× A100 with BF16+FA2+ZeRO-3, ~16 min total.

### Launch (`run/piranha_launch.sh`)
- `export HF_HOME=/local/nlp/aij2115/cache`
- `export TRANSFORMERS_CACHE=...`, `WANDB_DIR=...`, `WANDB_MODE=offline`, `TOKENIZERS_PARALLELISM=false`
- `export CUDA_VISIBLE_DEVICES=0,1,2,3`
- `source /local/nlp/aij2115/venv/bin/activate`
- `torchrun --nproc_per_node=4 -m pipeline.train.train --config pipeline/train/configs/qwen3_5_4b_full_ft.yaml`
- Runs in `tmux` / `nohup` so disconnects don't kill the job.

### Evaluation (`pipeline/train/eval_sft.py`)
- Point `pipeline/eval/eval_generative.py` at `/local/nlp/aij2115/runs/final/`.
- Run inference (vLLM if available, else transformers) on each language's test split.
- Compute acc@1/3/5, MRR, sim@k via the existing `pipeline/eval/metrics.py`.
- Write `pipeline/data/eval_results/sft_qwen3.5-4b_v1_test_{lang}_metrics.json` and a combined `..._all_metrics.json` for easy side-by-side with zero-shot.

## Existing code to reuse

- `pipeline/eval/metrics.py` — all MASIVE-style metrics (acc@k, MRR, sim@k). No changes.
- `pipeline/eval/eval_generative.py` — inference harness. Add a `--checkpoint_path` override; otherwise reuse the generation code.
- `pipeline/eval/report.py` — result aggregation.
- Data paths: `pipeline/data/eval_results/` (existing convention).
- `requirements.txt` — already pins `transformers==5.0.0`, `torch==2.2.2`, `datasets==4.5.0`. Add `deepspeed`, `flash-attn`, `accelerate`, `peft` (unused but harmless), `wandb`, `vllm` (already present).

## Hygiene (per lab guide)

- `save_total_limit=1` so at most one intermediate checkpoint on disk (≈16 GB BF16).
- After run: `rm -rf /local/nlp/aij2115/cache/hub/models--Qwen--Qwen3.5-4B` once the final checkpoint is saved (original weights are embedded).
- `du -sh /local/nlp/aij2115/` logged at end of run.
- Trap SIGINT in the launcher to make sure `torch.cuda.empty_cache()` runs and no idle Python processes linger.
- Use `nvidia-smi` before starting; abort if another user is using all 4 GPUs.

## Verification

1. **Data check (local, fast)**:
   ```bash
   python -m pipeline.train.prepare_data --output /tmp/asi_multilingual
   python -c "from datasets import load_from_disk; d=load_from_disk('/tmp/asi_multilingual'); print(d); print(d['train'][0])"
   ```
   Expect 5000 train / 1250 val / 5000 test rows across 5 languages (language column present, shuffled).

2. **Tokenization smoke test (local)**:
   ```bash
   python -m pipeline.train.prompts --sanity
   ```
   Prints one formatted example, shows which tokens have `-100` labels (prompt) vs real IDs (response + `<|im_end|>`), asserts no `<think>` tokens.

3. **1-step dry-run on piranha** (single GPU, subset of 32 rows):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m pipeline.train.train --config ... --max_steps 2 --per_device_train_batch_size 2
   ```
   Confirms model loads, collator works, one gradient step completes.

4. **Full run**:
   ```bash
   bash run/piranha_launch.sh
   ```
   Monitor via `tmux a` and `tail -f /local/nlp/aij2115/runs/logs/train.log`.

5. **Eval**:
   ```bash
   python -m pipeline.train.eval_sft --checkpoint /local/nlp/aij2115/runs/final --split test
   ```
   Produces per-language metric JSONs. Compare to zero-shot numbers in `pipeline/data/eval_results/`.

6. **Sign-off checklist**:
   - Test acc@1 per language is non-trivial vs random baseline (ro ≈ 1/414, en/es/fa comparable, hi ≈ 1/153).
   - Model emits single-word outputs (spot-check 20 predictions per language).
   - No `<think>` content in predictions.
   - Romanian is competitive with the zero-shot Qwen3.5-9B acc@1 (0.22) from commit `fdda682` — with a 4B model fine-tuned, we'd hope to exceed it.
   - Hindi acc@1 is expected to be higher than the other languages (smaller label space + heavy head); report per-language, don't average.

## Out of scope (explicit, for follow-up runs)

- Curriculum by language (one-task-per-language). If we want to route this through `continual_learning_loop` later, we'll figure out the integration *without* modifying the upstream day-in-the-life repo (e.g., vendor a minimal text-only loop into Romanian_ASI, or fork day-in-the-life under our own account).
- Base-vs-instruct comparison (user chose instruct for this first run).
- Collapsing surface-form labels (`fericit`/`fericită`, `mulțumit`/`multumit`).
- Switching label space to `emotion_category`.
- LoRA / 8-bit optimizer alternatives.
