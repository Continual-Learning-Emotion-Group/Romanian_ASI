# Qwen3.5-4B SFT — run log & resume notes

Snapshot from session 2026-04-23. The pipeline scaffold is committed on
`qwen-finetune` (branched from `mt5-finetune`); a tigerfish dry-run has been
executed and several dependency / architecture issues surfaced. **No full
training run was started.** This file records what works, what doesn't, and
what to do next time.

## Where we stopped

- Worktree at `/Users/alexjerpelea/Romanian_ASI_qwen_finetune`, branch
  `qwen-finetune` (commits `0e002e4` → `ad45fc0`, all pushed to origin).
- `pipeline/ft_qwen/` module scaffolded: `config.py`, `prompts.py`,
  `build_training_data.py`, `train.py`, `launch_tigerfish.sh`, `README.md`.
- Splits inherited from `mt5-finetune` unchanged: 45,181 / 2,658 / 5,315
  (test seed-word disjoint from train/val).
- Tokenizer-level loss-masking sanity check passes (`prompts.py --sanity`):
  prompt + empty `<think></think>` block is masked to -100, only the seed
  word + `<|im_end|>` is supervised.
- Dry-run on tigerfish GPU 1 reaches `train()` and produces metrics, but
  surfaces two blockers (see "Open issues" below) so the full run was not
  started.

## Tigerfish environment that worked (for resume)

Project root: `/local/nlp/aij2115/ro_asi_ft/`

```
ro_asi_ft/
├── repo/        # git clone, branch=qwen-finetune
├── venv/        # python 3.12 venv
├── hf_cache/    # HF_HOME (deleted at end of session — needs re-download next time)
└── runs/        # training output (cleaned up)
```

Final dependency set inside `venv/`:
- `torch == 2.11.0+cu128` (installed via `pip install --index-url https://download.pytorch.org/whl/cu128 torch`)
- `transformers == 5.6.0` (default `5.0.0` from requirements.txt does NOT recognize the `qwen3_5` model_type — must upgrade)
- `accelerate == 1.13.0`, `wandb == 0.26.0`, `datasets == 4.5.0`
- `flash-linear-attention == 0.5.0` + `fla-core == 0.5.0`, `einops == 0.8.2`
- `causal-conv1d == 1.6.1` (required source build — see install order below)

Required environment for builds and training:
```bash
export CUDA_HOME=/usr/local/cuda-12        # nvcc 12.9 — fine with cu128 wheels
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/local/nlp/aij2115/ro_asi_ft/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1              # avoid GPU 0 if other user is on it
```

### Install order that works (next session)

The naive `pip install -r requirements.txt` fails because `flash-attn` (since
removed from requirements) needs torch installed during PEP 517 isolated
builds. With our current requirements:

```bash
# 1. base pip + venv
python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip wheel packaging setuptools

# 2. install everything from requirements.txt (cu128-pinned torch from index)
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r repo/requirements.txt   # transformers 5.0.0 will be pinned here

# 3. upgrade transformers (5.0.0 doesn't know qwen3_5)
pip install -U transformers              # pulls 5.6.0+, also bumps huggingface_hub

# 4. linear-attention fast path (Triton-only, fast)
pip install flash-linear-attention

# 5. causal-conv1d — REQUIRES SOURCE BUILD with system CUDA
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
pip install causal-conv1d --no-build-isolation
```

Notes:
- The `--no-build-isolation` flag is essential for causal-conv1d so it picks
  up our cu128 torch instead of trying to provision its own.
- The build takes ~3–5 min and needs ~4 GB of /local during compile.
- `pip install --dry-run flash-linear-attention causal-conv1d` confirms
  resolution but pip will still build conv1d from sdist; there's no prebuilt
  wheel for our (torch 2.11+cu128, py312) combo.

## Dry-run timing observations

All dry runs: 50 train + 10 val rows, single A100-40GB SXM (GPU 1), bf16,
sdpa attn, gradient checkpointing on.

| Run | Setup | Steps × bs | train_runtime | samples/s | Notes |
|---|---|---|---|---|---|
| 1 | no fla, no conv1d | 2 × 2 | 81.6 s | 0.049 | Slow torch fallback for linear-attn |
| 2 | fla only | 2 × 2 | 212.6 s | 0.019 | fla loaded but unhelpful without conv1d — slower |
| 3 | fla + conv1d | 2 × 2 | 66.3 s | 0.060 | Fast path engaged |
| 4 | fla + conv1d | 5 × 4 | 93.6 s | 0.214 | Steady state — warmup amortized |

Extrapolated full run on 45,181 × 3 epochs at 0.214 samples/s:
**~7.3 days** of training on a single A100. Eval is fast (~11 samples/s).

## Open issues that blocked the full run

### 1. Architecture mismatch on checkpoint reload

`Qwen/Qwen3.5-4B` is actually a **vision-language multimodal model**
(`Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`) with a hybrid
attention stack (every 4th layer `full_attention`, the rest `linear_attention`,
plus `image_token_id` and `video_preprocessor_config.json`). Loading via
`AutoModelForCausalLM` returns the text decoder, but on `Trainer.train()` with
`load_best_model_at_end=True`, the saved checkpoint state_dict has every key
prefixed `model.language_model.*` (multimodal wrapper convention) while the
loading model expects un-prefixed keys. Result:

> `[transformers] There were unexpected keys in the checkpoint model loaded:
>  ['model.language_model.embed_tokens.weight', ...426 entries...]`

Effect: the "best" reload silently re-initializes most of the network, so the
final eval metrics and the saved checkpoint are not what was actually trained.

**Workarounds to try next session, in order of preference:**

1. **Switch to `Qwen/Qwen3-4B`** (text-only, standard transformer, fully
   supported by transformers 5.0.0+, no fast-path libs needed). My recommended
   path — see "Recommended pivot" below.
2. Disable `load_best_model_at_end=True` in `config.py` so the trainer keeps
   the in-memory model and skips the broken reload. Doesn't fix root cause but
   produces a usable final checkpoint.
3. Build a custom save/load shim that strips the `model.language_model.`
   prefix on save and adds it on load. Brittle.

### 2. Loss instability

Run 4 logged: 1.56 → 6.75 → 1.51 → 2.83 across 4 logged steps with
`grad_norm` 124–342. That's not normal AdamW + lr=2e-5 + warmup behavior; it
points to either (a) some submodules not being trained / having random init
weights despite the `from_pretrained`, or (b) the loss-masking not lining up
correctly for this architecture's tokenizer (the chat template adds an empty
`<think></think>` block that we mask, but the multimodal model may format
things slightly differently). Either way, didn't get to root-cause.

### 3. Slow training even with fast path

0.214 samples/s ≈ 4.7 s/sample with bs=4 + grad_chk + sdpa + fla fast path.
That extrapolates to ~7 days. By contrast a standard 4B causal LM (e.g.
Qwen3-4B) on the same hardware typically achieves ~0.5–1.0 samples/s for
this seq length, putting the full run at ~2 days.

## Recommended pivot for next session

Switch to **`Qwen/Qwen3-4B`** (text-only standard transformer):

- Supported by transformers 5.0.0 (so no upgrade chain needed if requirements
  pin holds), but works fine with 5.6.0 too.
- No `flash-linear-attention` / `causal-conv1d` dependency (sdpa is enough).
- Standard `LlamaDecoderLayer`-style architecture → no checkpoint key mangling.
- ~3–5× faster per step than Qwen3.5-4B in our current setup.
- Still multilingual (Qwen3 supports 100+ languages including Romanian).
- Full run estimate: **~2 days** instead of 7.

Concrete change is one line in `pipeline/ft_qwen/config.py`:
```python
model_name_or_path: str = "Qwen/Qwen3-4B"   # was "Qwen/Qwen3.5-4B"
```
plus optionally drop the fast-path libs from the install order (steps 4 & 5
above become unnecessary).

If the user insists on Qwen3.5-4B for some downstream reason, address open
issue #1 first with workaround #2 above (disable `load_best_model_at_end`),
do another dry run to confirm loss curves stabilize, then commit to the
~7-day full run.

## Resume checklist

1. `git -C /Users/alexjerpelea/Romanian_ASI worktree list` → confirm
   `Romanian_ASI_qwen_finetune` is still there. If not, recreate:
   `git worktree add /Users/alexjerpelea/Romanian_ASI_qwen_finetune qwen-finetune`.
2. `ssh aij2115@tigerfish.cs.columbia.edu` → check `/local/nlp/aij2115/ro_asi_ft/`.
   If wiped, redo `git clone --branch qwen-finetune ...` and the install
   order above.
3. `nvidia-smi` → pick a free GPU (avoid GPU 0 if another user is on it).
4. `df -h /local` → abort if `<40 G` free.
5. Decide model: probably edit `pipeline/ft_qwen/config.py` to
   `Qwen/Qwen3-4B`, push, pull on tigerfish.
6. Run `python -m pipeline.ft_qwen.train --max-train-samples 50 --max-val-samples 10 --max-steps 5 ...`
   to confirm sane loss curves on the chosen model before committing to the
   full run.
7. Full run: `bash pipeline/ft_qwen/launch_tigerfish.sh`.
