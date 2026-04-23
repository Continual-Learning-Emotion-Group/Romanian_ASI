# Romanian ASI — Multilingual SFT Run #1 on Qwen3.5-4B-Instruct

## Context

Goal: first supervised fine-tuning experiment on the presentation splits. Train Qwen3.5-4B's post-trained (instruct) model on all 5 language train sets (ro/en/es/fa/hi, 5×1000=5000 rows) scrambled together, evaluate on each language's 1000-row test set. Purpose is to see what happens with very little data before committing to the curriculum-by-language setup. Work on `piranha.cs.columbia.edu` (4× A100-40GB, NVLink, ~347 GB free on `/local`). Follow the NLP Lab GPU Guide: everything under `/local/nlp/aij2115/`, BF16 + Flash Attention 2 + TF32, DeepSpeed ZeRO-3 allowed (NVLink present).

Prior decisions in this conversation:
- Model: `Qwen/Qwen3.5-4B` (post-trained; no `-Instruct` suffix in 3.5 naming).
- Do **not** modify or patch the `day-in-the-life` repo. `continual_learning_loop` is VLM-only and doesn't orchestrate text-only single-task SFT, and the user wants day-in-the-life left untouched. This run is written directly against HF `Trainer` inside `Romanian_ASI`. We'll revisit integration for the follow-up curriculum-by-language run.
- Data: presentation CSVs (1000/{200-250}/1000 per language — hi/val is 200, others 250). ro/en/es/fa were sampled by Parmida from `pipeline/data/benchmark_ro_asi_clean.jsonl` (verified earlier in conversation). hi was converted from `presentation_data/hindi_samples_final/` by `scripts/convert_hindi_to_presentation_format.py` (whole-token masking, label = inflected form as it appears in text). Hindi label space = 153 inflected forms with some source-data noise (short stems like `डर`/fear substring-matching loanwords like `किंडरगार्टेन`/kindergarten) — known and accepted as training noise.
- Multi-mask rows (en/es/fa only; ro and hi are 100% single-mask): supervision is **` `.join(labels)` for every row**, because `labels` is the set of *distinct* affective expressions that fill the mask positions (in order of first appearance). Matched rows (`n_masks == len(labels)`) become one-word-per-mask; mismatched rows (`n_masks > len(labels)`, e.g. "I feel [MASK]. Or rather I am [MASK]" with `labels=['unfit']`) supervise the single expression since it covers both masks; FA phrase labels like `'دلم تنگ شده'` stay as multi-word idioms. No row is dropped; no row carries contradictory signal. See §v2 addendum for why this replaced the earlier "fall back to labels[0]" rule.
- EN/ES/FA test sets contain multi-mask rows too (135, 100, 167 of 1000 respectively). Evaluation is set-level — see §v2 addendum.
- Prompt: chat template with `enable_thinking=False`, assistant turn = ` `.join(labels) + `<|im_end|>`, loss masked on prompt tokens. System prompt describes the set-emission contract (see `prompts.py::SYSTEM_PROMPT`).
- Evaluation: **set-match** scoring (each gold label — single word or phrase — must appear as a contiguous whitespace-token subsequence of the completion). Legacy MASIVE metrics (acc@k, MRR, sim@k on `labels[0]`) are still emitted for direct comparison against the zero-shot report on ro (where `labels` is always length-1).

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
- Read `presentation_data/presentation_{ro,en,es,fa,hi}/{train,val,test}.csv` (accepts both the `presentation_<lang>/` and `<lang>/` layouts via an `--input` dir).
- Add `language` column (`ro|en|es|fa|hi`).
- Concatenate all train sets; `shuffle(seed=42)` → 5000 rows. This is the "scramble" per user instruction.
- Keep val concatenated (~1200 rows; hindi val is 200 rows vs 250 for the others) + per-language `test_<lang>` splits (5×1000).
- `labels` column contains a JSON list like `['uluita']`, `['loved', 'disgusted']`, or `['دلم تنگ شده']`. Supervision is always ` `.join(labels) — see §v2 addendum for the rationale.
- Row fields in the DatasetDict: `id, input, label (supervision string), labels (full list), n_masks, language`.
- Save once via `DatasetDict.save_to_disk("/local/nlp/aij2115/data/asi_multilingual/")`.
- Hindi prep is handled out-of-band by `scripts/convert_hindi_to_presentation_format.py`; `prepare_data.py` consumes the already-normalized `presentation_hi/` CSVs just like the other 4 languages.

### Prompt format (`pipeline/train/prompts.py`)
- System: `"You are an affective-state identifier. Given a sentence with one or more [MASK] positions, output the distinct affective expressions that fill them, in order of first appearance, separated by a single space. Expressions may be single words or short idiomatic phrases. No explanation, no punctuation, no repeats."`
- User: raw `input` column (already contains `[MASK]`).
- Assistant: ` `.join(labels).
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
- `learning_rate: 1.0e-5`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.03` (lowered from 2e-5/0.05 after the v1 warmup-eval-loss spike; see §v2 addendum)
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
- Point at `/local/nlp/aij2115/runs/final/`.
- Run inference (vLLM if available, else transformers) on each language's test split.
- **Set-match scoring**: normalize completions (lowercase, diacritics stripped, punctuation stripped), split on whitespace. For each gold label — single word OR multi-word phrase — check whether its token sequence appears contiguously in the completion. Top-k is taken over N sampled completions; best-over-prefix is the per-row score.
  - `set_acc@k` = fraction of rows where at least one top-k completion covers the full gold set.
  - `coverage@k` = mean fraction of gold labels matched by the best of top-k.
- **Legacy metrics** (`legacy_first_label`): `acc@1/3/5`, `MRR`, `sim@k` computed against `labels[0]` via the existing `pipeline/eval/metrics.py` — for direct apples-to-apples comparison with the zero-shot Qwen3.5-9B numbers on ro (where `len(labels) == 1` always).
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

## v2 addendum (supervision + eval rewrite after v1 diagnosis)

The first full run on piranha showed eval loss spiking during warmup (1.377 → 1.639 peak at step ~75) and plateauing around 1.45 while train loss oscillated 1.7–2.2 with grad norms 25–40 (clipped every step). The cause was **not** bad data — it was a semantic misread of the data in the v1 setup:

1. `labels` is the **set of distinct affective expressions** that fill the `[MASK]` positions (in order of first appearance), not a positional enumeration. Real examples:
   - EN `en_train_4262`: `n_masks=2, labels=['fit']` — the word "fit" is masked at two positions.
   - EN `en_train_37081`: `n_masks=3, labels=['depressed','unfocused']` — 3 mask positions, 2 unique expressions (one repeats). We cannot recover the position→label map without the original unmasked text, but the set is well-defined.
   - FA `fa_oscar_12925`: `n_masks=2, labels=['افسردگی']` — same Persian word masked twice.
2. **FA has multi-word phrase labels** (e.g. `'دلم تنگ شده'`, 3-word Persian idiom "my heart became tight" = "I miss"). 536/1000 FA train supervision tokens come from phrase labels; 204/1000 FA test rows have the phrase `'دلم تنگ شده'` as gold. These are single idiomatic expressions that occupy one `[MASK]` span.
3. The v1 `SYSTEM_PROMPT` said `"respond with only the single word"` — directly contradicting (a) all FA phrase rows and (b) all matched-multi-mask EN/ES rows.
4. The v1 `_supervision_target` fell back to `labels[0]` when shapes mismatched, throwing away half the signal on ~430 rows of EN/ES/FA (~8.6% of the train set).
5. v1 eval `parse_single_word` took the first whitespace token, so every FA row with a phrase gold was an automatic miss regardless of model output.

**v2 decisions (applied, with unit tests):**
- **Supervision**: `_supervision_target = " ".join(labels)` for every row. Handles matched multi-mask, repeated-label multi-mask, single-mask single-word, and FA phrase labels uniformly.
- **Prompt**: rewritten to "output the distinct affective expressions that fill them, in order of first appearance, separated by a single space. Expressions may be single words or short idiomatic phrases."
- **Eval**: substring-style **word-sequence** set-match — for each gold label (word or phrase), its whitespace-tokens must appear contiguously in the normalized completion. Reports `set_acc@k`, `coverage@k`, plus legacy `acc@k/mrr/sim@k` on `labels[0]` for backward-compat with the zero-shot report. Verified: avoids false positives like `'sad' ∈ 'saddled'` while still accepting single-word labels inside multi-word outputs and full phrase matches.
- **LR**: 2e-5 → 1e-5, warmup 0.05 → 0.03. The v1 eval spike during warmup and sustained grad-norm clipping (25–40x) suggested the step size was too aggressive for full-FT of a 4B model on 5000 rows.
- **Decision deferred**: *Option B* — reconstructing per-mask positional alignment from the original unmasked source text. Would give true positional supervision for every row but requires an out-of-band join against the pre-masking corpora. Revisit if v2 metrics plateau.

Sanity coverage (`prompts.py --sanity` now runs 4 cases):
- RO single-mask single-word
- EN multi-mask matched (`n_masks==len(labels)`)
- EN multi-mask repeated-label (`n_masks>len(labels)`, set-semantics)
- FA single-mask multi-word phrase label
