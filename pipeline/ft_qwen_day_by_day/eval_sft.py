"""Evaluate an SFT checkpoint on the per-language presentation test splits.

Runs the same chat-template prompt used at training (system + user with
`[MASK]`, `enable_thinking=False`), then scores model output against the
*set* of gold labels for each row. This matches the supervision contract:
`labels` is the set of distinct affective expressions that fill the mask
positions, so we check whether each gold label (single word OR idiomatic
phrase) appears in the model output.

Metrics per language:
  - set_acc@k         : fraction of rows where at least one of the top-k
                        completions contains every gold label (normalized
                        substring match).
  - set_coverage@k    : mean fraction of gold labels found in the best of
                        the top-k completions.
  - acc@k / mrr / sim@k : classical MASIVE-style single-word metrics
                        scored against labels[0] only, for direct
                        comparability with the zero-shot report.

Usage:
    python -m pipeline.ft_qwen_mixed.eval_sft \\
        --checkpoint /local/nlp/aij2115/runs/final \\
        --split test

Writes `pipeline/data/eval_results/sft_<tag>_test_<lang>_metrics.json` per
language and a combined `..._all_metrics.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from pipeline.eval.eval_generative import parse_single_word
from pipeline.eval.metrics import (
    ContextualSimilarityScorer,
    compute_all_metrics,
    normalize_prediction,
)
from pipeline.ft_qwen_mixed.prompts import build_messages

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "pipeline" / "data" / "eval_results"
LANGUAGES = ("ro", "en", "es", "fa", "hi")


def render_prompt(tokenizer, input_text: str) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            build_messages(input_text), **kwargs, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(build_messages(input_text), **kwargs)


def _contains_token_seq(haystack_tokens: list[str], needle: str) -> bool:
    """Whitespace-token-sequence match: needle's tokens must appear as a
    contiguous subsequence in haystack_tokens. Avoids `sad` matching inside
    `saddled` while still allowing multi-word phrase labels.
    """
    n_toks = needle.split()
    if not n_toks:
        return False
    m = len(n_toks)
    for i in range(len(haystack_tokens) - m + 1):
        if haystack_tokens[i:i + m] == n_toks:
            return True
    return False


def _score_completion(generated_text: str, labels_norm: list[str]) -> dict:
    """Score one completion against the gold label set.

    Matching is on normalized (lowercased, diacritics-stripped, punctuation-
    stripped) whitespace-tokens. Each gold label — single word or multi-word
    phrase — must appear as a contiguous token subsequence of the completion.
    """
    gen_norm = normalize_prediction(generated_text)
    gen_tokens = gen_norm.split()
    matched = [L for L in labels_norm if L and _contains_token_seq(gen_tokens, L)]
    return {
        "gen_norm": gen_norm,
        "matched": matched,
        "n_matched": len(matched),
        "exact_set": len(matched) == len(labels_norm) and len(labels_norm) > 0,
        "coverage": len(matched) / max(1, len(labels_norm)),
    }


def generate_vllm(records: list[dict], checkpoint: str, top_k: int,
                  max_tokens: int, gpu_mem: float) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    llm = LLM(model=checkpoint, gpu_memory_utilization=gpu_mem,
              trust_remote_code=True, max_model_len=2048)

    prompts = [render_prompt(tok, r["input"]) for r in records]
    params = SamplingParams(temperature=0.5, max_tokens=max_tokens, n=top_k)
    outputs = llm.generate(prompts, params)

    out: list[dict] = []
    for r, output in zip(records, outputs):
        sorted_completions = sorted(
            output.outputs,
            key=lambda o: o.cumulative_logprob or 0,
            reverse=True,
        )
        completions_raw = [c.text.strip() for c in sorted_completions[:top_k]]
        out.append(_make_result(r, completions_raw))
    return out


def generate_transformers(records: list[dict], checkpoint: str, top_k: int,
                          max_tokens: int, batch_size: int) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    prompts = [render_prompt(tok, r["input"]) for r in records]
    out: list[dict] = []

    for start in tqdm(range(0, len(records), batch_size), desc="gen batches"):
        batch_rec = records[start: start + batch_size]
        batch_prompts = prompts[start: start + batch_size]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=tok.pad_token_id)
        for i, r in enumerate(batch_rec):
            prompt_len = enc["input_ids"][i].shape[0]
            text = tok.decode(gen[i][prompt_len:], skip_special_tokens=True).strip()
            # Greedy transformers path produces a single completion per row;
            # pad with empty strings so top_k scoring is uniform.
            completions = [text] + [""] * (top_k - 1)
            out.append(_make_result(r, completions))
    return out


def _make_result(r: dict, completions_raw: list[str]) -> dict:
    labels = r.get("labels") or [r["label"]]
    labels_norm = [normalize_prediction(L) for L in labels]

    per_completion = [_score_completion(c, labels_norm) for c in completions_raw]

    # Top-k aggregates: best-over-prefix for set-match and coverage.
    k_values = (1, 3, 5)
    set_acc_at_k: dict[str, float] = {}
    coverage_at_k: dict[str, float] = {}
    for k in k_values:
        prefix = per_completion[:k]
        set_acc_at_k[f"set_acc@{k}"] = float(any(c["exact_set"] for c in prefix))
        coverage_at_k[f"coverage@{k}"] = float(max((c["coverage"] for c in prefix),
                                                   default=0.0))

    # Legacy per-result shape for compute_all_metrics: first-word parses of
    # each completion, compared against labels[0] as the single gold. Keeps
    # direct comparability with the zero-shot report on RO (and on any row
    # whose label set is a single word).
    legacy_preds_raw: list[str] = []
    seen: set[str] = set()
    for c in completions_raw:
        word = parse_single_word(c)
        norm = normalize_prediction(word) if word else ""
        if norm and norm not in seen:
            seen.add(norm)
            legacy_preds_raw.append(word)

    first_gold = labels[0]
    return {
        "id": r["id"],
        "language": r["language"],
        # Legacy fields (read by pipeline.eval.metrics.compute_all_metrics).
        "seed_word": first_gold,
        "gold_normalized": normalize_prediction(first_gold),
        "masked_text": r["input"],
        "predictions_raw": legacy_preds_raw,
        "predictions_normalized": [normalize_prediction(p) for p in legacy_preds_raw],
        # New set-level fields.
        "labels_full": labels,
        "labels_normalized": labels_norm,
        "n_masks": r.get("n_masks", r["input"].count("[MASK]")),
        "completions_raw": completions_raw,
        "per_completion": per_completion,
        **set_acc_at_k,
        **coverage_at_k,
    }


def _aggregate_set_metrics(results: list[dict], ks=(1, 3, 5)) -> dict:
    out: dict = {}
    for k in ks:
        out[f"set_acc@{k}"] = float(np.mean([r[f"set_acc@{k}"] for r in results]))
        out[f"coverage@{k}"] = float(np.mean([r[f"coverage@{k}"] for r in results]))
    # Breakdowns by mask count (1 vs >1) for readability.
    single = [r for r in results if r["n_masks"] == 1]
    multi = [r for r in results if r["n_masks"] > 1]
    if single:
        out["single_mask"] = {
            "n": len(single),
            "set_acc@1": float(np.mean([r["set_acc@1"] for r in single])),
        }
    if multi:
        out["multi_mask"] = {
            "n": len(multi),
            "set_acc@1": float(np.mean([r["set_acc@1"] for r in multi])),
            "coverage@1": float(np.mean([r["coverage@1"] for r in multi])),
        }
    return out


def evaluate(checkpoint: str, dataset_dir: Path, backend: str, top_k: int,
             tag: str, compute_similarity: bool, batch_size: int, gpu_mem: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dsd = load_from_disk(str(dataset_dir))

    all_records: list[dict] = []
    per_lang_records: dict[str, list[dict]] = {}
    for lang in LANGUAGES:
        key = f"test_{lang}"
        if key not in dsd:
            print(f"skipping missing split {key}")
            continue
        recs = [dict(r) for r in dsd[key]]
        per_lang_records[lang] = recs
        all_records.extend(recs)

    if backend == "vllm":
        results = generate_vllm(all_records, checkpoint, top_k,
                                max_tokens=50, gpu_mem=gpu_mem)
    else:
        results = generate_transformers(all_records, checkpoint, top_k,
                                        max_tokens=50, batch_size=batch_size)

    by_id = {r["id"]: r for r in results}

    scorer = None
    if compute_similarity:
        print("Loading similarity scorer ...")
        scorer = ContextualSimilarityScorer()

    summary: dict[str, dict] = {}
    for lang in LANGUAGES:
        if lang not in per_lang_records:
            continue
        lang_results = [by_id[r["id"]] for r in per_lang_records[lang] if r["id"] in by_id]

        legacy = compute_all_metrics(lang_results, scorer=scorer)
        set_metrics = _aggregate_set_metrics(lang_results)

        metrics = {
            "language": lang,
            "n_samples": len(lang_results),
            **set_metrics,
            "legacy_first_label": legacy,  # acc@k, mrr, sim@k on labels[0]
        }

        per_lang_path = RESULTS_DIR / f"sft_{tag}_test_{lang}_metrics.json"
        with per_lang_path.open("w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        preds_path = RESULTS_DIR / f"sft_{tag}_test_{lang}_predictions.jsonl"
        with preds_path.open("w") as f:
            for r in lang_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[lang] = metrics
        print(
            f"[{lang}] n={metrics['n_samples']}  "
            f"set_acc@1={metrics['set_acc@1']:.4f}  "
            f"coverage@1={metrics['coverage@1']:.4f}  "
            f"legacy_acc@1={legacy.get('acc@1', 0):.4f}  "
            f"mrr={legacy.get('mrr', 0):.4f}"
        )

    combined_path = RESULTS_DIR / f"sft_{tag}_all_metrics.json"
    with combined_path.open("w") as f:
        json.dump({"checkpoint": checkpoint, "per_language": summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved combined metrics → {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SFT checkpoint (e.g. /local/nlp/aij2115/runs/final)")
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("/local/nlp/aij2115/data/asi_multilingual"),
                        help="DatasetDict from prepare_data.py")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tag", default="qwen3.5-4b_v1",
                        help="Suffix used in output filenames")
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset_dir,
        backend=args.backend,
        top_k=args.top_k,
        tag=args.tag,
        compute_similarity=not args.no_similarity,
        batch_size=args.batch_size,
        gpu_mem=args.gpu_mem,
    )


if __name__ == "__main__":
    main()
