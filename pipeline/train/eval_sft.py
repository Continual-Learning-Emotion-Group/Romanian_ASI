"""Evaluate an SFT checkpoint on the per-language presentation test splits.

Runs the same chat-template prompt used at training (system + user with
`[MASK]`, `enable_thinking=False`), parses the single-word prediction, and
computes MASIVE-style metrics (acc@k, MRR, sim@k) via the existing
`pipeline/eval/metrics.py`.

Usage:
    python -m pipeline.train.eval_sft \\
        --checkpoint /local/nlp/aij2115/runs/final \\
        --split test

Writes `pipeline/data/eval_results/sft_<tag>_test_<lang>_metrics.json` per
language and a combined `..._all_metrics.json`.
"""
import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from pipeline.eval.eval_generative import parse_single_word
from pipeline.eval.metrics import (
    ContextualSimilarityScorer,
    compute_all_metrics,
    normalize_prediction,
)
from pipeline.train.prompts import build_messages

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
        preds_raw: list[str] = []
        seen: set[str] = set()
        for comp in sorted_completions:
            word = parse_single_word(comp.text.strip())
            norm = normalize_prediction(word) if word else ""
            if norm and norm not in seen:
                seen.add(norm)
                preds_raw.append(word)
            if len(preds_raw) >= top_k:
                break
        out.append(_make_result(r, preds_raw, sorted_completions[0].text.strip()
                                if sorted_completions else ""))
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
            word = parse_single_word(text)
            preds_raw = [word] if word else []
            out.append(_make_result(r, preds_raw, text))
    return out


def _make_result(r: dict, preds_raw: list[str], generated_text: str) -> dict:
    gold_norm = normalize_prediction(r["label"])
    return {
        "id": r["id"],
        "language": r["language"],
        "seed_word": r["label"],
        "gold_normalized": gold_norm,
        "masked_text": r["input"],
        "predictions_raw": preds_raw,
        "predictions_normalized": [normalize_prediction(p) for p in preds_raw],
        "generated_text": generated_text,
    }


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
                                max_tokens=20, gpu_mem=gpu_mem)
    else:
        results = generate_transformers(all_records, checkpoint, top_k,
                                        max_tokens=20, batch_size=batch_size)

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

        metrics = compute_all_metrics(lang_results, scorer=scorer)
        metrics["language"] = lang
        metrics["n_samples"] = len(lang_results)

        per_lang_path = RESULTS_DIR / f"sft_{tag}_test_{lang}_metrics.json"
        with per_lang_path.open("w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        preds_path = RESULTS_DIR / f"sft_{tag}_test_{lang}_predictions.jsonl"
        with preds_path.open("w") as f:
            for r in lang_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[lang] = metrics
        print(f"[{lang}] n={metrics['n_samples']}  acc@1={metrics.get('acc@1', 0):.4f}  "
              f"mrr={metrics.get('mrr', 0):.4f}")

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
