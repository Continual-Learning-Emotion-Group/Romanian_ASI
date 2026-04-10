"""Zero-shot MLM evaluation — fill [MASK] with encoder model's mask head.

Models: Romanian BERTs, XLM-RoBERTa, ModernBERT, RoBERTa (English).

Usage:
    python -m pipeline.eval.eval_mlm --model dumitrescustefan/bert-base-romanian-cased-v1 --split test
    python -m pipeline.eval.eval_mlm --model FacebookAI/xlm-roberta-large --split unseen
    python -m pipeline.eval.eval_mlm --model answerdotai/ModernBERT-large --split test --translated
    python -m pipeline.eval.eval_mlm --model FacebookAI/roberta-large --split test --translated --batch-size 64
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from pipeline.eval.metrics import (
    ContextualSimilarityScorer,
    compute_all_metrics,
    compute_metrics_by_group,
    normalize_prediction,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = DATA_DIR / "eval_results"

MODEL_SHORT_NAMES = {
    "dumitrescustefan/bert-base-romanian-cased-v1": "ro-bert",
    "readerbench/RoBERT-base": "robert-base",
    "FacebookAI/xlm-roberta-large": "xlm-r-large",
    "answerdotai/ModernBERT-large": "modernbert-large",
    "FacebookAI/roberta-large": "roberta-large",
}


def get_short_name(model_id: str) -> str:
    return MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])


def load_split(split: str, translated: bool = False) -> list[dict]:
    if translated:
        path = SPLITS_DIR / f"{split}_translated_en.jsonl"
    else:
        path = SPLITS_DIR / f"{split}.jsonl"

    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if translated:
                # Remap English fields to standard names
                r["_original_text"] = r["text"]
                r["text"] = r["text_en"]
                r["masked_text"] = r["masked_text_en"]
                r["seed_word"] = r.get("seed_word_en", r["seed_word"])
                r["seed_word_normalized"] = normalize_prediction(r["seed_word"])
            records.append(r)
    return records


def replace_mask_token(masked_text: str, tokenizer) -> str:
    """Replace literal [MASK] with the model's actual mask token."""
    return masked_text.replace("[MASK]", tokenizer.mask_token)


def is_whole_word(token_str: str, tokenizer) -> bool:
    """Check if a predicted token is a whole word (not a subword continuation).

    Returns True for word-initial / standalone tokens.
    """
    # Skip empty or whitespace-only
    stripped = token_str.strip()
    if not stripped:
        return False

    # BERT-style: subwords start with ##
    if stripped.startswith("##"):
        return False

    # SentencePiece (XLM-R, RoBERTa): word-initial tokens start with ▁ (U+2581)
    # or the decoded token starts with a space.  Continuation tokens lack this.
    if hasattr(tokenizer, "sp_model") or "sentencepiece" in str(type(tokenizer)).lower():
        # For SentencePiece tokenisers, word-initial tokens typically start with ▁
        if token_str.startswith("▁") or token_str.startswith(" "):
            return True
        # Single character tokens that are common words can be standalone
        # but we keep them only if they look like real words
        if len(stripped) <= 1:
            return False
        # If no ▁ prefix, it's likely a continuation — but some tokenisers
        # don't use ▁ at all. Fall through to heuristic below.

    # GPT-style (ModernBERT uses GPT-2 tokeniser): word-initial tokens start with Ġ
    if token_str.startswith("Ġ"):
        return True

    # Heuristic fallback: if the token is alphabetic and reasonably long, keep it
    if stripped.isalpha() and len(stripped) >= 2:
        return True

    return False


def predict_batch(
    texts: list[str],
    model,
    tokenizer,
    top_k: int = 50,
    device: str = "cuda",
) -> list[list[tuple[str, float]]]:
    """Run MLM inference on a batch. Returns top-k (token_str, log_prob) per sample."""
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=getattr(model.config, "max_position_embeddings", 512),
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits  # (batch, seq_len, vocab)

    mask_token_id = tokenizer.mask_token_id
    results = []

    for i in range(len(texts)):
        input_ids = enc["input_ids"][i]
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            results.append([])
            continue

        pos = mask_positions[0].item()
        token_logits = logits[i, pos]
        log_probs = torch.log_softmax(token_logits, dim=-1)

        topk_vals, topk_ids = torch.topk(log_probs, top_k)
        predictions = []
        for val, tid in zip(topk_vals.tolist(), topk_ids.tolist()):
            token_str = tokenizer.decode([tid])
            predictions.append((token_str, val))

        results.append(predictions)

    return results


def filter_subwords(
    predictions: list[tuple[str, float]],
    tokenizer,
    max_keep: int = 50,
) -> tuple[list[tuple[str, float]], int]:
    """Filter out subword continuations. Returns (filtered, n_filtered)."""
    filtered = []
    n_filtered = 0
    for token_str, score in predictions:
        if is_whole_word(token_str, tokenizer):
            filtered.append((token_str.strip("▁Ġ "), score))
            if len(filtered) >= max_keep:
                break
        else:
            n_filtered += 1
    return filtered, n_filtered


def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            done.add(r["id"])
    return done


def run_evaluation(
    model_id: str,
    split: str,
    translated: bool = False,
    top_k: int = 50,
    batch_size: int = 32,
    compute_similarity: bool = True,
    resume: bool = False,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    short = get_short_name(model_id)
    lang = "en" if translated else "ro"
    prefix = f"mlm_{short}_{split}_{lang}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{prefix}_results.jsonl"
    metrics_path = RESULTS_DIR / f"{prefix}_metrics.json"
    checkpoint_path = RESULTS_DIR / f"{prefix}_checkpoint.jsonl"

    print(f"Model:  {model_id}")
    print(f"Split:  {split} ({'translated EN' if translated else 'Romanian'})")
    print(f"Device: {device}")

    # Load data
    records = load_split(split, translated=translated)
    print(f"Loaded {len(records)} samples")

    # Resume
    done_ids = set()
    completed_results = []
    if resume and checkpoint_path.exists():
        done_ids = load_checkpoint(checkpoint_path)
        with open(checkpoint_path) as f:
            completed_results = [json.loads(line) for line in f]
        print(f"Resuming: {len(done_ids)} already done")

    remaining = [r for r in records if r["id"] not in done_ids]
    if not remaining:
        print("All samples already processed.")
        return

    # Load model
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device).eval()

    # Process in batches
    all_results = list(completed_results)
    checkpoint_f = open(checkpoint_path, "a")

    try:
        for batch_start in tqdm(range(0, len(remaining), batch_size), desc="Batches"):
            batch_records = remaining[batch_start : batch_start + batch_size]
            texts = [
                replace_mask_token(r["masked_text"], tokenizer)
                for r in batch_records
            ]

            raw_preds = predict_batch(texts, model, tokenizer, top_k=top_k, device=device)

            for r, preds in zip(batch_records, raw_preds):
                filtered, n_sub = filter_subwords(preds, tokenizer)

                preds_raw = [t for t, _ in filtered]
                preds_norm = [normalize_prediction(t) for t, _ in filtered]

                result = {
                    "id": r["id"],
                    "seed_word": r["seed_word"],
                    "seed_word_normalized": r["seed_word_normalized"],
                    "gold_normalized": r["seed_word_normalized"],
                    "source": r.get("source", ""),
                    "pattern_category": r.get("pattern_category", ""),
                    "emotion_category": r.get("emotion_category", []),
                    "masked_text": r["masked_text"],
                    "predictions_raw": preds_raw[:20],
                    "predictions_normalized": preds_norm[:20],
                    "predictions_scores": [s for _, s in filtered[:20]],
                    "subword_filtered": n_sub,
                }

                all_results.append(result)
                checkpoint_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            checkpoint_f.flush()
    finally:
        checkpoint_f.close()

    # Write final results
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute metrics
    print("Computing metrics ...")
    scorer = None
    if compute_similarity:
        print("  Loading similarity scorer (bert-base-multilingual-cased) ...")
        scorer = ContextualSimilarityScorer(device=device)

    metrics = compute_all_metrics(all_results, scorer=scorer)
    metrics["model"] = model_id
    metrics["split"] = split
    metrics["language"] = lang
    metrics["type"] = "mlm"

    # Breakdowns
    metrics["by_source"] = compute_metrics_by_group(all_results, "source")
    metrics["by_pattern"] = compute_metrics_by_group(all_results, "pattern_category")
    metrics["by_gender"] = compute_metrics_by_group(all_results, "gender")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print summary
    print(f"\n=== Results: {short} on {split} ({lang}) ===")
    print(f"  Samples:  {metrics['n_samples']}")
    for k in (1, 3, 5):
        acc = metrics.get(f"acc@{k}", 0)
        line = f"  Acc@{k}:   {acc:.4f} ({acc*100:.1f}%)"
        if scorer:
            sim = metrics.get(f"sim@{k}", 0)
            line += f"   Sim@{k}: {sim:.4f}"
        print(line)
    print(f"  MRR:      {metrics['mrr']:.4f}")
    print(f"\nSaved to {results_path}")
    print(f"Metrics: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot MLM evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--split", required=True, choices=["test", "unseen"])
    parser.add_argument("--translated", action="store_true", help="Use translated EN data")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_evaluation(
        model_id=args.model,
        split=args.split,
        translated=args.translated,
        top_k=args.top_k,
        batch_size=args.batch_size,
        compute_similarity=not args.no_similarity,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
