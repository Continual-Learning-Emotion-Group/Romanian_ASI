"""Zero-shot generative model evaluation — predict masked emotion word.

Supports encoder-decoder (mT5) via transformers and decoder-only LLMs via vLLM.

Usage:
    python -m pipeline.eval.eval_generative --model google/mt5-large --split test --backend transformers
    python -m pipeline.eval.eval_generative --model Qwen/Qwen3.5-9B --split test --backend vllm
    python -m pipeline.eval.eval_generative --model OpenLLM-Ro/RoGemma2-9b-Instruct --split test --backend vllm
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

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
    "google/mt5-large": "mt5-large",
    "Qwen/Qwen3.5-9B": "qwen3.5-9b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "OpenLLM-Ro/RoLlama3.1-8b-Instruct": "rollama3.1-8b",
    "OpenLLM-Ro/RoGemma2-9b-Instruct": "rogemma2-9b",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_RO = """\
Completează cuvântul lipsă care descrie o stare afectivă (emoție, dispoziție sau sentiment).

Text: {masked_text}

Răspunde cu UN SINGUR cuvânt. Cuvântul lipsă este:"""

PROMPT_TEMPLATE_EN = """\
Fill in the missing word that describes an affective state (emotion, mood, or feeling).

Text: {masked_text}

Respond with a SINGLE word. The missing word is:"""


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
                r["_original_text"] = r["text"]
                r["text"] = r["text_en"]
                r["masked_text"] = r["masked_text_en"]
                r["seed_word"] = r.get("seed_word_en", r["seed_word"])
                r["seed_word_normalized"] = normalize_prediction(r["seed_word"])
            records.append(r)
    return records


def truncate_around_mask(masked_text: str, max_chars: int = 3000) -> str:
    """Truncate text keeping a window centered on [MASK]."""
    if len(masked_text) <= max_chars:
        return masked_text
    mask_pos = masked_text.find("[MASK]")
    if mask_pos == -1:
        return masked_text[:max_chars]
    half = max_chars // 2
    start = max(0, mask_pos - half)
    end = min(len(masked_text), mask_pos + half)
    result = masked_text[start:end]
    if start > 0:
        result = "..." + result
    if end < len(masked_text):
        result = result + "..."
    return result


def build_prompt(masked_text: str, translated: bool = False) -> str:
    """Build fill-in-the-blank prompt for decoder-only models."""
    truncated = truncate_around_mask(masked_text)
    display_text = truncated.replace("[MASK]", "___")
    template = PROMPT_TEMPLATE_EN if translated else PROMPT_TEMPLATE_RO
    return template.format(masked_text=display_text)


def parse_single_word(response: str) -> str:
    """Extract the affective state word from model response.

    Handles:
    - <think>...</think> blocks (Qwen3.5 thinking mode)
    - **bold** markdown formatting
    - "Cuvântul lipsă este: X" and similar prefixes (anywhere in text)
    - Multiple-choice format: "A) word\nB) word" → extracts option A
    - Leading underscores/blanks before the actual answer
    - Numbered lists: "1. word\n2. word"
    """
    text = response.strip()

    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown bold
    text = text.replace("**", "")

    # Strip leading underscores, dots, and whitespace (models echo the blank)
    text = re.sub(r"^[_\s.]+", "", text).strip()

    # Look for "Cuvântul lipsă este: X" anywhere in text (not just start)
    for prefix_pat in [
        r"[Cc]uvântul\s+lipsă\s+este\s*:\s*",
        r"[Tt]he\s+missing\s+word\s+is\s*:\s*",
        r"[Rr]ăspuns\s*:\s*",
    ]:
        m = re.search(prefix_pat, text)
        if m:
            text = text[m.end():].strip()
            break

    # Handle multiple-choice format: "A) word" or "A: word" → take option A's word
    mc_match = re.match(r"^[A-D][):.]\s*(\S+)", text)
    if mc_match:
        text = mc_match.group(1)

    # Handle numbered list: "1. word" or "1) word"
    num_match = re.match(r"^\d+[).]\s*(\S+)", text)
    if num_match:
        text = num_match.group(1)

    # Strip quotes and formatting
    text = text.strip('"\'`*_').strip()

    # Junk words that are never valid emotion predictions
    junk = {"a", "de", "cu", "si", "în", "in", "la", "pe", "un", "o", "se",
            "nu", "ce", "ca", "și", "răspunde", "cuvântul", "cuvantul",
            "text", "sunt", "este"}

    # Take first valid word
    words = re.split(r"[\s,;.!?\n()+]+", text)
    for w in words:
        w = w.strip('"\'`*_()[]{}:')
        w_lower = w.lower()
        if w and len(w) >= 2 and w[0] != '<' and w_lower not in junk:
            return w

    # Fallback: return first non-empty word even if in junk
    for w in words:
        w = w.strip('"\'`*_()[]{}:')
        if w and len(w) >= 2 and w[0] != '<':
            return w
    return words[0].strip('"\'`*_()[]{}:') if words else ""


# ---------------------------------------------------------------------------
# Encoder-decoder (mT5) inference
# ---------------------------------------------------------------------------

def predict_encoder_decoder(
    records: list[dict],
    model_id: str,
    top_k: int = 5,
    batch_size: int = 16,
    device: str = "cuda",
    translated: bool = False,
) -> list[dict]:
    """mT5 inference via beam search."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()

    results = []

    for batch_start in tqdm(range(0, len(records), batch_size), desc="mT5 batches"):
        batch = records[batch_start : batch_start + batch_size]

        # Replace [MASK] with <extra_id_0>
        inputs = [r["masked_text"].replace("[MASK]", "<extra_id_0>") for r in batch]
        enc = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=10,
                num_beams=top_k,
                num_return_sequences=top_k,
                early_stopping=True,
            )

        # Parse: each sample gets top_k sequences
        for i, r in enumerate(batch):
            seqs = outputs[i * top_k : (i + 1) * top_k]
            preds_raw = []
            for seq in seqs:
                decoded = tokenizer.decode(seq, skip_special_tokens=False)
                # Extract word after <extra_id_0>
                match = re.search(r"<extra_id_0>\s*(\S+)", decoded)
                if match:
                    preds_raw.append(match.group(1))
                else:
                    # Fallback: first non-special token
                    clean = tokenizer.decode(seq, skip_special_tokens=True).strip()
                    word = clean.split()[0] if clean.split() else ""
                    if word:
                        preds_raw.append(word)

            # Deduplicate while preserving order
            seen = set()
            preds_dedup = []
            for p in preds_raw:
                normed = normalize_prediction(p)
                if normed and normed not in seen:
                    seen.add(normed)
                    preds_dedup.append(p)

            preds_norm = [normalize_prediction(p) for p in preds_dedup]

            results.append({
                "id": r["id"],
                "seed_word": r["seed_word"],
                "seed_word_normalized": r["seed_word_normalized"],
                "gold_normalized": r["seed_word_normalized"],
                "source": r.get("source", ""),
                "pattern_category": r.get("pattern_category", ""),
                "emotion_category": r.get("emotion_category", []),
                "masked_text": r["masked_text"],
                "predictions_raw": preds_dedup[:20],
                "predictions_normalized": preds_norm[:20],
            })

    return results


# ---------------------------------------------------------------------------
# Decoder-only (vLLM) inference
# ---------------------------------------------------------------------------

def predict_decoder_only_vllm(
    records: list[dict],
    model_id: str,
    top_k: int = 5,
    max_tokens: int = 20,
    translated: bool = False,
    gpu_memory_utilization: float = 0.90,
) -> list[dict]:
    """Decoder-only inference via vLLM with multiple samples for top-k."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,
    )

    # Apply chat template for instruction-tuned models
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    raw_prompts = [build_prompt(r["masked_text"], translated=translated) for r in records]

    if tokenizer.chat_template:
        prompts = []
        for p in tqdm(raw_prompts, desc="Rendering prompts"):
            messages = [{"role": "user", "content": p}]
            # Disable thinking mode for Qwen3.5 to avoid wasting tokens
            template_kwargs = {}
            if "qwen" in model_id.lower():
                template_kwargs["enable_thinking"] = False
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            prompts.append(formatted)
    else:
        prompts = raw_prompts

    # Generate multiple completions per prompt for real top-k diversity.
    # temperature=0.5 keeps top-1 close to greedy while allowing alternatives.
    params = SamplingParams(
        temperature=0.5,
        max_tokens=max_tokens,
        n=top_k,
    )

    outputs = llm.generate(prompts, params)
    results = []

    for r, output in zip(records, outputs):
        # Sort completions by cumulative logprob (most likely first)
        sorted_completions = sorted(
            output.outputs,
            key=lambda o: o.cumulative_logprob or 0,
            reverse=True,
        )

        preds_raw = []
        seen_norm = set()
        for comp in sorted_completions:
            word = parse_single_word(comp.text.strip())
            norm = normalize_prediction(word) if word else ""
            if norm and norm not in seen_norm:
                seen_norm.add(norm)
                preds_raw.append(word)
            if len(preds_raw) >= top_k:
                break

        preds_norm = [normalize_prediction(p) for p in preds_raw]

        results.append({
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
            "generated_texts": [c.text.strip() for c in sorted_completions],
            "generated_text": sorted_completions[0].text.strip() if sorted_completions else "",
        })

    return results


# ---------------------------------------------------------------------------
# Decoder-only (transformers) fallback
# ---------------------------------------------------------------------------

def predict_decoder_only_transformers(
    records: list[dict],
    model_id: str,
    top_k: int = 5,
    batch_size: int = 8,
    device: str = "cuda",
    translated: bool = False,
) -> list[dict]:
    """Decoder-only inference via transformers (fallback when vLLM unavailable)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()

    results = []

    for batch_start in tqdm(range(0, len(records), batch_size), desc="Gen batches"):
        batch = records[batch_start : batch_start + batch_size]
        prompts = [build_prompt(r["masked_text"], translated=translated) for r in batch]

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=20,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, r in enumerate(batch):
            prompt_len = enc["input_ids"][i].shape[0]
            generated_ids = outputs[i][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            word = parse_single_word(generated_text)

            preds_raw = [word] if word else []
            preds_norm = [normalize_prediction(p) for p in preds_raw]

            results.append({
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
                "generated_text": generated_text,
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def is_encoder_decoder(model_id: str) -> bool:
    return "t5" in model_id.lower()


def run_evaluation(
    model_id: str,
    split: str,
    translated: bool = False,
    backend: str = "auto",
    top_k: int = 5,
    batch_size: int = 16,
    compute_similarity: bool = True,
    resume: bool = False,
    device: str | None = None,
    gpu_memory_utilization: float = 0.90,
):
    device = device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    short = get_short_name(model_id)
    lang = "en" if translated else "ro"
    prefix = f"gen_{short}_{split}_{lang}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{prefix}_results.jsonl"
    metrics_path = RESULTS_DIR / f"{prefix}_metrics.json"
    checkpoint_path = RESULTS_DIR / f"{prefix}_checkpoint.jsonl"

    print(f"Model:   {model_id}")
    print(f"Split:   {split} ({'translated EN' if translated else 'Romanian'})")
    print(f"Backend: {backend}")

    records = load_split(split, translated=translated)
    print(f"Loaded {len(records)} samples")

    # Resume
    done_ids = set()
    completed_results = []
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed_results = [json.loads(line) for line in f]
        done_ids = {r["id"] for r in completed_results}
        print(f"Resuming: {len(done_ids)} already done")

    remaining = [r for r in records if r["id"] not in done_ids]
    if not remaining:
        print("All samples already processed.")
        return

    # Run inference
    if is_encoder_decoder(model_id):
        new_results = predict_encoder_decoder(
            remaining, model_id,
            top_k=top_k, batch_size=batch_size, device=device,
            translated=translated,
        )
    elif backend == "vllm" or (backend == "auto" and not is_encoder_decoder(model_id)):
        new_results = predict_decoder_only_vllm(
            remaining, model_id,
            top_k=top_k, translated=translated,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        new_results = predict_decoder_only_transformers(
            remaining, model_id,
            top_k=top_k, batch_size=batch_size, device=device,
            translated=translated,
        )

    # Save checkpoint
    with open(checkpoint_path, "w") as f:
        for r in completed_results + new_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_results = completed_results + new_results

    # Write final results
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute metrics
    print("Computing metrics ...")
    scorer = None
    if compute_similarity:
        print("  Loading similarity scorer ...")
        scorer = ContextualSimilarityScorer(device=device)

    metrics = compute_all_metrics(all_results, scorer=scorer)
    metrics["model"] = model_id
    metrics["split"] = split
    metrics["language"] = lang
    metrics["type"] = "generative"

    metrics["by_source"] = compute_metrics_by_group(all_results, "source")
    metrics["by_pattern"] = compute_metrics_by_group(all_results, "pattern_category")
    metrics["by_gender"] = compute_metrics_by_group(all_results, "gender")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print summary
    print(f"\n=== Results: {short} on {split} ({lang}) ===")
    print(f"  Samples: {metrics['n_samples']}")
    for k in (1, 3, 5):
        acc = metrics.get(f"acc@{k}", 0)
        line = f"  Acc@{k}:  {acc:.4f} ({acc*100:.1f}%)"
        if scorer:
            sim = metrics.get(f"sim@{k}", 0)
            line += f"   Sim@{k}: {sim:.4f}"
        print(line)
    print(f"  MRR:     {metrics['mrr']:.4f}")
    print(f"\nSaved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot generative evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--split", required=True, help="Split name (test, unseen, or custom)")
    parser.add_argument("--translated", action="store_true")
    parser.add_argument("--backend", choices=["vllm", "transformers", "auto"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM GPU memory utilization")
    args = parser.parse_args()

    run_evaluation(
        model_id=args.model,
        split=args.split,
        translated=args.translated,
        backend=args.backend,
        top_k=args.top_k,
        batch_size=args.batch_size,
        compute_similarity=not args.no_similarity,
        resume=args.resume,
        device=args.device,
        gpu_memory_utilization=args.gpu_mem,
    )


if __name__ == "__main__":
    main()
