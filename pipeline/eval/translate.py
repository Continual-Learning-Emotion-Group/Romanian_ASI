"""Translate Romanian benchmark samples to English for cross-lingual evaluation.

Uses NLLB-200-distilled-1.3B for RO -> EN translation.
Two-pass approach to handle [MASK] token properly.

Usage:
    python -m pipeline.eval.translate --split test
    python -m pipeline.eval.translate --split unseen --batch-size 64
"""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS_DIR = DATA_DIR / "splits"

TRANSLATION_MODEL = "facebook/nllb-200-distilled-1.3B"
SRC_LANG = "ron_Latn"  # Romanian in NLLB format
TGT_LANG = "eng_Latn"  # English in NLLB format


def load_split(split: str) -> list[dict]:
    path = SPLITS_DIR / f"{split}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def translate_batch(
    texts: list[str],
    model,
    tokenizer,
    max_length: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Translate a batch of Romanian texts to English."""
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **enc,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_new_tokens=max_length,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_seed_word(
    seed_word: str,
    model,
    tokenizer,
    device: str = "cuda",
) -> str:
    """Translate the seed word in a minimal emotion context.

    Uses 'Mă simt {word}' -> 'I feel {X}' to get the affective translation.
    """
    context = f"Mă simt {seed_word}."
    translated = translate_batch([context], model, tokenizer, device=device)[0]

    # Extract the word after "I feel" (or similar)
    patterns = [
        r"[Ii] feel\s+(\w+)",
        r"[Ii] am feeling\s+(\w+)",
        r"[Ii]'m feeling\s+(\w+)",
        r"[Ii] am\s+(\w+)",
    ]
    for pat in patterns:
        m = re.search(pat, translated)
        if m:
            return m.group(1).lower()

    # Fallback: translate the word alone
    translated_alone = translate_batch([seed_word], model, tokenizer, device=device)[0]
    return translated_alone.strip().split()[0].lower() if translated_alone.strip() else seed_word


def locate_word_in_text(text: str, word: str) -> tuple[int, int] | None:
    """Find word in text (case-insensitive, whole-word)."""
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return m.start(), m.end()
    return None


def translate_records(
    records: list[dict],
    model,
    tokenizer,
    batch_size: int = 32,
    device: str = "cuda",
) -> list[dict]:
    """Translate records with two-pass [MASK] handling."""

    # Step 1: Batch-translate all full texts (with emotion word present)
    print("Pass 1: Translating full texts ...")
    all_texts = [r["text"] for r in records]
    all_text_en = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Translating texts"):
        batch = all_texts[i : i + batch_size]
        all_text_en.extend(translate_batch(batch, model, tokenizer, device=device))

    # Step 2: Translate seed words (cache unique words)
    print("Pass 2: Translating seed words ...")
    unique_seeds = list(set(r["seed_word"] for r in records))
    seed_translations = {}
    for seed in tqdm(unique_seeds, desc="Translating seeds"):
        seed_translations[seed] = translate_seed_word(seed, model, tokenizer, device=device)

    # Step 3: Locate translated seed word in translated text and mask it
    print("Pass 3: Locating and masking ...")
    results = [None] * len(records)
    n_verified = 0
    fallback_indices = []
    fallback_texts = []

    for idx, (r, text_en) in enumerate(zip(records, all_text_en)):
        seed_word_en = seed_translations[r["seed_word"]]
        span = locate_word_in_text(text_en, seed_word_en)

        if span is not None:
            start, end = span
            masked_text_en = text_en[:start] + "[MASK]" + text_en[end:]
            translated_record = dict(r)
            translated_record["text_en"] = text_en
            translated_record["masked_text_en"] = masked_text_en
            translated_record["seed_word_en"] = seed_word_en
            translated_record["mask_verified"] = True
            results[idx] = translated_record
            n_verified += 1
        else:
            fallback_indices.append(idx)
            placeholder = "XXXMASKXXX"
            fallback_texts.append(r["masked_text"].replace("[MASK]", placeholder))

    # Batch-translate all fallbacks at once
    n_fallback = len(fallback_indices)
    print(f"  Verified: {n_verified} ({n_verified / len(records) * 100:.1f}%)")
    print(f"  Fallback: {n_fallback} ({n_fallback / len(records) * 100:.1f}%) — translating ...")

    all_fallback_en = []
    for i in tqdm(range(0, len(fallback_texts), batch_size), desc="Fallback batches"):
        batch = fallback_texts[i : i + batch_size]
        all_fallback_en.extend(translate_batch(batch, model, tokenizer, device=device))

    placeholder = "XXXMASKXXX"
    for fi, fb_idx in enumerate(fallback_indices):
        r = records[fb_idx]
        text_en = all_text_en[fb_idx]
        translated_masked = all_fallback_en[fi]

        # Restore [MASK] from placeholder
        masked_text_en = translated_masked
        for variant in [placeholder, "xxx mask xxx", "XXX MASK XXX"]:
            if variant.lower() in masked_text_en.lower():
                loc = masked_text_en.lower().index(variant.lower())
                masked_text_en = (
                    masked_text_en[:loc] + "[MASK]"
                    + masked_text_en[loc + len(variant):]
                )
                break
        else:
            masked_text_en = "[MASK] " + masked_text_en

        translated_record = dict(r)
        translated_record["text_en"] = text_en
        translated_record["masked_text_en"] = masked_text_en
        translated_record["seed_word_en"] = seed_translations[r["seed_word"]]
        translated_record["mask_verified"] = False
        results[fb_idx] = translated_record

    print(f"  Total: {len(records)}, verified: {n_verified}, fallback: {n_fallback}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Translate RO benchmark to EN")
    parser.add_argument("--split", required=True, help="Split name (test, unseen, or custom)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=TRANSLATION_MODEL)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    records = load_split(args.split)
    print(f"Loaded {len(records)} records from {args.split}")

    print(f"Loading translation model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    results = translate_records(
        records, model, tokenizer,
        batch_size=args.batch_size, device=device,
    )

    output_path = SPLITS_DIR / f"{args.split}_translated_en.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    n_verified = sum(1 for r in results if r["mask_verified"])
    unique_en_seeds = len(set(r["seed_word_en"] for r in results))

    stats = {
        "split": args.split,
        "total": len(results),
        "mask_verified": n_verified,
        "mask_fallback": len(results) - n_verified,
        "verification_rate": round(n_verified / len(results), 4),
        "unique_seed_words_en": unique_en_seeds,
        "translation_model": args.model,
    }
    stats_path = SPLITS_DIR / f"{args.split}_translated_en.stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Stats:  {stats_path}")


if __name__ == "__main__":
    main()
