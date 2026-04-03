#!/usr/bin/env python3
"""
Embedding Similarity ASI extraction pipeline.

Finds ASI expressions missed by regex by using semantic similarity
to known-good ASI sentences (anchors). If ANY sentence in a post
is similar enough to an anchor, the WHOLE post is added to the output.

Pipeline:
  1. Load RedditRoAP + PoPreRo datasets
  2. Sentence split + loose trigger-word pre-filter
  3. Build anchor embeddings from regex-matched ASI sentences (Modal GPU)
  4. Embed candidate trigger sentences (Modal GPU)
  5. Cosine similarity ranking → output posts above threshold

Usage:
    python -m experiments.embedding_similarity.embedding_candidates
    python -m experiments.embedding_similarity.embedding_candidates --threshold 0.75 --sample 20
"""

import argparse
import json
import re
import sys
import hashlib
from pathlib import Path
from typing import Generator

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ro_asi.pattern_matcher import (
    PatternMatcher,
    PatternMatch,
    remove_diacritics,
    normalize_text,
    DIACRITIC_MAP,
)
from scripts.ro_asi.curated_affective_states import build_curated_seed
from scripts.ro_asi.merge_datasets import process_reddit_roap, process_poprero

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMALL_DATASETS_PATH = PROJECT_ROOT / "small_datasets"
OUTPUT_PATH = PROJECT_ROOT / "data" / "embedding_asi_candidates.jsonl"

# "I feel" trigger patterns only — verb forms expressing feeling/sensing/state.
# Excludes generic verbs (sunt, este, era, a fost) that produce noise.
# Covers: a (se) simți, a-i fi (mi-e), îmi este/era/vine
TRIGGER_WORDS = [
    # simți - all persons, tenses, reflexive forms
    "mă simt", "te simți", "se simte", "ne simțim", "vă simțiți", "se simt",
    "mă simțeam", "te simțeai", "se simțea", "ne simțeam", "se simțeau",
    "m-am simțit", "te-ai simțit", "s-a simțit", "ne-am simțit", "s-au simțit",
    "mă voi simți", "o să mă simt",
    # bare simți (non-reflexive: "I feel/sense")
    "simt", "simți", "simte", "simțim", "simțiți",
    "simțeam", "simțeai", "simțea",
    "am simțit", "ai simțit", "a simțit", "au simțit",
    # mi-e / îmi este (dative "I feel cold/scared/etc.")
    "mi-e", "mi-i", "ți-e", "ți-i", "i-e", "i-i",
    "îmi e", "îți e", "îi e",
    "îmi este", "îți este", "îi este",
    "îmi era", "îți era", "îi era",
    "îmi vine", "îți vine", "îi vine",
    "îmi venea", "îți venea",
    "ne-e", "ne este", "ne era",
    # feel like / feel as if
    "mă simt ca și cum", "mă simt de parcă", "mă simt ca și când",
    "simt că", "simt ca",
]

# Pre-compute normalized trigger words for matching
TRIGGER_WORDS_NORMALIZED = [normalize_text(w) for w in TRIGGER_WORDS]

# Sentence splitting regex: split on sentence-ending punctuation or newlines
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


# ---------------------------------------------------------------------------
# Sentence splitting & pre-filtering
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation and newlines."""
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def sentence_has_trigger(sentence: str) -> bool:
    """Check if a sentence contains any trigger word (diacritic-normalized)."""
    normalized = normalize_text(sentence)
    for trigger in TRIGGER_WORDS_NORMALIZED:
        # Word boundary check: trigger should appear as whole words
        # Use simple substring match with boundary awareness
        idx = normalized.find(trigger)
        while idx != -1:
            # Check word boundaries
            before_ok = (idx == 0 or not normalized[idx - 1].isalpha())
            after_pos = idx + len(trigger)
            after_ok = (after_pos >= len(normalized) or not normalized[after_pos].isalpha())
            if before_ok and after_ok:
                return True
            idx = normalized.find(trigger, idx + 1)
    return False


def get_trigger_word(sentence: str) -> str | None:
    """Return the first trigger word found in the sentence."""
    normalized = normalize_text(sentence)
    for i, trigger in enumerate(TRIGGER_WORDS_NORMALIZED):
        idx = normalized.find(trigger)
        while idx != -1:
            before_ok = (idx == 0 or not normalized[idx - 1].isalpha())
            after_pos = idx + len(trigger)
            after_ok = (after_pos >= len(normalized) or not normalized[after_pos].isalpha())
            if before_ok and after_ok:
                return TRIGGER_WORDS[i]
            idx = normalized.find(trigger, idx + 1)
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_posts() -> list[dict]:
    """Load all posts from RedditRoAP and PoPreRo."""
    posts = []

    print("Loading RedditRoAP...")
    for record in tqdm(process_reddit_roap(SMALL_DATASETS_PATH), desc="RedditRoAP"):
        posts.append(record)

    print("Loading PoPreRo...")
    for record in tqdm(process_poprero(SMALL_DATASETS_PATH), desc="PoPreRo"):
        posts.append(record)

    print(f"Total posts loaded: {len(posts)}")
    return posts


def prefilter_posts(posts: list[dict]) -> list[dict]:
    """Pre-filter posts and extract trigger sentences.

    Returns posts that have at least one trigger sentence, annotated with:
      - trigger_sentences: list of sentences containing trigger words
      - trigger_words: list of which trigger word was found per sentence
    """
    filtered = []
    total_trigger_sentences = 0

    for post in tqdm(posts, desc="Pre-filtering"):
        text = post.get("text", "")
        if not text or len(text) < 10:
            continue

        sentences = split_sentences(text)
        trigger_sentences = []
        trigger_words = []

        for sent in sentences:
            if sentence_has_trigger(sent):
                trigger_sentences.append(sent)
                trigger_words.append(get_trigger_word(sent))

        if trigger_sentences:
            post_copy = dict(post)
            post_copy["trigger_sentences"] = trigger_sentences
            post_copy["trigger_words"] = trigger_words
            filtered.append(post_copy)
            total_trigger_sentences += len(trigger_sentences)

    print(f"Posts with trigger sentences: {len(filtered)} / {len(posts)}")
    print(f"Total trigger sentences: {total_trigger_sentences}")
    return filtered


# ---------------------------------------------------------------------------
# Anchor building (regex-matched ASI sentences)
# ---------------------------------------------------------------------------

def build_anchors(posts: list[dict]) -> list[dict]:
    """Run regex PatternMatcher on all posts to find known-good ASI sentences.

    Returns list of {sentence, emotions, pattern_name, seed_word, post_id}.
    """
    seed = build_curated_seed()
    matcher = PatternMatcher(
        word_to_emotions=seed["word_to_emotions"],
        noun_words=seed["nouns"],
    )

    anchors = []
    seen_sentences = set()

    for post in tqdm(posts, desc="Building anchors"):
        text = post.get("text", "")
        if not text:
            continue

        matches = matcher.find_matches(text, extract_sentences=True)
        for match in matches:
            sent = match.matched_text.strip()
            sent_hash = hashlib.md5(sent.encode()).hexdigest()
            if sent_hash not in seen_sentences:
                seen_sentences.add(sent_hash)
                anchors.append({
                    "sentence": sent,
                    "emotions": match.emotions,
                    "pattern_name": match.pattern_name,
                    "seed_word": match.seed_word,
                    "post_id": post["id"],
                })

    print(f"Unique anchor sentences: {len(anchors)}")
    return anchors


# ---------------------------------------------------------------------------
# Regex-matched post IDs (to identify novel finds)
# ---------------------------------------------------------------------------

def get_regex_matched_post_ids(posts: list[dict]) -> set[str]:
    """Get IDs of posts that have at least one regex match."""
    seed = build_curated_seed()
    matcher = PatternMatcher(
        word_to_emotions=seed["word_to_emotions"],
        noun_words=seed["nouns"],
    )
    matched_ids = set()
    for post in posts:
        text = post.get("text", "")
        if text and matcher.has_affective_pattern(text):
            matched_ids.add(post["id"])
    return matched_ids


# ---------------------------------------------------------------------------
# Modal embedding calls
# ---------------------------------------------------------------------------

def embed_all_modal(
    anchor_texts: list[str],
    candidate_texts: list[str],
    batch_size: int = 256,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed anchors and candidates in a single Modal session.

    Returns (anchor_embeddings, candidate_embeddings) as numpy arrays.
    """
    from experiments.embedding_similarity.modal_embeddings import app, Embedder

    all_anchor_embs = []
    all_candidate_embs = []

    with app.run():
        embedder = Embedder()

        print(f"Embedding {len(anchor_texts)} anchors...")
        for i in tqdm(range(0, len(anchor_texts), chunk_size), desc="Embedding (query)"):
            chunk = anchor_texts[i : i + chunk_size]
            embs = embedder.embed_batch.remote(chunk, prefix="query: ", batch_size=batch_size)
            all_anchor_embs.extend(embs)

        print(f"Embedding {len(candidate_texts)} candidate sentences...")
        for i in tqdm(range(0, len(candidate_texts), chunk_size), desc="Embedding (passage)"):
            chunk = candidate_texts[i : i + chunk_size]
            embs = embedder.embed_batch.remote(chunk, prefix="passage: ", batch_size=batch_size)
            all_candidate_embs.extend(embs)

    return (
        np.array(all_anchor_embs, dtype=np.float32),
        np.array(all_candidate_embs, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def compute_similarities(
    anchor_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute max cosine similarity of each candidate to any anchor.

    Returns array of shape (n_candidates,) with max similarity scores.
    Since embeddings are normalized, cosine sim = dot product.
    """
    # candidate_embeddings: (N_cand, 768)
    # anchor_embeddings: (N_anchor, 768)
    # Result: (N_cand, N_anchor)
    # We want max over anchors for each candidate

    # Do in chunks to avoid memory issues
    n_candidates = candidate_embeddings.shape[0]
    max_sims = np.zeros(n_candidates, dtype=np.float32)
    nearest_indices = np.zeros(n_candidates, dtype=np.int32)

    chunk_size = 4096
    for i in range(0, n_candidates, chunk_size):
        chunk = candidate_embeddings[i : i + chunk_size]
        sims = chunk @ anchor_embeddings.T  # (chunk, n_anchors)
        max_sims[i : i + chunk_size] = sims.max(axis=1)
        nearest_indices[i : i + chunk_size] = sims.argmax(axis=1)

    return max_sims, nearest_indices


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    threshold: float = 0.75,
    model_id: str = "intfloat/multilingual-e5-base",
    gpu: str = "T4",
    batch_size: int = 256,
    sample: int = 0,
):
    """Run the full embedding similarity pipeline."""

    # Step 1: Load data
    print("=" * 60)
    print("STEP 1: Loading datasets")
    print("=" * 60)
    posts = load_all_posts()

    # Step 2: Pre-filter
    print("\n" + "=" * 60)
    print("STEP 2: Sentence splitting + trigger word pre-filter")
    print("=" * 60)
    filtered_posts = prefilter_posts(posts)

    # Step 3: Build anchors from regex matches
    print("\n" + "=" * 60)
    print("STEP 3: Building anchor sentences from regex matches")
    print("=" * 60)
    anchors = build_anchors(posts)

    if not anchors:
        print("ERROR: No anchor sentences found. Cannot proceed.")
        return

    # Get regex-matched post IDs for novelty tracking
    print("Identifying regex-matched posts...")
    regex_post_ids = get_regex_matched_post_ids(posts)
    print(f"Posts with regex matches: {len(regex_post_ids)}")

    # Collect all trigger sentences across filtered posts
    all_trigger_sentences = []
    trigger_to_post_idx = []  # maps trigger sentence index → (post_idx, sent_idx)

    for post_idx, post in enumerate(filtered_posts):
        for sent_idx, sent in enumerate(post["trigger_sentences"]):
            all_trigger_sentences.append(sent)
            trigger_to_post_idx.append((post_idx, sent_idx))

    print(f"\nTotal trigger sentences to embed: {len(all_trigger_sentences)}")
    print(f"Total anchor sentences to embed: {len(anchors)}")

    # Step 4: Embed on Modal
    print("\n" + "=" * 60)
    print("STEP 4: Computing embeddings on Modal (GPU)")
    print("=" * 60)

    anchor_texts = [a["sentence"] for a in anchors]
    anchor_embeddings, candidate_embeddings = embed_all_modal(
        anchor_texts, all_trigger_sentences, batch_size=batch_size,
    )

    # Step 5: Similarity computation
    print("\n" + "=" * 60)
    print("STEP 5: Computing cosine similarities")
    print("=" * 60)

    max_sims, nearest_indices = compute_similarities(anchor_embeddings, candidate_embeddings)

    # Aggregate to post level: best trigger sentence per post
    post_best_score = {}   # post_idx → best similarity
    post_best_sent = {}    # post_idx → (sentence, trigger_word, sim, anchor_idx)
    post_all_scores = {}   # post_idx → list of (sentence, score)

    for trig_idx, (post_idx, sent_idx) in enumerate(trigger_to_post_idx):
        sim = float(max_sims[trig_idx])
        anchor_idx = int(nearest_indices[trig_idx])
        sent = all_trigger_sentences[trig_idx]
        trigger_word = filtered_posts[post_idx]["trigger_words"][sent_idx]

        if post_idx not in post_all_scores:
            post_all_scores[post_idx] = []
        post_all_scores[post_idx].append({"sentence": sent, "score": round(sim, 4)})

        if post_idx not in post_best_score or sim > post_best_score[post_idx]:
            post_best_score[post_idx] = sim
            post_best_sent[post_idx] = (sent, trigger_word, sim, anchor_idx)

    # Step 6: Threshold + output
    print("\n" + "=" * 60)
    print(f"STEP 6: Thresholding (>= {threshold}) + output")
    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates = []
    novel_count = 0

    for post_idx, best_sim in sorted(post_best_score.items(), key=lambda x: -x[1]):
        if best_sim < threshold:
            continue

        post = filtered_posts[post_idx]
        sent, trigger_word, sim, anchor_idx = post_best_sent[post_idx]
        anchor = anchors[anchor_idx]
        is_novel = post["id"] not in regex_post_ids

        candidate = {
            "id": post["id"],
            "text": post["text"],
            "matched_sentence": sent,
            "extraction_strategy": "embedding_similarity",
            "confidence": round(sim, 4),
            "seed_word": None,
            "emotion_category": anchor["emotions"],
            "source": post["source"],
            "metadata": {
                "cosine_similarity": round(sim, 4),
                "nearest_anchor": anchor["sentence"],
                "nearest_anchor_emotion": anchor["emotions"],
                "nearest_anchor_pattern": anchor["pattern_name"],
                "pre_filter_trigger": trigger_word,
                "is_novel": is_novel,
                "all_trigger_sentences_scores": sorted(
                    post_all_scores.get(post_idx, []),
                    key=lambda x: -x["score"],
                ),
            },
        }
        candidates.append(candidate)
        if is_novel:
            novel_count += 1

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for cand in candidates:
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")

    # Stats
    print(f"\nResults:")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Novel (not found by regex): {novel_count}")
    print(f"  Already regex-matched: {len(candidates) - novel_count}")
    print(f"  Output: {OUTPUT_PATH}")

    # Similarity distribution
    all_best_sims = sorted(post_best_score.values(), reverse=True)
    if all_best_sims:
        print(f"\nSimilarity distribution (all pre-filtered posts):")
        print(f"  Max:    {all_best_sims[0]:.4f}")
        print(f"  P95:    {np.percentile(all_best_sims, 95):.4f}")
        print(f"  P90:    {np.percentile(all_best_sims, 90):.4f}")
        print(f"  Median: {np.percentile(all_best_sims, 50):.4f}")
        print(f"  P10:    {np.percentile(all_best_sims, 10):.4f}")
        print(f"  Min:    {all_best_sims[-1]:.4f}")

    above_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    print(f"\n  Posts above threshold:")
    for t in above_thresholds:
        count = sum(1 for s in all_best_sims if s >= t)
        marker = " ← selected" if t == threshold else ""
        print(f"    >= {t:.2f}: {count}{marker}")

    # Sample output
    if sample > 0 and candidates:
        import random
        print(f"\n{'=' * 60}")
        print(f"SAMPLE OUTPUT ({min(sample, len(candidates))} random candidates)")
        print(f"{'=' * 60}")

        sampled = random.sample(candidates, min(sample, len(candidates)))
        for i, cand in enumerate(sampled):
            novel_tag = " [NOVEL]" if cand["metadata"]["is_novel"] else " [REGEX-KNOWN]"
            print(f"\n--- Sample {i+1}{novel_tag} ---")
            print(f"  ID:        {cand['id']}")
            print(f"  Source:    {cand['source']}")
            print(f"  Sim:       {cand['confidence']:.4f}")
            print(f"  Matched:   {cand['matched_sentence'][:120]}")
            print(f"  Anchor:    {cand['metadata']['nearest_anchor'][:120]}")
            print(f"  Emotions:  {cand['emotion_category']}")
            print(f"  Trigger:   {cand['metadata']['pre_filter_trigger']}")
            text_preview = cand["text"][:200].replace("\n", " ")
            print(f"  Text:      {text_preview}...")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding similarity ASI extraction pipeline"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Cosine similarity threshold (default: 0.75)",
    )
    parser.add_argument(
        "--model", type=str, default="intfloat/multilingual-e5-base",
        help="Embedding model ID (default: intfloat/multilingual-e5-base)",
    )
    parser.add_argument(
        "--gpu", type=str, default="T4",
        help="Modal GPU type (default: T4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Embedding batch size (default: 256)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Print N random samples at the end (default: 0)",
    )
    args = parser.parse_args()

    run_pipeline(
        threshold=args.threshold,
        model_id=args.model,
        gpu=args.gpu,
        batch_size=args.batch_size,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
