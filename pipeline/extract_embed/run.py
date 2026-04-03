#!/usr/bin/env python3
"""
Embedding similarity ASI extraction pipeline.

Finds ASI expressions by computing semantic similarity between ALL sentences
in the corpus and known-good ASI anchor sentences (from regex matching).
No trigger-word pre-filter — embeds everything for maximum discovery.

Output: one row per post, with a `hits` list of all qualifying sentences.

Pipeline:
  1. Load all posts from merged_corpus.jsonl
  2. Split every post into sentences (no pre-filter)
  3. Build anchor embeddings from regex-matched ASI sentences
  4. Embed all candidate sentences on Modal GPU
  5. Cosine similarity → group hits by post → output

Usage:
    python -m pipeline.extract_embed.run
    python -m pipeline.extract_embed.run --threshold 0.75 --sample 10
    python -m pipeline.extract_embed.run --dry-run
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.pattern_matcher import PatternMatcher
from pipeline.utils.text_utils import split_into_sentences, normalize_text
from pipeline.utils.corpus_reader import iter_corpus
from pipeline.seed.enriched import build_enriched_seed

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "embedding_asi_candidates.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_posts() -> list[dict]:
    """Load all posts from merged_corpus.jsonl."""
    posts = []
    for record_id, text, source in iter_corpus(sources=["merged_corpus"]):
        posts.append({"id": record_id, "text": text, "source": source})
    print(f"Total posts loaded: {len(posts)}")
    return posts


# ---------------------------------------------------------------------------
# Sentence splitting (no filter — embed everything)
# ---------------------------------------------------------------------------

def extract_all_sentences(posts: list[dict]) -> tuple[list[str], list[tuple[int, int]]]:
    """Split all posts into sentences.

    Returns:
        sentences: list of sentence texts
        sentence_to_post: list of (post_idx, sentence_idx_within_post)
    """
    sentences = []
    sentence_to_post = []

    for post_idx, post in enumerate(tqdm(posts, desc="Splitting sentences")):
        text = post.get("text", "")
        if not text or len(text) < 10:
            continue

        post_sentences = split_into_sentences(text)
        for sent_idx, (start, end, sent_text) in enumerate(post_sentences):
            sentences.append(sent_text)
            sentence_to_post.append((post_idx, sent_idx))

    print(f"Total sentences: {len(sentences):,} from {len(posts):,} posts")
    return sentences, sentence_to_post


# ---------------------------------------------------------------------------
# Anchor building (regex-matched ASI sentences)
# ---------------------------------------------------------------------------

def build_matcher() -> PatternMatcher:
    """Build PatternMatcher with the enriched seed."""
    seed = build_enriched_seed()
    # word_to_affect_categ maps word → string; PatternMatcher expects word → list
    word_to_emotions = {
        word: [categ] if isinstance(categ, str) else categ
        for word, categ in seed["word_to_affect_categ"].items()
    }
    noun_words = seed.get("nouns", {})
    if isinstance(noun_words, dict):
        noun_words = list(noun_words.keys())
    return PatternMatcher(word_to_emotions, noun_words=noun_words)


def build_anchors(posts: list[dict], matcher: PatternMatcher) -> list[dict]:
    """Run regex PatternMatcher on all posts to find known-good ASI sentences.

    Returns list of {sentence, emotions, pattern_name, seed_word, post_id}.
    """
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


def get_regex_matched_sentences(posts: list[dict], matcher: PatternMatcher) -> set[str]:
    """Get MD5 hashes of all regex-matched sentences (for novelty tagging)."""
    matched_hashes = set()
    for post in posts:
        text = post.get("text", "")
        if not text:
            continue
        matches = matcher.find_matches(text, extract_sentences=True)
        for match in matches:
            sent = match.matched_text.strip()
            matched_hashes.add(hashlib.md5(sent.encode()).hexdigest())
    return matched_hashes


# ---------------------------------------------------------------------------
# Modal embedding
# ---------------------------------------------------------------------------

def embed_all_modal(
    anchor_texts: list[str],
    candidate_texts: list[str],
    batch_size: int = 256,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed anchors and candidates in a single Modal session."""
    from pipeline.extract_embed.modal_embeddings import app, Embedder

    all_anchor_embs = []
    all_candidate_embs = []

    with app.run():
        embedder = Embedder()

        print(f"Embedding {len(anchor_texts)} anchors...")
        for i in tqdm(range(0, len(anchor_texts), chunk_size), desc="Anchors (query)"):
            chunk = anchor_texts[i : i + chunk_size]
            embs = embedder.embed_batch.remote(chunk, prefix="query: ", batch_size=batch_size)
            all_anchor_embs.extend(embs)

        print(f"Embedding {len(candidate_texts)} candidate sentences...")
        for i in tqdm(range(0, len(candidate_texts), chunk_size), desc="Candidates (passage)"):
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute max cosine similarity of each candidate to any anchor.

    Returns (max_sims, nearest_indices) arrays of shape (n_candidates,).
    """
    n_candidates = candidate_embeddings.shape[0]
    max_sims = np.zeros(n_candidates, dtype=np.float32)
    nearest_indices = np.zeros(n_candidates, dtype=np.int32)

    chunk_size = 4096
    for i in range(0, n_candidates, chunk_size):
        chunk = candidate_embeddings[i : i + chunk_size]
        sims = chunk @ anchor_embeddings.T
        max_sims[i : i + chunk_size] = sims.max(axis=1)
        nearest_indices[i : i + chunk_size] = sims.argmax(axis=1)

    return max_sims, nearest_indices


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    threshold: float = 0.75,
    batch_size: int = 256,
    sample: int = 0,
    dry_run: bool = False,
    output: str = None,
):
    """Run the full embedding similarity pipeline."""
    output_path = Path(output) if output else OUTPUT_PATH

    # Step 1: Load data
    print("=" * 60)
    print("STEP 1: Loading posts")
    print("=" * 60)
    posts = load_all_posts()

    # Step 2: Split into sentences (no filter)
    print("\n" + "=" * 60)
    print("STEP 2: Splitting into sentences (no pre-filter)")
    print("=" * 60)
    all_sentences, sentence_to_post = extract_all_sentences(posts)

    # Step 3: Build anchors from regex matches
    print("\n" + "=" * 60)
    print("STEP 3: Building anchor sentences from regex matches")
    print("=" * 60)
    matcher = build_matcher()
    anchors = build_anchors(posts, matcher)

    if not anchors:
        print("ERROR: No anchor sentences found. Cannot proceed.")
        return

    # Get regex-matched sentence hashes for novelty tagging
    print("Identifying regex-matched sentences...")
    regex_sentence_hashes = get_regex_matched_sentences(posts, matcher)
    print(f"Regex-matched unique sentences: {len(regex_sentence_hashes)}")

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN — stopping before GPU embedding")
        print("=" * 60)
        print(f"  Posts: {len(posts):,}")
        print(f"  Sentences to embed: {len(all_sentences):,}")
        print(f"  Anchor sentences: {len(anchors):,}")
        print(f"  Regex-matched sentences: {len(regex_sentence_hashes):,}")
        return

    # Step 4: Embed on Modal
    print("\n" + "=" * 60)
    print("STEP 4: Computing embeddings on Modal (GPU)")
    print("=" * 60)

    anchor_texts = [a["sentence"] for a in anchors]
    anchor_embeddings, candidate_embeddings = embed_all_modal(
        anchor_texts, all_sentences, batch_size=batch_size,
    )

    # Step 5: Similarity computation
    print("\n" + "=" * 60)
    print("STEP 5: Computing cosine similarities")
    print("=" * 60)

    max_sims, nearest_indices = compute_similarities(anchor_embeddings, candidate_embeddings)

    # Step 6: Group hits by post and write output
    print("\n" + "=" * 60)
    print(f"STEP 6: Grouping hits (threshold >= {threshold})")
    print("=" * 60)

    # Collect all hits grouped by post
    post_hits: dict[int, list[dict]] = {}

    for sent_idx in range(len(all_sentences)):
        sim = float(max_sims[sent_idx])
        if sim < threshold:
            continue

        post_idx, _ = sentence_to_post[sent_idx]
        anchor_idx = int(nearest_indices[sent_idx])
        anchor = anchors[anchor_idx]
        sentence = all_sentences[sent_idx]
        sent_hash = hashlib.md5(sentence.strip().encode()).hexdigest()
        is_novel = sent_hash not in regex_sentence_hashes

        hit = {
            "sentence": sentence,
            "confidence": round(sim, 4),
            "emotion_category": anchor["emotions"],
            "nearest_anchor": anchor["sentence"],
            "nearest_anchor_pattern": anchor["pattern_name"],
            "is_novel": is_novel,
        }

        if post_idx not in post_hits:
            post_hits[post_idx] = []

        # Deduplicate by sentence within same post
        existing_sents = {h["sentence"] for h in post_hits[post_idx]}
        if sentence not in existing_sents:
            post_hits[post_idx].append(hit)

    # Build output rows (one per post, hits sorted by confidence desc)
    output_rows = []
    total_hits = 0
    novel_hits = 0

    for post_idx in sorted(post_hits.keys()):
        hits = sorted(post_hits[post_idx], key=lambda h: -h["confidence"])
        post = posts[post_idx]
        total_hits += len(hits)
        novel_hits += sum(1 for h in hits if h["is_novel"])

        output_rows.append({
            "id": post["id"],
            "text": post["text"],
            "source": post["source"],
            "extraction_method": "embedding_similarity",
            "hits": hits,
        })

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stats
    print(f"\nResults:")
    print(f"  Posts with hits: {len(output_rows):,}")
    print(f"  Total hits: {total_hits:,}")
    print(f"  Novel hits (not found by regex): {novel_hits:,}")
    print(f"  Regex-known hits: {total_hits - novel_hits:,}")
    print(f"  Output: {output_path}")

    # Similarity distribution (all sentences)
    above_threshold = max_sims[max_sims >= threshold]
    if len(above_threshold) > 0:
        print(f"\nSimilarity distribution (hits only):")
        print(f"  Max:    {above_threshold.max():.4f}")
        print(f"  P95:    {np.percentile(above_threshold, 95):.4f}")
        print(f"  Median: {np.percentile(above_threshold, 50):.4f}")
        print(f"  P5:     {np.percentile(above_threshold, 5):.4f}")
        print(f"  Min:    {above_threshold.min():.4f}")

    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    print(f"\n  Sentences above threshold:")
    for t in thresholds:
        count = int((max_sims >= t).sum())
        marker = " <-- selected" if t == threshold else ""
        print(f"    >= {t:.2f}: {count:,}{marker}")

    # Hits per post distribution
    hits_per_post = [len(h) for h in post_hits.values()]
    if hits_per_post:
        print(f"\n  Hits per post:")
        print(f"    Max: {max(hits_per_post)}")
        print(f"    Median: {np.median(hits_per_post):.0f}")
        print(f"    Mean: {np.mean(hits_per_post):.1f}")
        print(f"    1 hit: {sum(1 for h in hits_per_post if h == 1):,}")
        print(f"    2+ hits: {sum(1 for h in hits_per_post if h >= 2):,}")

    # Sample output
    if sample > 0 and output_rows:
        import random
        print(f"\n{'=' * 60}")
        print(f"SAMPLE OUTPUT ({min(sample, len(output_rows))} random posts)")
        print(f"{'=' * 60}")

        sampled = random.sample(output_rows, min(sample, len(output_rows)))
        for i, row in enumerate(sampled):
            print(f"\n--- Post {i + 1} ({len(row['hits'])} hits) ---")
            print(f"  ID:     {row['id']}")
            print(f"  Source: {row['source']}")
            text_preview = row["text"][:150].replace("\n", " ")
            print(f"  Text:   {text_preview}...")
            for j, hit in enumerate(row["hits"][:3]):
                novel_tag = " [NOVEL]" if hit["is_novel"] else ""
                print(f"  Hit {j + 1}{novel_tag}: ({hit['confidence']:.4f}) {hit['sentence'][:100]}")
                print(f"         Anchor: {hit['nearest_anchor'][:80]}")
                print(f"         Emotions: {hit['emotion_category']}")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding similarity ASI extraction pipeline"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Cosine similarity threshold (default: 0.75)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Embedding batch size (default: 256)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Print N random sample posts at the end (default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run steps 1-3 only (no GPU), print stats",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output path",
    )
    args = parser.parse_args()

    run_pipeline(
        threshold=args.threshold,
        batch_size=args.batch_size,
        sample=args.sample,
        dry_run=args.dry_run,
        output=args.output,
    )


if __name__ == "__main__":
    main()
