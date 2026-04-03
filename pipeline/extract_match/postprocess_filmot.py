#!/usr/bin/env python3
"""
Post-process filmot candidates: trim to first-person speaker context.

Reads pattern_candidates_filmot.jsonl, uses Stanza to identify verb person
in each sentence, and trims surrounding context to only keep first-person
sentences adjacent to the match.

Steps:
  1. Split subtitle text into sentences (capitalization + >> + #tags#)
  2. Run Stanza dependency parsing on each sentence
  3. Find the sentence containing the pattern match (anchor)
  4. Expand outward, keeping Person=1 / unknown sentences
  5. Stop at Person=2 or Person=3 root verbs
  6. Join kept sentences with ". "

Writes to a separate file (does NOT overwrite the original).

Requires:
    pip install stanza
    (downloads ro model on first run)

Usage:
    python -m pipeline.extract_match.postprocess_filmot
    python -m pipeline.extract_match.postprocess_filmot --max-records 500  # quick test
    python -m pipeline.extract_match.postprocess_filmot --input path/to/input.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "pattern_candidates_filmot.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "pattern_candidates_filmot_pp.jsonl"


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """
    Split subtitle text into sentences using available boundary markers.

    Boundaries:
      - >> (YouTube speaker change marker)
      - #...# (music/sound tags like #Muzică#)
      - Capital letter after lowercase/comma/? (auto-caption sentence start)
    """
    # First, split on >> and #tags#
    parts = re.split(r'>>|#[^#]+#', text)

    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Split on capital letter after lowercase, comma, or question mark
        segs = re.split(r'(?<=[a-zăâîșț,?]) (?=[A-ZĂÂÎȘȚ])', part)
        for seg in segs:
            seg = seg.strip()
            if seg:
                sentences.append(seg)

    return sentences


# ---------------------------------------------------------------------------
# Person detection via Stanza
# ---------------------------------------------------------------------------

def get_root_person(stanza_sent) -> Optional[int]:
    """
    Get the grammatical person of the root verb in a Stanza sentence.

    Returns 1, 2, 3, or None if no finite verb / no person feature.
    """
    # Try root first
    for word in stanza_sent.words:
        if word.deprel == 'root' and word.upos in ('VERB', 'AUX'):
            return _extract_person(word)

    # If root isn't a verb, look for the main finite verb
    # (cop, aux attached to root, or first verb with person)
    for word in stanza_sent.words:
        if word.upos in ('VERB', 'AUX') and word.feats:
            person = _extract_person(word)
            if person is not None:
                return person

    return None


def _extract_person(word) -> Optional[int]:
    """Extract Person feature from a Stanza word."""
    if not word.feats:
        return None
    for feat in word.feats.split('|'):
        if feat.startswith('Person='):
            try:
                return int(feat.split('=')[1])
            except ValueError:
                return None
    return None


def detect_sentence_persons(sentences: List[str], nlp) -> List[Optional[int]]:
    """
    Run Stanza on sentences in a batch and return root verb person for each.

    Batches all non-trivial sentences into a single Stanza call using
    pre-tokenized input with sentence boundaries, avoiding per-sentence overhead.

    Returns list of person values (1, 2, 3, or None) parallel to sentences.
    """
    persons: List[Optional[int]] = [None] * len(sentences)

    # Separate trivial (too short) from processable sentences
    batch_indices = []
    batch_texts = []
    for i, sent_text in enumerate(sentences):
        if len(sent_text.split()) < 2:
            continue
        batch_indices.append(i)
        batch_texts.append(sent_text)

    if not batch_texts:
        return persons

    # Join with newlines so Stanza treats each as a separate sentence
    joined = '\n\n'.join(batch_texts)
    try:
        doc = nlp(joined)
        for j, stanza_sent in enumerate(doc.sentences):
            if j < len(batch_indices):
                persons[batch_indices[j]] = get_root_person(stanza_sent)
    except Exception:
        # Fallback: process one by one
        for j, sent_text in enumerate(batch_texts):
            try:
                doc = nlp(sent_text)
                if doc.sentences:
                    persons[batch_indices[j]] = get_root_person(doc.sentences[0])
            except Exception:
                pass

    return persons


# ---------------------------------------------------------------------------
# Context trimming
# ---------------------------------------------------------------------------

def find_match_sentence(sentences: List[str], seed_word: str, pattern_name: str) -> int:
    """Find which sentence contains the pattern match."""
    seed_lower = seed_word.lower()

    # Build a simple trigger check based on pattern name
    trigger_words = {
        'ma_simt_present': ['simt'],
        'ma_simteam_imperfect': ['simteam', 'simțeam'],
        'mam_simtit_perfect': ['simtit', 'simțit'],
        'ma_voi_simti_future': ['voi'],
        'o_sa_ma_simt_future': ['simt'],
        'mas_simti_conditional': ['simti', 'simți'],
        'sa_ma_simt_subjunctive': ['simt'],
        'simt_ca': ['simt'],
        'simt_noun': ['simt'],
        'simteam_noun': ['simteam', 'simțeam'],
        'sunt_adj_present': ['sunt'],
        'eram_adj_imperfect': ['eram'],
        'am_fost_adj_perfect': ['fost'],
        'o_sa_fiu_future': ['fiu'],
        'ma_fac_reflexive': ['fac'],
        'imi_este_present': ['este'],
        'imi_era_imperfect': ['era'],
        'mie_short': ['mi-e', 'mie'],
        'am_noun_present': ['am'],
        'aveam_noun_imperfect': ['aveam'],
    }

    triggers = trigger_words.get(pattern_name, [])

    # Best match: sentence contains both trigger and seed word
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        has_seed = seed_lower in sent_lower
        has_trigger = any(t in sent_lower for t in triggers)
        if has_seed and has_trigger:
            return i

    # Fallback: sentence contains seed word
    for i, sent in enumerate(sentences):
        if seed_lower in sent.lower():
            return i

    # Last resort: return middle sentence
    return len(sentences) // 2


def trim_to_first_person(
    sentences: List[str],
    persons: List[Optional[int]],
    anchor_idx: int,
) -> List[str]:
    """
    Keep the anchor sentence and expand outward while sentences are first-person
    or unknown. Stop at second/third person boundaries.
    """
    n = len(sentences)
    start = anchor_idx
    end = anchor_idx

    # Expand backwards
    for i in range(anchor_idx - 1, -1, -1):
        if persons[i] in (2, 3):
            break
        start = i

    # Expand forwards
    for i in range(anchor_idx + 1, n):
        if persons[i] in (2, 3):
            break
        end = i

    return sentences[start:end + 1]


# ---------------------------------------------------------------------------
# Main post-processing
# ---------------------------------------------------------------------------

def postprocess(
    input_path: Path = None,
    output_path: Path = None,
    max_records: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Post-process filmot candidates: trim to first-person context.
    """
    if input_path is None:
        input_path = DEFAULT_INPUT
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    # Load input
    candidates = []
    bad_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                candidates.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                bad_lines += 1

    if max_records > 0:
        candidates = candidates[:max_records]

    if verbose:
        print(f"Loaded {len(candidates)} candidates from {input_path}")
        if bad_lines:
            print(f"  ({bad_lines} bad lines skipped)")

    # Init Stanza
    import stanza
    if verbose:
        print("Loading Stanza Romanian model...")
    nlp = stanza.Pipeline(
        'ro',
        processors='tokenize,pos,lemma,depparse',
        verbose=False,
    )

    BATCH_SIZE = 64  # candidates per Stanza batch

    stats = {
        "total": len(candidates),
        "trimmed": 0,
        "unchanged": 0,
        "avg_sentences_before": 0,
        "avg_sentences_after": 0,
        "avg_chars_before": 0,
        "avg_chars_after": 0,
    }

    total_sents_before = 0
    total_sents_after = 0
    total_chars_before = 0
    total_chars_after = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _process_batch(batch, nlp):
        """
        Process a batch of candidates: split sentences, batch-run Stanza,
        trim to first person.

        Returns list of processed candidate dicts.
        """
        # Phase 1: split all candidates into sentences
        all_split = []  # (cand_idx, sentences_list)
        for ci, cand in enumerate(batch):
            sents = split_sentences(cand["text"])
            all_split.append(sents)

        # Phase 2: collect all non-trivial sentences into one big batch
        # Track which candidate and sentence index each belongs to
        flat_texts = []
        flat_map = []  # (cand_idx, sent_idx)
        for ci, sents in enumerate(all_split):
            for si, sent_text in enumerate(sents):
                if len(sent_text.split()) >= 2:
                    flat_texts.append(sent_text)
                    flat_map.append((ci, si))

        # Phase 3: run Stanza on all sentences at once
        all_persons = [[None] * len(sents) for sents in all_split]

        if flat_texts:
            joined = '\n\n'.join(flat_texts)
            try:
                doc = nlp(joined)
                for j, stanza_sent in enumerate(doc.sentences):
                    if j < len(flat_map):
                        ci, si = flat_map[j]
                        all_persons[ci][si] = get_root_person(stanza_sent)
            except Exception:
                # Fallback: skip person detection for this batch
                pass

        # Phase 4: trim each candidate
        results = []
        for ci, cand in enumerate(batch):
            original_text = cand["text"]
            sentences = all_split[ci]
            persons = all_persons[ci]

            if len(sentences) <= 1:
                processed_text = original_text.strip()
                if not processed_text.endswith(('.', '?', '!')):
                    processed_text += '.'
                cand["text_pp"] = processed_text
                cand["text_original"] = original_text
                cand["pp_sentences_before"] = 1
                cand["pp_sentences_after"] = 1
                results.append((cand, False))
                continue

            anchor = find_match_sentence(
                sentences, cand["seed_word"], cand["pattern_used"]
            )
            kept = trim_to_first_person(sentences, persons, anchor)

            processed_text = '. '.join(s.rstrip('.?!, ') for s in kept)
            if not processed_text.endswith(('.', '?', '!')):
                processed_text += '.'

            cand["text_pp"] = processed_text
            cand["text_original"] = original_text
            cand["pp_sentences_before"] = len(sentences)
            cand["pp_sentences_after"] = len(kept)
            results.append((cand, len(kept) < len(sentences)))

        return results

    with open(output_path, "w", encoding="utf-8") as out_f:
        pbar = tqdm(total=len(candidates), desc="Post-processing", unit="rec") if verbose else None

        for batch_start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[batch_start:batch_start + BATCH_SIZE]
            results = _process_batch(batch, nlp)

            for cand, was_trimmed in results:
                if was_trimmed:
                    stats["trimmed"] += 1
                else:
                    stats["unchanged"] += 1

                total_sents_before += cand["pp_sentences_before"]
                total_sents_after += cand["pp_sentences_after"]
                total_chars_before += len(cand["text_original"])
                total_chars_after += len(cand["text_pp"])

                out_f.write(json.dumps(cand, ensure_ascii=False) + "\n")

            if pbar:
                pbar.update(len(batch))

        if pbar:
            pbar.close()

    # Compute averages
    n = len(candidates)
    stats["avg_sentences_before"] = round(total_sents_before / max(n, 1), 1)
    stats["avg_sentences_after"] = round(total_sents_after / max(n, 1), 1)
    stats["avg_chars_before"] = round(total_chars_before / max(n, 1), 0)
    stats["avg_chars_after"] = round(total_chars_after / max(n, 1), 0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Post-processing complete")
        print(f"{'='*60}")
        print(f"Total: {stats['total']}")
        print(f"Trimmed: {stats['trimmed']} ({stats['trimmed']/max(n,1)*100:.1f}%)")
        print(f"Unchanged: {stats['unchanged']} ({stats['unchanged']/max(n,1)*100:.1f}%)")
        print(f"Avg sentences: {stats['avg_sentences_before']} → {stats['avg_sentences_after']}")
        print(f"Avg chars: {stats['avg_chars_before']:.0f} → {stats['avg_chars_after']:.0f}")
        print(f"Output: {output_path}")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Post-process filmot candidates: trim to first-person context"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help=f"Input JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="Process only first N records (0 = all)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    postprocess(
        input_path=args.input,
        output_path=args.output,
        max_records=args.max_records,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
