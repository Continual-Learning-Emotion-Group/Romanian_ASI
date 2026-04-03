"""
MASIVE-style bootstrapping for seed enrichment.

Finds "I feel X and Y" conjunction patterns in text. When X is a known seed
word, Y is a candidate new word. Iterative: each round's new words become
seeds for the next round.

Starts from the 375-word merged seed (pipeline.seed.merged).
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline.utils.text_utils import normalize_text
from pipeline.utils.pattern_matcher import extract_sentence, MODIFIER_PATTERN
from pipeline.utils.stoplists import STOPWORDS, infer_gender
from pipeline.utils.corpus_reader import iter_corpus

# ---------------------------------------------------------------------------
# Conjunction pattern building
# ---------------------------------------------------------------------------

# Verb forms for conjunction patterns (normalized, no diacritics).
# Only unambiguous "simt" family — no "sunt/eram/am fost" (too ambiguous),
# no plural, no "mă fac" or "o să fiu".
VERB_FORMS = [
    r"ma\s+simt",
    r"m-?am\s+simtit",
    r"ma\s+simteam",
    r"o\s+sa\s+ma\s+simt",
    r"m-?as\s+simti",
    r"sa\s+ma\s+simt",
]

# Conjunctions — longer alternatives first to avoid partial matches
CONJUNCTIONS = r"(?:dar\s+si|ba\s+chiar|si|dar|insa)"

# Fully normalized modifier pattern for conjunction matching
_CONJ_MOD = r'(?:(?:foarte|mai|putin|cam|destul\s+de|asa\s+de|tot|deja|chiar|atat\s+de)\s+)?'


def build_conjunction_patterns(seed_words: List[str]) -> List[re.Pattern]:
    """
    Build conjunction regex patterns: VERB [mod] X CONJ [mod] Y
    where X is from seed, Y is an open capture.
    """
    normalized_seeds = sorted(
        set(normalize_text(w) for w in seed_words),
        key=len, reverse=True,
    )
    x_alt = "|".join(re.escape(w) for w in normalized_seeds)

    patterns = []
    for verb in VERB_FORMS:
        pat_str = (
            r"\b(?:nu\s+)?"
            + verb
            + r"\s+"
            + _CONJ_MOD
            + r"(" + x_alt + r")"
            + r"\s*[,]?\s*"
            + r"(?:" + CONJUNCTIONS + r"|,)"
            + r"\s+"
            + _CONJ_MOD
            + r"(\w+)\b"
        )
        try:
            patterns.append(re.compile(pat_str, re.IGNORECASE | re.UNICODE))
        except re.error as e:
            print(f"Warning: Failed to compile conjunction pattern for verb '{verb}': {e}")

    return patterns


# ---------------------------------------------------------------------------
# Evidence tracking
# ---------------------------------------------------------------------------

@dataclass
class CandidateEvidence:
    """Evidence collected for a Y candidate word."""
    word: str
    normalized: str
    co_occurring_seeds: Set[str] = field(default_factory=set)
    seed_emotions: List[str] = field(default_factory=list)
    seed_genders: List[str] = field(default_factory=list)
    pattern_types: Set[str] = field(default_factory=set)
    source_sentences: List[str] = field(default_factory=list)
    source_count: int = 0

    def add_evidence(
        self, x_seed: str, x_emotions: List[str],
        x_gender: str, verb_form: str, sentence: str,
    ):
        self.co_occurring_seeds.add(x_seed)
        self.seed_emotions.extend(x_emotions)
        self.seed_genders.append(x_gender)
        self.pattern_types.add(verb_form)
        if len(self.source_sentences) < 10:
            self.source_sentences.append(sentence)
        self.source_count += 1


# ---------------------------------------------------------------------------
# Validation and scoring
# ---------------------------------------------------------------------------

def validate_candidate(
    candidate: CandidateEvidence,
    current_seed_normalized: Set[str],
    co_occurrence_threshold: int = 2,
) -> Tuple[bool, str]:
    """
    Validate a Y candidate.

    Filters:
    1. Min length >= 3
    2. Not already in seed
    3. Gender agreement with co-occurring X seeds
    4. Not a stopword (closed-class only)
    5. Co-occurrence threshold (>= N distinct X seeds)
    """
    w = candidate.normalized

    if len(w) < 3:
        return False, "too_short"

    if w in current_seed_normalized:
        return False, "already_in_seed"

    y_gender = infer_gender(candidate.word)
    if y_gender:
        matching_genders = [g for g in candidate.seed_genders if g == y_gender]
        if not matching_genders:
            return False, "gender_mismatch"

    if w in STOPWORDS:
        return False, "stopword"

    if len(candidate.co_occurring_seeds) < co_occurrence_threshold:
        return False, "low_co_occurrence"

    return True, "accepted"


def compute_confidence(candidate: CandidateEvidence) -> float:
    """Compute confidence score (0-1) for a candidate."""
    co_occ = min(len(candidate.co_occurring_seeds) / 5.0, 1.0)
    pat_div = min(len(candidate.pattern_types) / 3.0, 1.0)
    src_cnt = min(candidate.source_count / 5.0, 1.0)

    primary_verbs = {"ma simt", "m-am simtit", "ma simteam", "o sa ma simt"}
    has_primary = any(
        p in " ".join(normalize_text(v) for v in candidate.pattern_types)
        for p in primary_verbs
    )
    primary_bonus = 0.1 if has_primary else 0.0

    return min((co_occ * 0.4 + pat_div * 0.3 + src_cnt * 0.3) + primary_bonus, 1.0)


def infer_emotions(candidate: CandidateEvidence, threshold: float = 0.3) -> List[str]:
    """Infer emotions for Y via majority vote from co-occurring X seeds."""
    emotion_counts = Counter(candidate.seed_emotions)
    total = len(candidate.seed_emotions)
    if total == 0:
        return ["unknown"]
    return sorted(e for e, c in emotion_counts.items() if c / total >= threshold)


# ---------------------------------------------------------------------------
# Single-pass scanning (works for both loaded texts and streams)
# ---------------------------------------------------------------------------

def _scan_texts(
    text_iterator,
    conj_patterns: List[re.Pattern],
    norm_seed_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, CandidateEvidence], int]:
    """
    Scan texts for conjunction matches. Works with any iterator of (orig, normalized)
    text pairs or (record_id, text, source) tuples.

    Returns (candidates_dict, match_count).
    """
    candidates: Dict[str, CandidateEvidence] = {}
    match_count = 0

    for item in text_iterator:
        # Accept both (orig, norm) pairs and (id, text, source) tuples
        if len(item) == 2:
            orig_text, norm_text = item
        else:
            _, orig_text, _ = item
            norm_text = normalize_text(orig_text)

        for pat in conj_patterns:
            for m in pat.finditer(norm_text):
                groups = m.groups()
                if len(groups) < 2:
                    continue
                x_word = groups[-2].lower()
                y_word_norm = groups[-1].lower()

                x_info = norm_seed_lookup.get(x_word)
                if not x_info:
                    continue

                match_count += 1

                # Recover original Y form
                y_orig = y_word_norm
                for w in re.findall(r"\b\w+\b", orig_text, re.UNICODE):
                    if normalize_text(w) == y_word_norm:
                        y_orig = w
                        break

                sentence = extract_sentence(orig_text, m.start(), m.end())
                verb_form = m.group(0)[:30]

                if y_word_norm not in candidates:
                    candidates[y_word_norm] = CandidateEvidence(
                        word=y_orig, normalized=y_word_norm,
                    )
                candidates[y_word_norm].add_evidence(
                    x_seed=x_word,
                    x_emotions=x_info["emotions"],
                    x_gender=x_info["gender"],
                    verb_form=verb_form,
                    sentence=sentence,
                )

    return candidates, match_count


def _validate_and_accept(
    candidates: Dict[str, CandidateEvidence],
    current_seed_normalized: Set[str],
    co_occurrence_threshold: int,
    verbose: bool,
) -> Tuple[List[Dict], Counter]:
    """Validate candidates and return accepted list + rejection stats."""
    accepted = []
    rejected_reasons = Counter()

    for y_norm, evidence in candidates.items():
        valid, reason = validate_candidate(
            evidence, current_seed_normalized, co_occurrence_threshold,
        )
        if valid:
            confidence = compute_confidence(evidence)
            emotions = infer_emotions(evidence)
            y_gender = infer_gender(evidence.word) or "m"
            accepted.append({
                "word": evidence.word,
                "normalized": y_norm,
                "emotions": emotions,
                "gender": y_gender,
                "confidence": round(confidence, 3),
                "co_occurring_seeds": sorted(evidence.co_occurring_seeds),
                "pattern_types": sorted(evidence.pattern_types),
                "source_count": evidence.source_count,
                "sample_sentences": evidence.source_sentences[:3],
            })
        else:
            rejected_reasons[reason] += 1

    if verbose:
        print(f"  Accepted: {len(accepted)}")
        print(f"  Rejected: {dict(rejected_reasons)}")
        if accepted:
            top = sorted(accepted, key=lambda x: -x["confidence"])[:10]
            print(f"  Top new words: {[a['word'] for a in top]}")

    return accepted, rejected_reasons


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _build_seed_lookup(
    seed: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build normalized seed lookup from seed dict."""
    lookup = {}
    for word, info in seed.items():
        lookup[normalize_text(word)] = {
            "word": word,
            "emotions": info.get("emotions", []),
            "gender": info.get("gender", infer_gender(word) or "m"),
        }
    return lookup


def run_bootstrapping(
    data_dir: Path,
    seed: Dict[str, Dict[str, Any]],
    rounds: int = 4,
    co_occurrence_threshold: int = 2,
    sources: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run multi-round bootstrapping on JSONL files (loaded into memory).

    Args:
        data_dir: Directory with JSONL corpus files.
        seed: Starting seed — word → {"emotions": [...], "gender": "m"/"f"}.
        rounds: Number of bootstrapping rounds.
        co_occurrence_threshold: Min distinct X seeds for Y to be accepted.
        sources: Which JSONL files to use (None = all).
        verbose: Print progress.

    Returns:
        {"new_words": {word: info}, "provenance": {...}, "expanded_seed": {...}}
    """
    current_seed: Dict[str, Dict[str, Any]] = dict(seed)
    provenance = {"initial_seed_size": len(current_seed), "rounds": []}
    original_seed_normalized = {normalize_text(w) for w in seed}

    if verbose:
        print(f"\nBootstrapping: starting with {len(current_seed)} seed words")
        print(f"  Rounds: {rounds}, Co-occurrence threshold: {co_occurrence_threshold}")

    # Load all texts once for multi-round scanning
    print("\nLoading corpus...")
    texts = []
    for record_id, text, source in iter_corpus(data_dir, sources=sources):
        texts.append(text)

    if verbose:
        print(f"  Loaded {len(texts):,} texts")

    normalized_texts = [(t, normalize_text(t)) for t in texts]

    for round_num in range(1, rounds + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Round {round_num} (seed size: {len(current_seed)})")
            print(f"{'=' * 60}")

        conj_patterns = build_conjunction_patterns(list(current_seed.keys()))
        norm_seed_lookup = _build_seed_lookup(current_seed)
        current_seed_normalized = set(norm_seed_lookup.keys())

        candidates, match_count = _scan_texts(
            normalized_texts, conj_patterns, norm_seed_lookup,
        )

        if verbose:
            print(f"  Conjunction matches: {match_count}")
            print(f"  Unique Y candidates: {len(candidates)}")

        accepted, rejected_reasons = _validate_and_accept(
            candidates, current_seed_normalized, co_occurrence_threshold, verbose,
        )

        new_words_added = 0
        for a in accepted:
            if a["normalized"] not in current_seed_normalized:
                current_seed[a["word"]] = {
                    "emotions": a["emotions"],
                    "gender": a["gender"],
                }
                new_words_added += 1

        provenance["rounds"].append({
            "round": round_num,
            "seed_size_before": len(current_seed) - new_words_added,
            "seed_size_after": len(current_seed),
            "conjunction_matches": match_count,
            "unique_candidates": len(candidates),
            "accepted": len(accepted),
            "new_words_added": new_words_added,
            "rejected_reasons": dict(rejected_reasons),
            "accepted_words": [
                {"word": a["word"], "emotions": a["emotions"],
                 "confidence": a["confidence"]}
                for a in sorted(accepted, key=lambda x: -x["confidence"])
            ],
        })

        if new_words_added == 0:
            if verbose:
                print(f"\n  No new words — stopping early at round {round_num}")
            break

    new_words = {
        w: info for w, info in current_seed.items()
        if normalize_text(w) not in original_seed_normalized
    }
    provenance["final_seed_size"] = len(current_seed)
    provenance["new_words_count"] = len(new_words)

    return {"new_words": new_words, "expanded_seed": current_seed, "provenance": provenance}


def run_bootstrapping_streaming(
    text_iterator,
    seed: Dict[str, Dict[str, Any]],
    co_occurrence_threshold: int = 2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run single-pass bootstrapping on a streaming text source (e.g., FULG).

    No multi-round — scans the stream once with the full seed.

    Args:
        text_iterator: Iterator yielding (record_id, text, source) tuples.
        seed: Starting seed — word → {"emotions": [...], "gender": "m"/"f"}.
        co_occurrence_threshold: Min distinct X seeds for Y to be accepted.
        verbose: Print progress.

    Returns:
        {"new_words": {word: info}, "provenance": {...}}
    """
    original_seed_normalized = {normalize_text(w) for w in seed}
    conj_patterns = build_conjunction_patterns(list(seed.keys()))
    norm_seed_lookup = _build_seed_lookup(seed)

    if verbose:
        print(f"\nBootstrapping (streaming): {len(seed)} seed words, single pass")

    candidates, match_count = _scan_texts(
        text_iterator, conj_patterns, norm_seed_lookup,
    )

    if verbose:
        print(f"  Conjunction matches: {match_count}")
        print(f"  Unique Y candidates: {len(candidates)}")

    accepted, rejected_reasons = _validate_and_accept(
        candidates, original_seed_normalized, co_occurrence_threshold, verbose,
    )

    new_words = {}
    for a in accepted:
        if a["normalized"] not in original_seed_normalized:
            new_words[a["word"]] = {
                "emotions": a["emotions"],
                "gender": a["gender"],
            }

    provenance = {
        "initial_seed_size": len(seed),
        "mode": "streaming",
        "conjunction_matches": match_count,
        "unique_candidates": len(candidates),
        "accepted": len(accepted),
        "new_words_count": len(new_words),
        "rejected_reasons": dict(rejected_reasons),
        "accepted_words": [
            {"word": a["word"], "emotions": a["emotions"],
             "confidence": a["confidence"]}
            for a in sorted(accepted, key=lambda x: -x["confidence"])
        ],
    }

    return {"new_words": new_words, "provenance": provenance}
