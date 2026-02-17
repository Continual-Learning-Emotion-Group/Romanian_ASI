#!/usr/bin/env python3
"""
MASIVE-style bootstrapping for Romanian ASI.

Starts with 6 Ekman basic emotions (~24 adjective forms), then iteratively
expands the seed by mining "I feel X and Y" conjunction patterns from
RedditRoAP (26,517 posts) + PoPreRo (28,107 posts).

Usage:
    python -m experiments.bootstrapping.bootstrap_candidates
    python -m experiments.bootstrapping.bootstrap_candidates --rounds 4 --co-occurrence-threshold 2
    python -m experiments.bootstrapping.bootstrap_candidates --sample 20
"""

import re
import csv
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime

# Reuse from existing modules
from scripts.ro_asi.pattern_matcher import (
    PatternMatcher, normalize_text, remove_diacritics,
    extract_sentence,
)

# Fully normalized modifier pattern for conjunction matching on normalized text
MODIFIER_PATTERN = r'(?:(?:foarte|mai|putin|cam|destul\s+de|asa\s+de|tot|deja|chiar|atat\s+de)\s+)?'

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / "data"
SMALL_DATASETS = BASE_PATH / "small_datasets"

# --------------------------------------------------------------------------- #
# 1. Ekman Seed (~24 adjective forms)
# --------------------------------------------------------------------------- #

EKMAN_SEED: Dict[str, Dict[str, Any]] = {
    # Joy
    "fericit":   {"emotions": ["joy"], "gender": "m"},
    "fericită":  {"emotions": ["joy"], "gender": "f"},
    "bucuros":   {"emotions": ["joy"], "gender": "m"},
    "bucuroasă": {"emotions": ["joy"], "gender": "f"},
    "vesel":     {"emotions": ["joy"], "gender": "m"},
    "veselă":    {"emotions": ["joy"], "gender": "f"},
    # Sadness
    "trist":     {"emotions": ["sadness"], "gender": "m"},
    "tristă":    {"emotions": ["sadness"], "gender": "f"},
    # Anger
    "furios":    {"emotions": ["anger"], "gender": "m"},
    "furioasă":  {"emotions": ["anger"], "gender": "f"},
    "supărat":   {"emotions": ["anger"], "gender": "m"},
    "supărată":  {"emotions": ["anger"], "gender": "f"},
    # Fear
    "speriat":       {"emotions": ["fear"], "gender": "m"},
    "speriată":      {"emotions": ["fear"], "gender": "f"},
    "înspăimântat":  {"emotions": ["fear"], "gender": "m"},
    "înspăimântată": {"emotions": ["fear"], "gender": "f"},
    # Disgust
    "dezgustat":  {"emotions": ["disgust"], "gender": "m"},
    "dezgustată": {"emotions": ["disgust"], "gender": "f"},
    "scârbit":    {"emotions": ["disgust"], "gender": "m"},
    "scârbită":   {"emotions": ["disgust"], "gender": "f"},
    # Surprise
    "surprins":  {"emotions": ["surprise"], "gender": "m"},
    "surprinsă": {"emotions": ["surprise"], "gender": "f"},
    "uimit":     {"emotions": ["surprise"], "gender": "m"},
    "uimită":    {"emotions": ["surprise"], "gender": "f"},
}

# --------------------------------------------------------------------------- #
# 2. Data Loading
# --------------------------------------------------------------------------- #

def load_reddit_roap(base_path: Path) -> List[str]:
    """Load RedditRoAP texts from parquet."""
    file_path = base_path / "RedditRoAP" / "train.parquet"
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, skipping RedditRoAP")
        return []
    df = pd.read_parquet(file_path)
    texts = []
    for _, row in df.iterrows():
        text = str(row.get("TEXT", "")).strip()
        if text:
            texts.append(text)
    print(f"  RedditRoAP: {len(texts)} texts")
    return texts


def load_poprero(base_path: Path) -> List[str]:
    """Load PoPreRo texts from CSV."""
    texts = []
    for split in ["train", "validation", "test"]:
        file_path = base_path / "PoPreRo" / "Dataset" / f"{split}.csv"
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("full_text", "").strip()
                if text:
                    texts.append(text)
    print(f"  PoPreRo: {len(texts)} texts")
    return texts


def load_all_texts(base_path: Path) -> List[str]:
    """Load all texts from RedditRoAP + PoPreRo."""
    print("Loading datasets...")
    texts = load_reddit_roap(base_path) + load_poprero(base_path)
    print(f"  Total: {len(texts)} texts")
    return texts

# --------------------------------------------------------------------------- #
# 3. Conjunction Patterns
# --------------------------------------------------------------------------- #

# Verb forms for conjunction patterns (normalized, no diacritics)
VERB_FORMS = [
    # Primary: "mă simt" family (normalized — no diacritics)
    r"ma\s+simt",
    r"m-?am\s+simtit",
    r"ma\s+simteam",
    # Secondary: "sunt" family
    r"sunt",
    r"eram",
    r"am\s+fost",
    r"suntem",
    # Plural: "ne simțim" family (normalized)
    r"ne\s+simtim",
    r"ne-?am\s+simtit",
]

# Conjunctions (normalized — no diacritics, since we match on normalized text)
# Longer alternatives first to avoid partial matches ("dar si" before "dar")
CONJUNCTIONS = r"(?:dar\s+si|ba\s+chiar|si|dar|insa)"

# Romanian function words to reject (stored in normalized form)
FUNCTION_WORDS = {
    # Pronouns
    "eu", "tu", "el", "ea", "noi", "voi", "ei", "ele",
    "meu", "mea", "mei", "mele", "tau", "ta", "tai", "tale",
    "sau", "sa", "sai", "sale", "lor", "nostru", "noastra",
    "mine", "tine", "sine", "mie", "tie", "sie",
    # Relative/interrogative
    "care", "ce", "cine", "unde", "cand", "cum", "cat", "cate", "cati",
    # Articles / determiners
    "un", "una", "niste", "al", "ai", "ale",
    "acest", "aceasta", "acesta", "aceste", "acesti",
    "acel", "acea", "acei", "acele", "acela", "aceea",
    "cel", "cea", "cei", "cele",
    "tot", "toata", "toti", "toate", "toat",
    "alt", "alta", "alti", "alte", "altul", "altceva",
    "fiecare", "oricare", "orice", "niciun", "nicio",
    # Prepositions / conjunctions
    "la", "de", "pe", "cu", "in", "din", "prin", "pentru",
    "ca", "daca", "dar", "si", "sau", "ori", "nici",
    "fara", "despre", "peste", "intre", "pana", "dupa",
    # Adverbs / particles
    "nu", "mai", "doar", "chiar", "deja", "inca",
    "asa", "asta", "atat", "atata", "apoi", "acum",
    "aici", "acolo", "sus", "jos", "bine", "rau", "mult",
    "putin", "des", "rar", "mereu", "niciodata",
    "foarte", "probabil", "sigur", "poate", "oricum",
    "atunci", "totusi", "insa", "deci", "asadar",
    "cam", "prea", "destul", "tocmai", "macar", "oare",
    # Common verbs (not adjectives)
    "am", "sunt", "este", "era", "eram", "fost",
    "are", "avea", "avem", "aveti", "aveau",
    "face", "fac", "facut", "faci", "facem",
    "pot", "putea", "putem", "poti",
    "vrea", "vreau", "vrei", "vrem", "vor",
    "zice", "zis", "spune", "spus",
    "stiu", "stie", "stim", "stiti",
    "cred", "crede", "credem",
    "trebuie", "trebui",
    # Common non-affective adjectives/words often appearing after "and"
    "nici", "nimic", "nimeni", "nicaieri",
}

# Feminine adjective suffixes
FEMININE_SUFFIXES = ("ă", "ată", "ită", "ută", "oasă", "easă")

# Masculine adjective endings (common)
MASCULINE_ENDINGS = ("t", "s", "c", "l", "r", "os", "at", "it", "ut")


def infer_gender(word: str) -> Optional[str]:
    """Infer gender from Romanian adjective morphology."""
    w = word.lower()
    # Check feminine first (more distinctive endings)
    for suffix in FEMININE_SUFFIXES:
        norm_suffix = normalize_text(suffix)
        if normalize_text(w).endswith(norm_suffix):
            return "f"
    # Check masculine
    norm_w = normalize_text(w)
    for ending in MASCULINE_ENDINGS:
        if norm_w.endswith(ending):
            return "m"
    # Consonant-final is typically masculine
    if norm_w and norm_w[-1] not in "aeiou":
        return "m"
    return None


def build_conjunction_patterns(seed_words: List[str]) -> List[re.Pattern]:
    """
    Build conjunction regex patterns: VERB [mod] X CONJ [mod] Y
    where X is from seed, Y is an open capture.
    """
    # Build X alternation from seed (normalized)
    normalized_seeds = sorted(
        set(normalize_text(w) for w in seed_words),
        key=len, reverse=True,
    )
    x_alt = "|".join(re.escape(w) for w in normalized_seeds)

    patterns = []
    for verb in VERB_FORMS:
        # Optional negation, verb, modifier, X, conjunction/comma, modifier, Y capture
        pat_str = (
            r"\b(?:nu\s+)?"
            + verb
            + r"\s+"
            + MODIFIER_PATTERN
            + r"(" + x_alt + r")"
            + r"\s*[,]?\s*"
            + r"(?:" + CONJUNCTIONS + r"|,)"
            + r"\s+"
            + MODIFIER_PATTERN
            + r"(\w+)\b"
        )
        try:
            patterns.append(re.compile(pat_str, re.IGNORECASE | re.UNICODE))
        except re.error as e:
            print(f"Warning: Failed to compile conjunction pattern for verb '{verb}': {e}")

    return patterns

# --------------------------------------------------------------------------- #
# 4. Evidence & Candidate dataclass
# --------------------------------------------------------------------------- #

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

    def add_evidence(self, x_seed: str, x_emotions: List[str],
                     x_gender: str, verb_form: str, sentence: str):
        self.co_occurring_seeds.add(x_seed)
        self.seed_emotions.extend(x_emotions)
        self.seed_genders.append(x_gender)
        self.pattern_types.add(verb_form)
        if len(self.source_sentences) < 10:
            self.source_sentences.append(sentence)
        self.source_count += 1

# --------------------------------------------------------------------------- #
# 5. Validation
# --------------------------------------------------------------------------- #

def validate_candidate(
    candidate: CandidateEvidence,
    current_seed: Dict[str, Dict[str, Any]],
    co_occurrence_threshold: int = 2,
) -> tuple[bool, str]:
    """
    Validate a Y candidate. Returns (accepted, reason).

    Filters:
    1. Min length >= 3
    2. Not already in seed
    3. Gender agreement with co-occurring X seeds
    4. Not a function word
    5. Co-occurrence threshold (>= 2 distinct X seeds)
    """
    w = candidate.normalized

    # 1. Min length
    if len(w) < 3:
        return False, "too_short"

    # 2. Already in seed
    if w in {normalize_text(k) for k in current_seed}:
        return False, "already_in_seed"

    # 3. Gender agreement
    y_gender = infer_gender(candidate.word)
    if y_gender:
        # Check that at least some X seeds share the same gender
        matching_genders = [g for g in candidate.seed_genders if g == y_gender]
        if not matching_genders:
            return False, "gender_mismatch"

    # 4. Function word (FUNCTION_WORDS is already normalized)
    if w in FUNCTION_WORDS:
        return False, "function_word"

    # 5. Co-occurrence threshold
    if len(candidate.co_occurring_seeds) < co_occurrence_threshold:
        return False, "low_co_occurrence"

    return True, "accepted"


def compute_confidence(candidate: CandidateEvidence) -> float:
    """Compute confidence score (0-1) for a candidate."""
    # Factors: co-occurrence count, pattern diversity, source count
    co_occ = min(len(candidate.co_occurring_seeds) / 5.0, 1.0)  # saturates at 5
    pat_div = min(len(candidate.pattern_types) / 3.0, 1.0)      # saturates at 3
    src_cnt = min(candidate.source_count / 5.0, 1.0)             # saturates at 5

    # Primary patterns ("ma simt" family) get a boost
    primary_verbs = {"ma simt", "m-am simtit", "ma simteam", "ne simtim"}
    has_primary = any(
        normalize_text(p) in " ".join(normalize_text(v) for v in candidate.pattern_types)
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
    return sorted(
        e for e, c in emotion_counts.items()
        if c / total >= threshold
    )

# --------------------------------------------------------------------------- #
# 6. Bootstrapping Loop
# --------------------------------------------------------------------------- #

def run_bootstrapping(
    texts: List[str],
    rounds: int = 4,
    co_occurrence_threshold: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Run MASIVE-style bootstrapping.

    Returns dict with expanded_seed, provenance, and stats.
    """
    # Initialize seed
    current_seed: Dict[str, Dict[str, Any]] = dict(EKMAN_SEED)
    provenance = {
        "initial_seed_size": len(current_seed),
        "rounds": [],
    }

    if verbose:
        print(f"\nStarting bootstrapping with {len(current_seed)} Ekman seed words")
        print(f"  Rounds: {rounds}, Co-occurrence threshold: {co_occurrence_threshold}")

    # Normalize texts once for faster scanning
    normalized_texts = [(t, normalize_text(t)) for t in texts]

    for round_num in range(1, rounds + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_num}")
            print(f"{'='*60}")
            print(f"  Current seed size: {len(current_seed)}")

        # 1. Compile conjunction patterns with current seed
        seed_words = list(current_seed.keys())
        conj_patterns = build_conjunction_patterns(seed_words)

        if verbose:
            print(f"  Compiled {len(conj_patterns)} conjunction patterns")

        # Build normalized seed lookup
        norm_seed_lookup: Dict[str, Dict[str, Any]] = {}
        for word, info in current_seed.items():
            norm_seed_lookup[normalize_text(word)] = {
                "word": word,
                "emotions": info["emotions"],
                "gender": info["gender"],
            }

        # 2. Scan all texts
        candidates: Dict[str, CandidateEvidence] = {}
        match_count = 0

        for orig_text, norm_text in normalized_texts:
            for pat in conj_patterns:
                for m in pat.finditer(norm_text):
                    groups = m.groups()
                    if len(groups) < 2:
                        continue
                    x_word = groups[-2].lower()
                    y_word_norm = groups[-1].lower()

                    # Look up X in seed
                    x_info = norm_seed_lookup.get(x_word)
                    if not x_info:
                        continue

                    match_count += 1

                    # Find original Y word from text
                    y_start = m.start(len(groups))  # last group
                    y_orig = y_word_norm
                    # Try to recover original form from text
                    for w in re.findall(r"\b\w+\b", orig_text, re.UNICODE):
                        if normalize_text(w) == y_word_norm:
                            y_orig = w
                            break

                    # Extract sentence for context
                    sentence = extract_sentence(orig_text, m.start(), m.end())

                    # Determine verb form
                    verb_form = m.group(0)[:30]

                    # Add evidence
                    if y_word_norm not in candidates:
                        candidates[y_word_norm] = CandidateEvidence(
                            word=y_orig,
                            normalized=y_word_norm,
                        )
                    candidates[y_word_norm].add_evidence(
                        x_seed=x_word,
                        x_emotions=x_info["emotions"],
                        x_gender=x_info["gender"],
                        verb_form=verb_form,
                        sentence=sentence,
                    )

        if verbose:
            print(f"  Total conjunction matches: {match_count}")
            print(f"  Unique Y candidates: {len(candidates)}")

        # 3. Validate candidates
        accepted = []
        rejected_words = []
        rejected_reasons = Counter()

        for y_norm, evidence in candidates.items():
            valid, reason = validate_candidate(
                evidence, current_seed, co_occurrence_threshold,
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
                rejected_words.append({
                    "word": evidence.word,
                    "reason": reason,
                    "co_occurring_seeds": sorted(evidence.co_occurring_seeds),
                })

        if verbose:
            print(f"  Accepted: {len(accepted)}")
            print(f"  Rejected: {dict(rejected_reasons)}")
            if accepted:
                top = sorted(accepted, key=lambda x: -x["confidence"])[:10]
                print(f"  Top new words: {[a['word'] for a in top]}")

        # 4. Add validated Ys to seed
        new_words_added = 0
        for a in accepted:
            if a["normalized"] not in {normalize_text(k) for k in current_seed}:
                current_seed[a["word"]] = {
                    "emotions": a["emotions"],
                    "gender": a["gender"],
                }
                new_words_added += 1

        # 5. Log round statistics
        round_info = {
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
                 "confidence": a["confidence"],
                 "co_occurring_seeds": a["co_occurring_seeds"]}
                for a in sorted(accepted, key=lambda x: -x["confidence"])
            ],
            "rejected_words": rejected_words,
        }
        provenance["rounds"].append(round_info)

        # 6. Early stop
        if new_words_added == 0:
            if verbose:
                print(f"\n  No new words found — stopping early at round {round_num}")
            break

    provenance["final_seed_size"] = len(current_seed)
    return {
        "expanded_seed": current_seed,
        "provenance": provenance,
    }

# --------------------------------------------------------------------------- #
# 7. Final Extraction
# --------------------------------------------------------------------------- #

def run_final_extraction(
    texts: List[str],
    expanded_seed: Dict[str, Dict[str, Any]],
    output_path: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full extraction using expanded seed + PatternMatcher.
    Outputs in common schema.
    """
    # Build word_to_emotions dict for PatternMatcher
    word_to_emotions = {
        word: info["emotions"] for word, info in expanded_seed.items()
    }

    matcher = PatternMatcher(word_to_emotions)

    stats = {
        "total_processed": 0,
        "total_matches": 0,
        "unique_texts_matched": 0,
        "by_pattern": defaultdict(int),
        "by_emotion": defaultdict(int),
        "unique_seed_words": set(),
    }

    seen_hashes: Set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for text in texts:
            stats["total_processed"] += 1
            if not text:
                continue

            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in seen_hashes:
                continue

            matches = matcher.find_matches(text, extract_sentences=True)
            if matches:
                seen_hashes.add(text_hash)
                stats["unique_texts_matched"] += 1

                for m in matches:
                    candidate = {
                        "id": f"bootstrap_{text_hash[:12]}_{m.start_pos}",
                        "text": text,
                        "matched_sentence": m.matched_text,
                        "pattern_used": m.pattern_name,
                        "pattern_category": m.pattern_category,
                        "seed_word": m.seed_word,
                        "seed_word_normalized": m.seed_word_normalized,
                        "emotion_category": m.emotions,
                        "source": "bootstrapping",
                        "extraction_strategy": "masive_bootstrapping",
                    }
                    out_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

                    stats["total_matches"] += 1
                    stats["by_pattern"][m.pattern_name] += 1
                    stats["unique_seed_words"].add(m.seed_word_normalized)
                    for e in m.emotions:
                        stats["by_emotion"][e] += 1

    stats["unique_seed_words_count"] = len(stats["unique_seed_words"])
    stats["unique_seed_words"] = sorted(stats["unique_seed_words"])
    stats["by_pattern"] = dict(stats["by_pattern"])
    stats["by_emotion"] = dict(stats["by_emotion"])

    if verbose:
        print(f"\nFinal extraction:")
        print(f"  Processed: {stats['total_processed']}")
        print(f"  Matched texts: {stats['unique_texts_matched']}")
        print(f"  Total matches: {stats['total_matches']}")
        print(f"  Unique seed words used: {stats['unique_seed_words_count']}")

    return stats

# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #

def sample_output(path: Path, n: int = 10):
    """Print sample candidates from output file."""
    print(f"\n{'='*60}")
    print(f"Sample Candidates (first {n})")
    print(f"{'='*60}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            c = json.loads(line)
            print(f"\n[{i+1}] Pattern: {c['pattern_used']} ({c['pattern_category']})")
            print(f"    Seed: {c['seed_word']} → {c['emotion_category']}")
            sent = c["matched_sentence"]
            if len(sent) > 120:
                sent = sent[:120] + "..."
            print(f"    Matched: \"{sent}\"")


def main():
    parser = argparse.ArgumentParser(
        description="MASIVE-style bootstrapping for Romanian ASI"
    )
    parser.add_argument(
        "--rounds", type=int, default=4,
        help="Number of bootstrapping rounds (default: 4)"
    )
    parser.add_argument(
        "--co-occurrence-threshold", type=int, default=2,
        help="Min distinct X seeds a Y must co-occur with (default: 2)"
    )
    parser.add_argument(
        "--sample", type=int, default=10,
        help="Number of sample candidates to print (default: 10)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    verbose = not args.quiet
    started = datetime.now()

    # Load data
    texts = load_all_texts(SMALL_DATASETS)
    if not texts:
        print("Error: No texts loaded. Check dataset paths.")
        return 1

    # Run bootstrapping
    result = run_bootstrapping(
        texts,
        rounds=args.rounds,
        co_occurrence_threshold=args.co_occurrence_threshold,
        verbose=verbose,
    )
    expanded_seed = result["expanded_seed"]
    provenance = result["provenance"]

    # Save expanded seed
    seed_path = DATA_PATH / "bootstrap_expanded_seed.json"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_export = {
        word: info for word, info in expanded_seed.items()
    }
    with open(seed_path, "w", encoding="utf-8") as f:
        json.dump(seed_export, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\nExpanded seed saved to: {seed_path}")
        print(f"  Seed grew from {provenance['initial_seed_size']} to {provenance['final_seed_size']} words")

    # Save provenance
    prov_path = DATA_PATH / "bootstrap_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"Provenance saved to: {prov_path}")

    # Run final extraction
    output_path = DATA_PATH / "bootstrapped_asi_candidates.jsonl"
    extraction_stats = run_final_extraction(
        texts, expanded_seed, output_path, verbose=verbose,
    )

    # Save extraction stats
    stats_path = output_path.with_suffix(".stats.json")
    extraction_stats["bootstrapping"] = {
        "rounds_run": len(provenance["rounds"]),
        "initial_seed": provenance["initial_seed_size"],
        "final_seed": provenance["final_seed_size"],
        "co_occurrence_threshold": args.co_occurrence_threshold,
        "duration_seconds": (datetime.now() - started).total_seconds(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(extraction_stats, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nExtraction stats saved to: {stats_path}")
        print(f"Output saved to: {output_path}")

    # Compare: Ekman-only vs expanded
    if verbose:
        print(f"\n{'='*60}")
        print("Yield Comparison")
        print(f"{'='*60}")

        # Quick Ekman-only extraction count
        ekman_w2e = {w: info["emotions"] for w, info in EKMAN_SEED.items()}
        ekman_matcher = PatternMatcher(ekman_w2e)
        ekman_matches = 0
        for text in texts:
            if ekman_matcher.find_matches(text, max_matches=1):
                ekman_matches += 1

        print(f"  Ekman seed ({len(EKMAN_SEED)} words): {ekman_matches} texts matched")
        print(f"  Expanded seed ({len(expanded_seed)} words): {extraction_stats['unique_texts_matched']} texts matched")
        if ekman_matches > 0:
            improvement = (extraction_stats["unique_texts_matched"] - ekman_matches) / ekman_matches * 100
            print(f"  Improvement: {improvement:+.1f}%")

    # Print sample
    if args.sample > 0 and extraction_stats["total_matches"] > 0:
        sample_output(output_path, args.sample)

    return 0


if __name__ == "__main__":
    exit(main())
