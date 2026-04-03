"""
Romanian "I Feel" pattern matcher for the ASI pipeline.

20 patterns covering present, imperfect, perfect, future, conditional,
subjunctive, and colloquial forms. Supports lemma-based seeds with
automatic inflection expansion via MULTEXT-East.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .text_utils import normalize_text, remove_diacritics

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """Represents a pattern match in text."""
    pattern_name: str
    pattern_category: str       # "primary" or "secondary"
    matched_text: str           # Full sentence containing match
    seed_word: str              # Original form with diacritics
    seed_word_normalized: str   # Normalized form (no diacritics, lowercase)
    start_pos: int
    end_pos: int
    emotions: List[str]

# ---------------------------------------------------------------------------
# Modifier pattern (optional adverbs between verb and seed word)
# ---------------------------------------------------------------------------

MODIFIER_PATTERN = (
    r'(?:(?:foarte|mai|putin|pu[tț]in|cam|destul\s+de|a[sș]a\s+de'
    r'|tot|deja|chiar|at[aâ]t\s+de)\s+)?'
)

# ---------------------------------------------------------------------------
# Curated emotion nouns for noun-only patterns
# ---------------------------------------------------------------------------

EMOTION_NOUNS_ONLY = {
    # Fear-related
    "frica", "frică", "teama", "teamă", "groaza", "groază", "panica", "panică",
    "anxietate", "neliniste", "neliniște", "ingrijorare", "îngrijorare",
    # Sadness-related
    "tristete", "tristețe", "durere", "suferinta", "suferință", "chin",
    "dezamagire", "dezamăgire", "melancolie", "nostalgie", "dor",
    # Joy-related
    "bucurie", "fericire", "placere", "plăcere", "satisfactie", "satisfacție",
    "entuziasm", "veselie",
    # Anger-related
    "furie", "manie", "mânie", "nervozitate", "iritare", "frustrare",
    "indignare", "revolta", "revoltă", "ura", "ură",
    # Disgust-related
    "dezgust", "scarba", "scârbă", "greata", "greață", "repulsie",
    # Surprise-related
    "surpriza", "surpriză", "uimire", "mirare", "stupoare",
    # Trust-related
    "incredere", "încredere", "siguranta", "siguranță", "liniste", "liniște",
    "pace", "calm",
    # Other emotions
    "rusine", "rușine", "vina", "vină", "gelozie", "invidie",
    "mandrie", "mândrie", "recunostinta", "recunoștință", "iubire",
    "speranta", "speranță", "disperare", "nerabdare", "nerăbdare",
    "curiozitate",
}

# Emotion mappings for curated nouns (normalized forms)
NOUN_EMOTION_MAP = {
    # Fear
    "frica": ["fear"], "teama": ["fear"], "groaza": ["fear"],
    "panica": ["fear"], "anxietate": ["fear"], "neliniste": ["fear"],
    "ingrijorare": ["fear"],
    # Sadness
    "tristete": ["sadness"], "durere": ["sadness"], "suferinta": ["sadness"],
    "chin": ["sadness"], "dezamagire": ["sadness"], "melancolie": ["sadness"],
    "nostalgie": ["sadness"], "dor": ["sadness"],
    # Joy
    "bucurie": ["joy"], "fericire": ["joy"], "placere": ["joy"],
    "satisfactie": ["joy"], "entuziasm": ["joy", "anticipation"],
    "veselie": ["joy"],
    # Anger
    "furie": ["anger"], "manie": ["anger"], "nervozitate": ["anger", "fear"],
    "iritare": ["anger"], "frustrare": ["anger", "sadness"],
    "indignare": ["anger"], "revolta": ["anger"], "ura": ["anger", "disgust"],
    # Disgust
    "dezgust": ["disgust"], "scarba": ["disgust"], "greata": ["disgust"],
    "repulsie": ["disgust"],
    # Surprise
    "surpriza": ["surprise"], "uimire": ["surprise"], "mirare": ["surprise"],
    "stupoare": ["surprise"],
    # Trust
    "incredere": ["trust"], "siguranta": ["trust"], "liniste": ["trust"],
    "pace": ["trust"], "calm": ["trust"],
    # Mixed
    "rusine": ["sadness", "fear"], "vina": ["sadness"],
    "gelozie": ["anger", "sadness"], "invidie": ["anger", "sadness"],
    "mandrie": ["joy"], "recunostinta": ["joy", "trust"],
    "iubire": ["joy", "trust"], "speranta": ["anticipation", "joy"],
    "disperare": ["sadness", "fear"], "nerabdare": ["anticipation"],
    "curiozitate": ["anticipation"],
}

# ---------------------------------------------------------------------------
# Pattern definitions (20 patterns, first person singular only)
# ---------------------------------------------------------------------------

_MOD = MODIFIER_PATTERN

PATTERNS = [
    # === Primary: "mă simt" (I feel) variations ===

    ("ma_simt_present", "primary",
     r'\b(m[aă]\s+simt|me\s+simt)\s+' + _MOD + r'({SEED})\b', False),

    ("ma_simteam_imperfect", "primary",
     r'\b(m[aă]\s+sim[tț]eam|me\s+simteam)\s+' + _MOD + r'({SEED})\b', False),

    ("mam_simtit_perfect", "primary",
     r'\b(m-?am\s+sim[tț]it)\s+' + _MOD + r'({SEED})\b', False),

    ("ma_voi_simti_future", "primary",
     r'\b(m[aă]\s+voi\s+sim[tț]i|me\s+voi\s+simti)\s+' + _MOD + r'({SEED})\b', False),

    # Colloquial future — very common in informal Romanian
    ("o_sa_ma_simt_future", "primary",
     r'\b(o\s+s[aă]\s+m[aă]\s+simt)\s+' + _MOD + r'({SEED})\b', False),

    # Conditional — "I would feel"
    ("mas_simti_conditional", "primary",
     r'\b(m-?a[sș]\s+sim[tț]i)\s+' + _MOD + r'({SEED})\b', False),

    # Subjunctive — "to feel" / "that I feel"
    ("sa_ma_simt_subjunctive", "primary",
     r'\b(s[aă]\s+m[aă]\s+simt)\s+' + _MOD + r'({SEED})\b', False),

    ("simt_ca", "primary",
     r'\b(simt\s+c[aă])\s+' + _MOD + r'({SEED})\b', False),

    # "simt [emotion_noun]" — NOUN ONLY
    ("simt_noun", "primary",
     r'\b(simt)\s+' + _MOD + r'({SEED})\b', True),

    # "simțeam [emotion_noun]" — imperfect, NOUN ONLY
    ("simteam_noun", "primary",
     r'\b(sim[tț]eam)\s+' + _MOD + r'({SEED})\b', True),

    # === Secondary: "sunt" (I am), dative, possessive ===

    ("sunt_adj_present", "secondary",
     r'\b(sunt)\s+' + _MOD + r'({SEED})\b', False),

    ("eram_adj_imperfect", "secondary",
     r'\b(eram)\s+' + _MOD + r'({SEED})\b', False),

    ("am_fost_adj_perfect", "secondary",
     r'\b(am\s+fost)\s+' + _MOD + r'({SEED})\b', False),

    # Colloquial future of "to be"
    ("o_sa_fiu_future", "secondary",
     r'\b(o\s+s[aă]\s+fiu)\s+' + _MOD + r'({SEED})\b', False),

    # Reflexive "I become [state]"
    ("ma_fac_reflexive", "secondary",
     r'\b(m[aă]\s+fac)\s+' + _MOD + r'({SEED})\b', False),

    # Dative: "îmi este" — NOUN ONLY
    ("imi_este_present", "secondary",
     r'\b([îi]mi\s+este|imi\s+este)\s+' + _MOD + r'({SEED})\b', True),

    ("imi_era_imperfect", "secondary",
     r'\b([îi]mi\s+era|imi\s+era)\s+' + _MOD + r'({SEED})\b', True),

    # Short form: "mi-e" — NOUN ONLY
    ("mie_short", "secondary",
     r'\b(mi-?e|mi-?i)\s+' + _MOD + r'({SEED})\b', True),

    # "am [noun]" — NOUN ONLY to avoid participles
    ("am_noun_present", "secondary",
     r'\b(am)\s+' + _MOD + r'({SEED})\b', True),

    ("aveam_noun_imperfect", "secondary",
     r'\b(aveam)\s+' + _MOD + r'({SEED})\b', True),
]

# ---------------------------------------------------------------------------
# Trigger words and Filmot queries (derived from patterns)
# ---------------------------------------------------------------------------

# Single-word triggers for pre-filtering large corpora (e.g., FULG).
# These are verb stems that appear in the patterns above.
TRIGGER_WORDS = {
    "simt", "sunt", "eram", "fost", "mi-e", "mie",
    "imi", "îmi", "simteam", "simțeam", "simtit", "simțit",
}

# Multi-word triggers for the new patterns (too vague as single words)
TRIGGER_PHRASES = {
    "o sa ma simt", "o sa fiu", "m-as simti", "m-aș simți",
    "sa ma simt", "să mă simt", "ma fac",
}

# Filmot API queries — quoted phrases for subtitle search
FILMOT_QUERIES_PRIMARY = [
    '"mă simt"',
    '"mi-e"',
    '"m-am simțit"',
    '"îmi este"',
    '"mă simțeam"',
    '"simt că"',
    '"îmi era"',
    '"mă voi simți"',
    '"o să mă simt"',
    '"m-aș simți"',
    '"să mă simt"',
]

FILMOT_QUERIES_SECONDARY = [
    # No-diacritic variants
    '"ma simt"',
    '"m-am simtit"',
    '"imi este"',
    '"imi era"',
    '"ma simteam"',
    '"o sa ma simt"',
    '"m-as simti"',
    '"sa ma simt"',
    '"o sa fiu"',
    '"o să fiu"',
]


def get_trigger_words() -> Set[str]:
    """Get all trigger words/phrases for pre-filtering."""
    return TRIGGER_WORDS | TRIGGER_PHRASES


def get_filmot_queries(include_secondary: bool = True) -> List[str]:
    """Get Filmot API search queries."""
    queries = list(FILMOT_QUERIES_PRIMARY)
    if include_secondary:
        queries.extend(FILMOT_QUERIES_SECONDARY)
    return queries

# ---------------------------------------------------------------------------
# Pattern compilation
# ---------------------------------------------------------------------------


def build_seed_regex(seed_words: List[str]) -> str:
    """
    Build regex alternation for seed words.

    All words are normalized (lowercase, no diacritics) before building.
    Sorted longest-first to avoid partial matches.
    """
    normalized_seeds = set()
    for word in seed_words:
        normalized_seeds.add(normalize_text(word))

    sorted_seeds = sorted(normalized_seeds, key=len, reverse=True)
    escaped = [re.escape(word) for word in sorted_seeds]
    return '|'.join(escaped)


def compile_patterns(
    seed_words: List[str],
    noun_only_words: Optional[Set[str]] = None,
) -> List[Tuple[str, str, 're.Pattern', bool]]:
    """
    Compile pattern regexes with seed words.

    Returns list of (name, category, compiled_pattern, noun_only).
    """
    if noun_only_words is None:
        noun_only_words = EMOTION_NOUNS_ONLY

    seed_regex = build_seed_regex(seed_words)
    noun_regex = build_seed_regex(list(noun_only_words))

    compiled = []
    for name, category, template, noun_only in PATTERNS:
        current_regex = noun_regex if noun_only else seed_regex
        pattern_str = template.replace('{SEED}', current_regex)
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
            compiled.append((name, category, pattern, noun_only))
        except re.error as e:
            print(f"Warning: Failed to compile pattern {name}: {e}")

    return compiled


def extract_sentence(text: str, match_start: int, match_end: int) -> str:
    """Extract the sentence containing the match."""
    sentence_start = 0
    for i in range(match_start - 1, -1, -1):
        if text[i] in '.!?\n':
            sentence_start = i + 1
            break

    sentence_end = len(text)
    for i in range(match_end, len(text)):
        if text[i] in '.!?\n':
            sentence_end = i + 1
            break

    return text[sentence_start:sentence_end].strip()


def find_original_word(original_text: str, normalized_match: str, approx_pos: int) -> str:
    """Find the original (non-normalized) word in the text."""
    text_segment = original_text[max(0, approx_pos - 50):approx_pos + 50]

    for word in re.findall(r'\b\w+\b', text_segment, re.UNICODE):
        if normalize_text(word) == normalized_match:
            return word

    return normalized_match


# ---------------------------------------------------------------------------
# PatternMatcher class
# ---------------------------------------------------------------------------

class PatternMatcher:
    """
    Romanian affective state pattern matcher.

    Usage:
        from pipeline.seed.merged import build_seed
        seed = build_seed()
        matcher = PatternMatcher(seed["word_to_affect_categ"])
    """

    def __init__(
        self,
        word_to_emotions: Dict[str, List[str]],
        noun_words: Optional[List[str]] = None,
        expand_forms: bool = True,
    ):
        """
        Initialize matcher.

        Args:
            word_to_emotions: Dict mapping words to emotion lists.
            noun_words: Optional noun list for noun-only patterns.
            expand_forms: If True, auto-expand lemmas via MULTEXT-East.
        """
        # Optionally expand lemmas to all inflected forms
        if expand_forms:
            word_to_emotions = self._expand_word_to_emotions(word_to_emotions)

        self.word_to_emotions = word_to_emotions
        self.seed_words = list(word_to_emotions.keys())

        # Build normalized lookups
        self.normalized_to_original: Dict[str, List[str]] = {}
        self.normalized_to_emotions: Dict[str, List[str]] = {}

        for word, emotions in word_to_emotions.items():
            normalized = normalize_text(word)
            if normalized not in self.normalized_to_original:
                self.normalized_to_original[normalized] = []
            self.normalized_to_original[normalized].append(word)

            if normalized not in self.normalized_to_emotions:
                self.normalized_to_emotions[normalized] = []
            for e in emotions:
                if e not in self.normalized_to_emotions[normalized]:
                    self.normalized_to_emotions[normalized].append(e)

        # Noun words
        if noun_words is not None:
            noun_word_set = set(noun_words)
        else:
            noun_word_set = EMOTION_NOUNS_ONLY

        self.noun_only_normalized = set()
        for word in noun_word_set:
            self.noun_only_normalized.add(normalize_text(word))

        # Compile patterns
        all_normalized_seeds = list(self.normalized_to_original.keys())
        self.compiled_patterns = compile_patterns(all_normalized_seeds, noun_word_set)

        noun_only_count = sum(1 for _, _, _, n in self.compiled_patterns if n)
        print(f"PatternMatcher initialized:")
        print(f"  Seed forms: {len(all_normalized_seeds)} (from {len(word_to_emotions)} entries)")
        print(f"  Noun words: {len(noun_word_set)}")
        print(f"  Patterns: {len(self.compiled_patterns)} ({noun_only_count} noun-only)")

    @staticmethod
    def _expand_word_to_emotions(
        word_to_emotions: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Expand lemmas to all inflected + diacritic-stripped forms."""
        try:
            from .inflect import expand_lemma, get_multext_east
            multext = get_multext_east()
        except Exception:
            # If MULTEXT-East not available, return as-is
            return word_to_emotions

        expanded = {}
        for word, emotions in word_to_emotions.items():
            # Try adjective first, then noun, then adverb
            for pos in ("adjective", "noun", "adverb"):
                forms = expand_lemma(word, pos, multext)
                if len(forms) > 1:
                    break
            else:
                forms = {word, remove_diacritics(word)}

            for form in forms:
                if form not in expanded:
                    expanded[form] = emotions
                else:
                    # Merge emotions
                    for e in emotions:
                        if e not in expanded[form]:
                            expanded[form].append(e)

        return expanded

    def find_matches(
        self,
        text: str,
        extract_sentences: bool = True,
        max_matches: int = 100,
    ) -> List[PatternMatch]:
        """Find all pattern matches in text."""
        matches = []
        normalized = normalize_text(text)

        for pattern_name, category, pattern, noun_only in self.compiled_patterns:
            for match in pattern.finditer(normalized):
                if len(matches) >= max_matches:
                    break

                groups = match.groups()
                if len(groups) < 2:
                    continue

                seed_word_normalized = groups[1].lower()

                # Look up emotions
                emotions = self.normalized_to_emotions.get(seed_word_normalized, [])
                if not emotions and noun_only:
                    emotions = NOUN_EMOTION_MAP.get(seed_word_normalized, [])
                if not emotions:
                    continue

                original_seed = find_original_word(text, seed_word_normalized, match.start())

                if extract_sentences:
                    sentence = extract_sentence(text, match.start(), match.end())
                else:
                    sentence = match.group(0)

                matches.append(PatternMatch(
                    pattern_name=pattern_name,
                    pattern_category=category,
                    matched_text=sentence,
                    seed_word=original_seed,
                    seed_word_normalized=seed_word_normalized,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    emotions=emotions,
                ))

        return matches

    def match_text(self, text: str) -> Optional[PatternMatch]:
        """Find first match in text."""
        matches = self.find_matches(text, extract_sentences=True, max_matches=1)
        return matches[0] if matches else None

    def has_affective_pattern(self, text: str) -> bool:
        """Check if text contains any affective state pattern."""
        return self.match_text(text) is not None


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_seed = {
        "fericit": ["joy"],
        "trist": ["sadness"],
        "frică": ["fear"],
        "teamă": ["fear"],
        "furios": ["anger"],
        "bine": ["joy"],
        "rău": ["sadness", "anger"],
        "calm": ["trust"],
        "relaxat": ["trust"],
        "surprins": ["surprise"],
    }

    matcher = PatternMatcher(test_seed, expand_forms=True)

    test_cases = [
        "Mă simt fericit astăzi.",
        "Mă simt fericită și liberă.",        # feminine form (expanded)
        "M-am simțit foarte bine la petrecere.",
        "O să mă simt mai bine mâine.",       # colloquial future (NEW)
        "O să fiu trist fără tine.",           # colloquial future of "to be" (NEW)
        "M-aș simți mai relaxat acasă.",       # conditional (NEW)
        "Vreau să mă simt calm.",              # subjunctive (NEW)
        "Mă fac fericit când aud muzică.",     # reflexive "become" (NEW)
        "Sunt furios pe situația actuală.",
        "Mi-e frică de câini.",
        "Îmi este teamă.",
    ]

    print(f"\nTesting {len(test_cases)} sentences:\n")
    for text in test_cases:
        matches = matcher.find_matches(text)
        if matches:
            m = matches[0]
            print(f"  [{m.pattern_name}] {text}")
            print(f"    seed={m.seed_word}, emotions={m.emotions}")
        else:
            print(f"  [NO MATCH] {text}")
        print()
