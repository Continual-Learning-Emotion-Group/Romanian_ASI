"""
Minimal stoplist and gender inference for Romanian ASI pipeline.

The stoplist contains ONLY closed-class words that are certainly not affective
states: pronouns, conjunctions, prepositions, articles, auxiliary verbs, etc.

No adjectives, nouns, or adverbs that could conceivably be affective states are
included — the data and downstream validation decide what is emotional.
"""

from typing import Optional

from .text_utils import normalize_text

# ---------------------------------------------------------------------------
# Stopwords — closed-class words only
# ---------------------------------------------------------------------------

STOPWORDS = {
    # === Pronouns ===
    "eu", "tu", "el", "ea", "noi", "voi", "ei", "ele",
    "mine", "tine", "sine", "mie", "tie", "sie",
    "meu", "mea", "mei", "mele",
    "tau", "ta", "tai", "tale",
    "sau", "sa", "sai", "sale",
    "lor", "nostru", "noastra", "vostru", "voastra",
    "se", "ma", "te", "ne", "va",
    "imi", "iti", "ii", "ni", "vi", "le",
    # Relative / interrogative
    "care", "ce", "cine", "unde", "cand", "cum", "cat", "cate", "cati",

    # === Articles / determiners ===
    "un", "una", "niste",
    "al", "ai", "ale",
    "cel", "cea", "cei", "cele",
    "acest", "aceasta", "acesta", "aceste", "acesti",
    "acel", "acea", "acei", "acele", "acela", "aceea",
    "fiecare", "oricare", "orice", "niciun", "nicio",
    "tot", "toata", "toti", "toate",
    "alt", "alta", "alti", "alte", "altul", "altceva",

    # === Prepositions ===
    "la", "de", "pe", "cu", "in", "din", "prin", "pentru",
    "fara", "despre", "peste", "intre", "pana", "dupa",
    "spre", "sub", "langa", "contra",

    # === Conjunctions / particles ===
    "si", "dar", "sau", "ori", "nici", "ca", "daca",
    "insa", "deci", "asadar", "totusi",
    "ba", "chiar", "macar", "oare",
    "nu", "da",

    # === Auxiliary / modal verbs ===
    "am", "ai", "are", "avem", "aveti", "au",
    "sunt", "esti", "este", "suntem", "sunteti",
    "era", "eram", "erai", "erau",
    "fost",
    "fi", "fiu", "fii", "fie", "fim", "fiti",
    "pot", "poti", "poate", "putem", "puteti", "putea",
    "vrea", "vreau", "vrei", "vrem", "vreti", "vor",
    "trebuie", "trebui",

    # === Common verbs (not states) ===
    "face", "fac", "faci", "facem", "faceti", "facut",
    "zice", "zis", "spune", "spus",
    "stiu", "stie", "stim", "stiti",
    "cred", "crede", "credem",
    "avea", "aveau",
    "vin", "vine", "veni",

    # === Adverbs of time / place / manner (not affective) ===
    "acum", "apoi", "atunci", "ieri", "azi", "maine",
    "aici", "acolo", "unde",
    "doar", "deja", "inca",
    "mai", "foarte", "prea", "cam", "destul", "tocmai",
    "asa", "probabil", "poate",

    # === Numerals ===
    "doi", "doua", "trei", "patru", "cinci",

    # === Other closed-class ===
    "nimic", "nimeni", "nicaieri", "ceva",
    "asta", "aia",
}

# Normalize all stopwords
STOPWORDS = {normalize_text(w) for w in STOPWORDS}

# ---------------------------------------------------------------------------
# Gender inference for Romanian adjectives
# ---------------------------------------------------------------------------

FEMININE_SUFFIXES = [
    "ă", "oasă", "ească", "ică", "ită", "ată", "ută",
]

MASCULINE_ENDINGS = [
    "os", "esc", "ic", "it", "at", "ut", "iu",
]


def infer_gender(word: str) -> Optional[str]:
    """
    Infer gender from Romanian adjective morphology.

    Returns "f" for feminine, "m" for masculine, None if uncertain.
    """
    w = normalize_text(word)

    # Check feminine first (more distinctive endings)
    for suffix in FEMININE_SUFFIXES:
        norm_suffix = normalize_text(suffix)
        if w.endswith(norm_suffix):
            return "f"

    # Check masculine endings
    for ending in MASCULINE_ENDINGS:
        if w.endswith(ending):
            return "m"

    # Consonant-final is typically masculine in Romanian
    if w and w[-1] not in "aeiou":
        return "m"

    return None
