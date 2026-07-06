"""Canonical basic-emotion (Plutchik-8) lexemes for MASIVE English & Spanish.

Analogous to pipeline/viz_hidden/canonical.py (Romanian), but the surface forms
are the MASIVE gold affective-state words. Selection is adjectives-only (uniform
"I feel [adj]" frame -> no noun/adjective POS confound) and several classical
synonyms per emotion so each ring vertex is an EMOTION, not a single lexeme.

Only forms that actually occur in MASIVE native train+val+test are included, and
each form maps to exactly one emotion (asserted below). Spanish keeps m./f. forms
(matched on an accent-stripped, lowercased normalization).

Coverage (matched label-slots, native train+val+test):
  EN  joy 1903  trust 721  fear 1793  surprise 412  sadness 2185
      disgust 408  anger 1974  anticipation 1658
  ES  joy  781  trust 998  fear  351  surprise 176  sadness  881
      disgust 140  anger  598  anticipation  804

Caveats to keep in mind when reading the ring:
  * "anxious"/"ansioso" straddle anticipation<->fear (apprehension); "excited"/
    "emocionado" straddle anticipation<->joy. We file them under anticipation
    (Plutchik), so if that vertex drifts toward fear/joy it is likely lexical.
  * ES "seguro" is trust here but also means "sure/confident".
  * ES "asqueroso" is self-disgust here but literally means "disgusting".
"""
from __future__ import annotations

import unicodedata

# Plutchik wheel order (adjacent emotions are adjacent on the wheel).
WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
# Plutchik-standard valence sign (matches Romanian canonical.py for comparability).
VALENCE = {"joy": 1, "trust": 1, "anticipation": 1, "surprise": 0,
           "fear": -1, "sadness": -1, "disgust": -1, "anger": -1}

# plutchik -> {lemma: [normalized surface forms]}
CANONICAL_EN: dict[str, dict[str, list[str]]] = {
    "joy":          {"happy": ["happy"], "glad": ["glad"], "joyful": ["joyful"],
                     "cheerful": ["cheerful"], "elated": ["elated"], "ecstatic": ["ecstatic"],
                     "thrilled": ["thrilled"], "overjoyed": ["overjoyed"]},
    "trust":        {"safe": ["safe"], "supported": ["supported"], "protected": ["protected"],
                     "secure": ["secure"], "accepted": ["accepted"]},
    "fear":         {"scared": ["scared"], "afraid": ["afraid"], "frightened": ["frightened"],
                     "terrified": ["terrified"], "fearful": ["fearful"]},
    "surprise":     {"surprised": ["surprised"], "shocked": ["shocked"], "amazed": ["amazed"],
                     "stunned": ["stunned"]},
    "sadness":      {"sad": ["sad"], "depressed": ["depressed"], "miserable": ["miserable"],
                     "heartbroken": ["heartbroken"], "unhappy": ["unhappy"]},
    "disgust":      {"disgusted": ["disgusted"], "grossed": ["grossed"],
                     "nauseated": ["nauseated"], "repulsed": ["repulsed"]},
    "anger":        {"angry": ["angry"], "furious": ["furious"], "annoyed": ["annoyed"],
                     "irritated": ["irritated"], "resentful": ["resentful"], "mad": ["mad"],
                     "livid": ["livid"], "enraged": ["enraged"]},
    "anticipation": {"anxious": ["anxious"], "excited": ["excited"], "hopeful": ["hopeful"],
                     "eager": ["eager"], "curious": ["curious"]},
}

# plutchik -> {lemma: [normalized (accent-stripped) m./f. forms]}
CANONICAL_ES: dict[str, dict[str, list[str]]] = {
    "joy":          {"feliz": ["feliz"], "contento": ["contento", "contenta"], "alegre": ["alegre"]},
    "trust":        {"seguro": ["seguro", "segura"], "confiado": ["confiado", "confiada"],
                     "apoyado": ["apoyado"], "aceptado": ["aceptado"]},
    "fear":         {"asustado": ["asustado", "asustada"], "aterrado": ["aterrado", "aterrada"],
                     "temeroso": ["temeroso"]},
    "surprise":     {"sorprendido": ["sorprendido", "sorprendida"], "estupefacto": ["estupefacto"],
                     "impactado": ["impactado"]},
    "sadness":      {"triste": ["triste"], "deprimido": ["deprimido", "deprimida"],
                     "infeliz": ["infeliz"], "desanimado": ["desanimado"]},
    "disgust":      {"asqueado": ["asqueado", "asqueada"], "asqueroso": ["asqueroso"]},
    "anger":        {"enojado": ["enojado", "enojada"], "enfadado": ["enfadado", "enfadada"],
                     "furioso": ["furioso", "furiosa"], "irritado": ["irritado"],
                     "molesto": ["molesto"], "cabreado": ["cabreado"], "indignado": ["indignado"]},
    "anticipation": {"ansioso": ["ansioso", "ansiosa"], "emocionado": ["emocionado", "emocionada"]},
}

TABLES = {"en": CANONICAL_EN, "es": CANONICAL_ES}


def norm(s: str) -> str:
    """Lowercase, strip accents, strip surrounding punctuation (matching key)."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip(".,;:!?\"'()[]{}«»¿¡…")


def form2emo(lang: str) -> dict[str, tuple[str, str]]:
    """Normalized form -> (plutchik, lemma). Asserts every form is unambiguous."""
    table = TABLES[lang]
    m: dict[str, tuple[str, str]] = {}
    for p, lemmas in table.items():
        for lem, forms in lemmas.items():
            for f in forms:
                key = norm(f)
                assert key not in m, f"form {key!r} maps to both {m.get(key)} and {(p, lem)}"
                m[key] = (p, lem)
    return m
