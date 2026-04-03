"""
Lemma → inflected forms expansion using MULTEXT-East Romanian lexicon.

Expands seed lemmas (masculine singular adjectives, base nouns, adverbs) to all
gender/number/diacritics variants for robust pattern matching.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .text_utils import remove_diacritics

MULTEXT_EAST_PATH = Path(__file__).parent.parent / "seed" / "multext-east" / "wfl-ro.txt"

# MSD tag prefixes for each POS
# Af = adjective (full form), Nc = common noun, Rg = adverb (general)
POS_TAG_PREFIX = {
    "adjective": "Af",
    "noun": "Nc",
    "adverb": "Rg",
}


def load_multext_east(path: Path = None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load MULTEXT-East lexicon as lemma → [(form, MSD_tag), ...].

    File format: tab-separated (form, lemma, MSD_tag).
    Returns dict keyed by lemma.
    """
    if path is None:
        path = MULTEXT_EAST_PATH

    lemma_forms: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            form, lemma, msd = parts[0], parts[1], parts[2]
            lemma_forms[lemma].append((form, msd))

    return dict(lemma_forms)


# Module-level cache
_multext_cache: Optional[Dict[str, List[Tuple[str, str]]]] = None


def get_multext_east(path: Path = None) -> Dict[str, List[Tuple[str, str]]]:
    """Get cached MULTEXT-East lookup table."""
    global _multext_cache
    if _multext_cache is None:
        _multext_cache = load_multext_east(path)
    return _multext_cache


def _is_valid_adjective_form(msd: str) -> bool:
    """
    Check if an adjective MSD tag represents a form usable in "I feel" patterns.

    We only need singular, no clitic (no definite article), not oblique case.
    MSD format: Afp{gender}{number}{case}{clitic}
    - Position 4: gender (m/f)
    - Position 5: number (s/p)
    - Position 6: case (r=nominative, o=oblique, -=unspecified, v=vocative)
    - Position 7 (last): clitic (n=no, y=yes)
    """
    if len(msd) < 6:
        return False
    number = msd[5] if len(msd) > 5 else '-'
    case = msd[6] if len(msd) > 6 else '-'
    clitic = msd[-1]
    return number == 's' and clitic == 'n' and case not in ('o', 'v')


def _is_valid_noun_form(msd: str) -> bool:
    """
    Check if a noun MSD tag represents a base form (singular, no article).

    MSD format: Nc{gender}{number}{case}{clitic}
    """
    if len(msd) < 5:
        return False
    number = msd[4] if len(msd) > 4 else '-'
    clitic = msd[-1]
    return number == 's' and clitic == 'n'


def expand_lemma(
    lemma: str,
    pos: str,
    multext: Dict[str, List[Tuple[str, str]]] = None,
) -> Set[str]:
    """
    Expand a lemma to forms usable in first-person singular "I feel" patterns.

    For adjectives: masculine + feminine singular (no articles, no plural,
    no oblique/vocative). E.g., fericit → {fericit, fericită}.

    For nouns: singular base form only.

    For adverbs: just the lemma (adverbs don't inflect).

    All forms also get diacritic-stripped variants added.
    """
    if multext is None:
        multext = get_multext_east()

    tag_prefix = POS_TAG_PREFIX.get(pos, "")
    forms = set()

    # Look up in MULTEXT-East
    entries = multext.get(lemma, [])
    for form, msd in entries:
        if tag_prefix and not msd.startswith(tag_prefix):
            continue
        # Filter to forms that make sense in "I feel [X]" patterns
        if pos == "adjective" and not _is_valid_adjective_form(msd):
            continue
        if pos == "noun" and not _is_valid_noun_form(msd):
            continue
        forms.add(form)

    # Always include the lemma itself
    forms.add(lemma)

    # If not found in MULTEXT-East, try simple suffix rules for adjectives
    if len(forms) <= 1 and pos == "adjective":
        forms.update(_fallback_adjective_forms(lemma))

    # Generate diacritic-stripped variants
    diacritic_variants = set()
    for form in forms:
        stripped = remove_diacritics(form)
        if stripped != form:
            diacritic_variants.add(stripped)

    forms.update(diacritic_variants)

    return forms


def _fallback_adjective_forms(lemma: str) -> Set[str]:
    """
    Simple suffix rules for Romanian adjective feminine singular.

    Used when a lemma is not found in MULTEXT-East.
    Only generates masculine + feminine singular (for "I feel" patterns).
    """
    forms = {lemma}

    # Common masculine → feminine singular patterns
    if lemma.endswith("it"):
        forms.add(lemma[:-2] + "ită")
    elif lemma.endswith("at"):
        forms.add(lemma[:-2] + "ată")
    elif lemma.endswith("ut"):
        forms.add(lemma[:-2] + "ută")
    elif lemma.endswith("os"):
        forms.add(lemma[:-2] + "oasă")
    elif lemma.endswith("nic"):
        forms.add(lemma[:-2] + "ică")
    elif lemma.endswith("t"):
        forms.add(lemma[:-1] + "tă")

    return forms


def expand_seed(
    seed: Dict[str, dict],
    pos: str,
    multext: Dict[str, List[Tuple[str, str]]] = None,
) -> Dict[str, Set[str]]:
    """
    Expand all lemmas in a seed dictionary to their inflected forms.

    Args:
        seed: Dict mapping lemma → metadata (e.g., affect category).
        pos: Part of speech for all lemmas in this dict.
        multext: MULTEXT-East lookup table.

    Returns:
        Dict mapping lemma → set of all forms (including diacritic variants).
    """
    if multext is None:
        multext = get_multext_east()

    expanded = {}
    for lemma in seed:
        expanded[lemma] = expand_lemma(lemma, pos, multext)

    return expanded


def expand_full_seed(
    adjectives: Dict[str, str],
    nouns: Dict[str, str],
    adverbs: Dict[str, str],
    multext: Dict[str, List[Tuple[str, str]]] = None,
) -> Dict[str, str]:
    """
    Expand a full seed (adjectives + nouns + adverbs) to all forms.

    Returns a flat dict mapping every inflected form → affect category
    of the original lemma.
    """
    if multext is None:
        multext = get_multext_east()

    form_to_categ = {}

    for lemma, categ in adjectives.items():
        for form in expand_lemma(lemma, "adjective", multext):
            form_to_categ[form] = categ

    for lemma, categ in nouns.items():
        for form in expand_lemma(lemma, "noun", multext):
            form_to_categ[form] = categ

    for lemma, categ in adverbs.items():
        for form in expand_lemma(lemma, "adverb", multext):
            form_to_categ[form] = categ

    return form_to_categ
