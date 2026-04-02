#!/usr/bin/env python3
"""
Romanian affective state seed list for ASI benchmark.

Built from RoEmoLex V3 filtered through WordNet-Affect 1.1 via UPC/TALP
WN 1.6→3.0 offset mapping, then manually reviewed for quality.

Each word passes two criteria:
  1. Pattern fit: works in "mă simt [adj]", "mi-e [noun]", "simt [noun]", etc.
  2. Affective state: describes how someone FEELS internally (emotion, mood,
     bodily-felt state, psychological state).

Words are stored as LEMMAS only (masculine singular for adjectives).
Gender/number inflection and diacritics normalization should be handled
downstream by the pattern matcher using MULTEXT-East or similar.

Source chain: RoEmoLex V3 → WordNet-Affect 1.1 → manual review
  - Bridge script: pipeline/test_wn_affect_bridge.py
  - Bridge results: pipeline/wn_affect_bridge_results.json
  - Quality review: manual pass over all 398 bridge words + 200 fails/questionable
"""

# Affective state ADJECTIVES
# Used with: "mă simt [X]", "sunt [X]", "m-am simțit [X]"
ADJECTIVES = {
    # --- From bridge passes (WordNet-Affect validated) ---

    # Joy / Happiness
    "fericit": "happiness",
    "bucuros": "gladness",
    "mulțumit": "contentment",
    "satisfăcut": "contentment",
    "radios": "gladness",
    "euforic": "happiness",
    "jovial": "jollity",
    "vesel": "cheerfulness",
    "voios": "cheerfulness",
    "exultant": "triumph",
    "jubilant": "triumph",

    # Sadness / Gloom
    "trist": "sadness",
    "nefericit": "dysphoria",
    "deprimat": "gloom",
    "abătut": "gloom",
    "amărât": "distress",
    "mohorât": "gloom",
    "mâhnit": "dysphoria",
    "nenorocit": "misery",
    "posac": "gloom",
    "sumbru": "gloom",          # salvaged from questionable
    "demoralizat": "discouragement",
    "descurajat": "discouragement",
    "necăjit": "distress",
    "îndurerat": "lost-sorrow",
    "întristat": "sadness",     # salvaged: întristător -> întristat
    "împovărat": "oppression",

    # Fear / Anxiety
    "înfricoșat": "panic",
    "îngrozit": "panic",
    "înspăimântat": "panic",    # salvaged: înspăimântător -> înspăimântat
    "panicat": "panic",
    "temător": "negative-fear",
    "anxios": "anxiousness",
    "nervos": "jitteriness",
    "agitat": "fever",
    "înfiorat": "chill",        # salvaged: înfiorător -> înfiorat
    "cutremurat": "scare",      # salvaged: cutremurător -> cutremurat

    # Anger / Irritation
    "furios": "anger",
    "mânios": "anger",
    "înfuriat": "infuriation",
    "enervat": "annoyance",
    "iritat": "annoyance",
    "agasat": "annoyance",
    "supărat": "distress",
    "frustrat": "defeatism",
    "sâcâit": "annoyance",     # salvaged: sâcâitor -> sâcâit
    "iritabil": "jitteriness",  # salvaged from questionable
    "ostil": "hostility",

    # Disgust
    "dezgustat": "disgust",
    "scârbit": "disgust",
    "grețos": "disgust",
    "îngrețoșat": "nausea",
    "sătul": "disgust",         # salvaged from questionable (fed up)

    # Surprise / Astonishment
    "surprins": "stupefaction",
    "uimit": "stupefaction",
    "uluit": "stupefaction",
    "consternat": "stupefaction",
    "stupefiat": "stupefaction",
    "perplex": "stupefaction",

    # Trust / Calm / Security
    "liniștit": "tranquillity",
    "calm": "tranquillity",
    "cumpănit": "tranquillity",
    "stăpânit": "tranquillity",
    "încurajat": "encouragement",  # salvaged: încurajator -> încurajat

    # Jealousy / Envy
    "gelos": "jealousy",
    "invidios": "jealousy",

    # Shame / Embarrassment
    "rușinos": "shame",
    "jenat": "embarrassment",
    "stânjenit": "embarrassment",
    "stingherit": "embarrassment",

    # Other affective states
    "recunoscător": "gratitude",
    "afectuos": "tenderness",
    "tandru": "tenderness",
    "apatic": "indifference",
    "indiferent": "indifference",  # salvaged from questionable
    "alienat": "alienation",       # salvaged from questionable
    "pasionat": "electricity",     # salvaged: pasionant -> pasionat
    "fermecat": "captivation",     # salvaged: fermecător -> fermecat
    "triumfător": "triumph",       # salvaged from fails
    "deranjat": "distress",        # salvaged: deranjant -> deranjat
    "alinat": "calmness",          # salvaged: alinător -> alinat
    "isteric": "hysteria",
    "febril": "fever",
    "infatuat": "smugness",
    "încrezut": "smugness",
    "îngâmfat": "smugness",
    "înnebunit": "huffiness",
    "doritor": "impatience",
    "nerăbdător": "impatience",
    "sfios": "timidity",
    "timid": "timidity",
    "dezorientat": "alienation",
}

# Affective state NOUNS
# Used with: "mi-e [X]", "am [X]", "simt [X]", "îmi este [X]"
NOUNS = {
    # --- From bridge passes (WordNet-Affect validated) ---

    # Joy / Happiness
    "bucurie": "joy",
    "veselie": "cheerfulness",
    "voioșie": "cheerfulness",
    "încântare": "gladness",
    "entuziasm": "gusto",
    "elan": "gusto",            # salvaged from noun fails
    "vervă": "gusto",           # salvaged from noun fails
    "exultanță": "exultation",
    "exultare": "exultation",
    "exultație": "exultation",
    "jubilare": "exultation",
    "jubilație": "exultation",
    "satisfacție": "satisfaction-pride",
    "mândrie": "satisfaction-pride",
    "împlinire": "fulfillment", # salvaged from noun fails
    "ilaritate": "hilarity",

    # Sadness / Grief
    "tristețe": "gloom",
    "amărăciune": "gloom",
    "mâhnire": "gloom",
    "durere": "dolor",
    "suferință": "harassment",
    "deznădejde": "despondency",
    "disperare": "despondency",
    "desperare": "despondency",
    "depresie": "blue-devils",
    "deprimare": "despondency",
    "demoralizare": "despondency",
    "descurajare": "despondency",
    "dezolare": "forlornness",
    "doliu": "grief",           # salvaged from noun fails
    "văicăreală": "self-pity",
    "înlăcrimare": "weepiness",

    # Fear / Anxiety
    "frică": "negative-fear",
    "teamă": "negative-fear",
    "spaimă": "panic",
    "groază": "panic",
    "teroare": "panic",
    "panică": "panic",
    "anxietate": "anxiety",
    "neliniște": "anxiety",
    "frământare": "anxiety",
    "agitație": "anxiety",      # salvaged from noun questionable
    "trepidație": "trepidation",
    "îngrijorare": "trepidation",
    "ezitare": "hesitance",
    "îndoială": "hesitance",
    "șovăială": "hesitance",
    "fior": "chill",            # salvaged from noun questionable
    "înfiorare": "chill",       # salvaged from noun questionable

    # Anger / Hostility
    "furie": "anger",
    "mânie": "anger",
    "enervare": "annoyance",
    "iritare": "annoyance",
    "iritație": "annoyance",
    "agasare": "annoyance",
    "exasperare": "aggravation",
    "frustrare": "frustration",
    "frustrație": "frustration",
    "mâniere": "infuriation",
    "înfuriere": "infuriation",
    "indignare": "indignation",
    "revoltă": "indignation",
    "necaz": "resentment",
    "ciudă": "resentment",
    "pică": "resentment",
    "ranchiună": "resentment",
    "resentiment": "resentment",
    "animozitate": "animosity",
    "antipatie": "antipathy",
    "dușmănie": "hate",
    "ură": "hate",
    "agresivitate": "aggression",  # salvaged from noun fails
    "pizmă": "envy",
    "gelozie": "jealousy",
    "invidie": "envy",

    # Disgust
    "dezgust": "disgust",
    "scârbă": "disgust",
    "silă": "disgust",
    "repulsie": "repugnance",
    "oroare": "repugnance",
    "aversiune": "antipathy",

    # Surprise / Astonishment
    "surprindere": "surprise",
    "uimire": "astonishment",
    "uluială": "astonishment",
    "uluire": "astonishment",
    "minunare": "wonder",
    "consternare": "stupefaction",
    "stupefacție": "stupefaction",
    "stupefiere": "astonishment",
    "stupoare": "stupefaction",

    # Trust / Calm / Peace
    "calm": "peace",
    "seninătate": "peace",
    "serenitate": "peace",
    "speranță": "positive-hope",
    "nădejde": "positive-hope",
    "optimism": "optimism",
    "recunoștință": "gratitude",
    "gratitudine": "gratitude",

    # Love / Affection
    "dragoste": "love",
    "iubire": "love",
    "afecțiune": "affection",
    "atașament": "attachment",
    "devotament": "devotion",
    "devoțiune": "devotion",
    "empatie": "empathy",

    # Shame / Embarrassment
    "jenă": "embarrassment",
    "sfială": "embarrassment",
    "stinghereală": "embarrassment",
    "stânjeneală": "embarrassment",
    "stânjenire": "embarrassment",
    "sfiiciune": "shyness",
    "timiditate": "shyness",
    "rușine": "shame",         # not in bridge but obvious

    # Guilt / Remorse
    "regret": "compunction",
    "remușcare": "guilt",
    "căință": "compunction",
    "pocăință": "repentance",
    "mustrare": "compunction",  # salvaged from noun fails

    # Other
    "chin": "harassment",
    "pesimism": "pessimism",
    "nerăbdare": "eagerness",
    "fâstâcire": "confusion",
    "zăpăcire": "confusion",
    "înstrăinare": "alienation",
    "alienare": "alienation",   # salvaged from noun questionable
    "izolare": "isolation",     # salvaged from noun questionable
    "apăsare": "weight",       # salvaged from noun questionable
    "detașare": "withdrawal",  # salvaged from noun questionable
    "exuberanță": "exuberance", # salvaged from noun questionable
    "patimă": "harassment",    # salvaged from noun fails
    "isterie": "hysteria",     # salvaged from noun fails
}

# Affective state ADVERBS
# Used with: "mă simt [X]"
ADVERBS = {
    "trist": "cheerlessness",
    "mohorât": "cheerlessness",
    "posomorât": "cheerlessness",
    "mâhnit": "attrition",
    "îndurerat": "attrition",
    "indignat": "indignation",
    "isteric": "hysteria",
    "resemnat": "resignation",
    "sfios": "timidity",
    "timid": "timidity",
    "spășit": "repentance",
    "drăgăstos": "love",
    "iubitor": "love",
    "dureros": "regret-sorrow",  # salvaged from adverb questionable
}


def build_seed():
    """Build the seed dictionary."""
    word_to_categ = {}
    word_to_categ.update(ADJECTIVES)
    word_to_categ.update(NOUNS)
    word_to_categ.update(ADVERBS)

    return {
        "source": "roemolex_wn_affect_bridge",
        "word_to_affect_categ": word_to_categ,
        "all_words": sorted(word_to_categ.keys()),
        "statistics": {
            "total_adjectives": len(ADJECTIVES),
            "total_nouns": len(NOUNS),
            "total_adverbs": len(ADVERBS),
            "total_words": len(word_to_categ),
        },
        "adjectives": list(ADJECTIVES.keys()),
        "nouns": list(NOUNS.keys()),
        "adverbs": list(ADVERBS.keys()),
    }


def save_seed(output_path=None):
    """Save seed to JSON."""
    import json
    from pathlib import Path

    if output_path is None:
        output_path = Path(__file__).parent / "emotion_seed.json"

    seed = build_seed()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)

    print(f"Saved seed to {output_path}")
    print(f"  Adjectives: {seed['statistics']['total_adjectives']}")
    print(f"  Nouns: {seed['statistics']['total_nouns']}")
    print(f"  Adverbs: {seed['statistics']['total_adverbs']}")
    print(f"  Total: {seed['statistics']['total_words']}")

    return seed


if __name__ == "__main__":
    save_seed()
