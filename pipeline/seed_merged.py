#!/usr/bin/env python3
"""
Romanian affective state seed list for ASI benchmark (merged).

375 lemmas: 193 adjectives, 165 nouns, 27 adverbs.

Merged from two sources:

  Source 1: RoEmoLex V3 × WordNet-Affect bridge (229 words, 61%)
    RoEmoLex V3 (~9K Romanian words with WordNet 3.0 synset IDs) was filtered
    through WordNet-Affect 1.1 (~798 synsets labeled as genuinely affective).
    The bridge uses UPC/TALP offset mapping files to convert WN 1.6 → 3.0
    synset IDs, then joins: if a RoEmoLex word's synset appears in
    WordNet-Affect, it describes a felt state (not just an emotion-associated
    word like "accident" or "concert").

    The raw bridge output (398 words) was then manually reviewed:
      - 199 passed as genuine affective states
      - 156 failed (causative adjectives, external qualities, events, traits)
      - 43 questionable
      - 33 words salvaged from fails by converting causative forms to
        felt-state participles (e.g., enervant → enervat, fascinant → fascinat)

    See: pipeline/test_wn_affect_bridge.py, pipeline/wn_affect_bridge_results.json

  Source 2: Old curated seed, base lemmas only (148 words, 39%)
    The previous hand-curated seed (scripts/ro_asi/curated_affective_states.py,
    511 entries) was deduplicated to base lemmas (removing feminine forms like
    fericită and diacritics-free variants like frica), then filtered to remove
    words that failed quality review:
      - Removed: sigur (epistemic), voie/chef (permission/desire), violent/
        agresiv/beligerant (behavioral), confident (false friend), etc.
      - Kept: common affective words the bridge missed because they weren't
        in RoEmoLex with valid synset IDs (e.g., obosit, stresat, dezamăgit,
        șocat, relaxat, speriat, plictisit, confuz).

Quality criteria (both sources):
  1. Pattern fit: works grammatically in Romanian "I feel" patterns
     - Adjectives: "mă simt [X]", "sunt [X]", "m-am simțit [X]"
     - Nouns: "mi-e [X]", "am [X]", "simt [X]"
     - Adverbs: "mă simt [X]"
  2. Affective state: describes how someone FEELS internally
     - Includes: emotions, moods, bodily-felt states, psychological states
     - Excludes: epistemic states, personality traits, external qualities,
       causative adjectives, social roles, events/situations

Words are stored as LEMMAS only (masculine singular for adjectives).
Gender/number inflection and diacritics normalization should be handled
downstream by the pattern matcher using MULTEXT-East (pipeline/multext-east/)
or similar.

Each word maps to a WordNet-Affect category (e.g., "happiness", "gloom",
"panic") rather than Plutchik emotion labels. Words from the old seed that
have no WN-Affect category use their original Plutchik label instead.
"""

# Affective state ADJECTIVES
# Used with: "mă simt [X]", "sunt [X]", "m-am simțit [X]"
ADJECTIVES = {
    # --- Joy / Happiness ---
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
    "binedispus": "joy",             # old seed
    "amuzat": "joy",                 # old seed
    "încântat": "joy",               # old seed
    "entuziasmat": "joy",            # old seed
    "excitat": "joy",                # old seed
    "extaziat": "joy",               # old seed
    "exuberant": "joy",              # old seed
    "mândru": "joy",                 # old seed
    "împlinit": "joy",               # old seed
    "înviorat": "joy",               # old seed
    "captivat": "joy",               # old seed
    "fascinat": "joy",               # old seed
    "optimist": "joy",               # old seed

    # --- Sadness / Gloom ---
    "trist": "sadness",
    "nefericit": "dysphoria",
    "deprimat": "gloom",
    "abătut": "gloom",
    "amărât": "distress",
    "mohorât": "gloom",
    "mâhnit": "dysphoria",
    "nenorocit": "misery",
    "posac": "gloom",
    "sumbru": "gloom",               # bridge salvaged
    "demoralizat": "discouragement",
    "descurajat": "discouragement",
    "necăjit": "distress",
    "îndurerat": "lost-sorrow",
    "întristat": "sadness",          # bridge salvaged
    "împovărat": "oppression",
    "devastat": "sadness",           # old seed
    "dezamăgit": "sadness",          # old seed
    "disperat": "sadness",           # old seed
    "distrus": "sadness",            # old seed
    "epuizat": "sadness",            # old seed
    "istovit": "sadness",            # old seed
    "melancolic": "sadness",         # old seed
    "neconsolat": "sadness",         # old seed
    "nostalgic": "sadness",          # old seed
    "năpăstuit": "sadness",          # old seed
    "obosit": "sadness",             # old seed
    "pesimist": "sadness",           # old seed
    "plictisit": "sadness",          # old seed
    "posomorât": "sadness",          # old seed
    "pustiu": "sadness",             # old seed
    "rănit": "sadness",              # old seed
    "singur": "sadness",             # old seed
    "sleit": "sadness",              # old seed
    "zdrobit": "sadness",            # old seed
    "însingurat": "sadness",         # old seed
    "gol": "sadness",                # old seed (metaphorical emptiness)

    # --- Fear / Anxiety ---
    "înfricoșat": "panic",
    "îngrozit": "panic",
    "înspăimântat": "panic",         # bridge salvaged
    "panicat": "panic",
    "temător": "negative-fear",
    "anxios": "anxiousness",
    "nervos": "jitteriness",
    "agitat": "fever",
    "înfiorat": "chill",             # bridge salvaged
    "cutremurat": "scare",           # bridge salvaged
    "alarmat": "fear",               # old seed
    "speriat": "fear",               # old seed
    "stresat": "fear",               # old seed
    "neliniștit": "fear",            # old seed
    "îngrijorat": "fear",            # old seed
    "tensionat": "fear",             # old seed
    "încordat": "fear",              # old seed
    "nesigur": "fear",               # old seed
    "confuz": "fear",                # old seed
    "tulburat": "fear",              # old seed
    "perturbat": "fear",             # old seed
    "terorizat": "fear",             # old seed
    "traumatizat": "fear",           # old seed
    "inhibat": "fear",               # old seed
    "intimidat": "fear",             # old seed
    "fricos": "fear",                # old seed
    "suspicios": "fear",             # old seed
    "precaut": "fear",               # old seed
    "circumspect": "fear",           # old seed
    "paranoid": "fear",              # old seed
    "vulnerabil": "fear",            # old seed
    "fragil": "fear",                # old seed
    "neajutorat": "fear",            # old seed

    # --- Anger / Irritation ---
    "furios": "anger",
    "mânios": "anger",
    "înfuriat": "infuriation",
    "enervat": "annoyance",
    "iritat": "annoyance",
    "agasat": "annoyance",
    "supărat": "distress",
    "frustrat": "defeatism",
    "sâcâit": "annoyance",          # bridge salvaged
    "iritabil": "jitteriness",       # bridge salvaged
    "ostil": "hostility",
    "exasperat": "anger",            # old seed
    "indignat": "anger",             # old seed
    "revoltat": "anger",             # old seed
    "ofensat": "anger",              # old seed
    "jignit": "anger",               # old seed
    "nemulțumit": "anger",           # old seed
    "bosumflat": "anger",            # old seed
    "morocănos": "anger",            # old seed
    "ursuz": "anger",                # old seed
    "turbat": "anger",               # old seed
    "ranchiunos": "anger",           # old seed
    "vinovat": "sadness",            # old seed

    # --- Disgust ---
    "dezgustat": "disgust",
    "scârbit": "disgust",
    "grețos": "disgust",
    "îngrețoșat": "nausea",
    "sătul": "disgust",              # bridge salvaged (fed up)
    "oripilat": "disgust",           # old seed
    "astomacat": "disgust",          # old seed
    "respins": "disgust",            # old seed

    # --- Surprise / Astonishment ---
    "surprins": "stupefaction",
    "uimit": "stupefaction",
    "uluit": "stupefaction",
    "consternat": "stupefaction",
    "stupefiat": "stupefaction",
    "perplex": "stupefaction",
    "șocat": "surprise",             # old seed
    "năucit": "surprise",            # old seed
    "buimăcit": "surprise",          # old seed
    "amețit": "surprise",            # old seed
    "zăpăcit": "surprise",           # old seed
    "descumpănit": "surprise",       # old seed
    "nedumerit": "surprise",         # old seed
    "mirat": "surprise",             # old seed
    "impresionat": "surprise",       # old seed
    "copleșit": "surprise",          # old seed
    "contrariat": "surprise",        # old seed

    # --- Trust / Calm / Security ---
    "liniștit": "tranquillity",
    "calm": "tranquillity",
    "cumpănit": "tranquillity",
    "stăpânit": "tranquillity",
    "încurajat": "encouragement",    # bridge salvaged
    "relaxat": "trust",              # old seed
    "senin": "trust",                # old seed
    "echilibrat": "trust",           # old seed
    "împăcat": "trust",              # old seed
    "încrezător": "trust",           # old seed
    "confortabil": "trust",          # old seed

    # --- Shame / Embarrassment ---
    "rușinos": "shame",
    "jenat": "embarrassment",
    "stânjenit": "embarrassment",
    "stingherit": "embarrassment",
    "rușinat": "shame",              # old seed
    "umilit": "shame",               # old seed

    # --- Love / Tenderness ---
    "afectuos": "tenderness",
    "tandru": "tenderness",
    "iubit": "trust",                # old seed
    "apreciat": "trust",             # old seed
    "onorat": "trust",               # old seed

    # --- Other affective states ---
    "recunoscător": "gratitude",
    "apatic": "indifference",
    "indiferent": "indifference",    # bridge salvaged
    "alienat": "alienation",         # bridge salvaged
    "pasionat": "electricity",       # bridge salvaged
    "fermecat": "captivation",       # bridge salvaged
    "triumfător": "triumph",         # bridge salvaged
    "deranjat": "distress",          # bridge salvaged
    "alinat": "calmness",            # bridge salvaged
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
    "emoționat": "joy",              # old seed
    "mișcat": "joy",                 # old seed
    "înduioșat": "joy",              # old seed
    "sensibil": "sadness",           # old seed
    "sentimental": "sadness",        # old seed
    "curios": "anticipation",        # old seed (known noise risk)
    "intrigat": "anticipation",      # old seed
    "dornic": "anticipation",        # old seed
    "ațâțat": "anticipation",        # old seed
}

# Affective state NOUNS
# Used with: "mi-e [X]", "am [X]", "simt [X]", "îmi este [X]"
NOUNS = {
    # --- Joy / Happiness ---
    "bucurie": "joy",
    "veselie": "cheerfulness",
    "voioșie": "cheerfulness",
    "încântare": "gladness",
    "entuziasm": "gusto",
    "elan": "gusto",                 # bridge salvaged
    "vervă": "gusto",                # bridge salvaged
    "exultanță": "exultation",
    "exultare": "exultation",
    "exultație": "exultation",
    "jubilare": "exultation",
    "jubilație": "exultation",
    "satisfacție": "satisfaction-pride",
    "mândrie": "satisfaction-pride",
    "împlinire": "fulfillment",      # bridge salvaged
    "ilaritate": "hilarity",
    "fericire": "joy",               # old seed
    "plăcere": "joy",                # old seed
    "mulțumire": "joy",              # old seed
    "euforie": "joy",                # old seed
    "extaz": "joy",                  # old seed
    "deliciu": "joy",                # old seed
    "beatitudine": "joy",            # old seed
    "ușurare": "joy",                # old seed

    # --- Sadness / Grief ---
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
    "doliu": "grief",                # bridge salvaged
    "văicăreală": "self-pity",
    "înlăcrimare": "weepiness",
    "melancolie": "sadness",         # old seed
    "nostalgie": "sadness",          # old seed
    "singurătate": "sadness",        # old seed
    "dor": "sadness",                # old seed
    "oboseală": "sadness",           # old seed
    "epuizare": "sadness",           # old seed
    "plictiseală": "sadness",        # old seed
    "apatie": "sadness",             # old seed
    "vină": "sadness",               # old seed

    # --- Fear / Anxiety ---
    "frică": "negative-fear",
    "teamă": "negative-fear",
    "spaimă": "panic",
    "groază": "panic",
    "teroare": "panic",
    "panică": "panic",
    "anxietate": "anxiety",
    "neliniște": "anxiety",
    "frământare": "anxiety",
    "agitație": "anxiety",           # bridge salvaged
    "trepidație": "trepidation",
    "îngrijorare": "trepidation",
    "ezitare": "hesitance",
    "îndoială": "hesitance",
    "șovăială": "hesitance",
    "fior": "chill",                 # bridge salvaged
    "înfiorare": "chill",            # bridge salvaged
    "siguranță": "trust",            # old seed

    # --- Anger / Hostility ---
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
    "agresivitate": "aggression",    # bridge salvaged
    "pizmă": "envy",
    "gelozie": "jealousy",
    "invidie": "envy",
    "nervi": "anger",                # old seed
    "ostilitate": "anger",           # old seed

    # --- Disgust ---
    "dezgust": "disgust",
    "scârbă": "disgust",
    "silă": "disgust",
    "repulsie": "repugnance",
    "oroare": "repugnance",
    "aversiune": "antipathy",
    "greață": "disgust",             # old seed

    # --- Surprise ---
    "surprindere": "surprise",
    "uimire": "astonishment",
    "uluială": "astonishment",
    "uluire": "astonishment",
    "minunare": "wonder",
    "consternare": "stupefaction",
    "stupefacție": "stupefaction",
    "stupefiere": "astonishment",
    "stupoare": "stupefaction",
    "surpriză": "surprise",          # old seed
    "mirare": "surprise",            # old seed
    "șoc": "surprise",               # old seed
    "perplexitate": "surprise",      # old seed

    # --- Trust / Calm / Peace ---
    "calm": "peace",
    "seninătate": "peace",
    "serenitate": "peace",
    "speranță": "positive-hope",
    "nădejde": "positive-hope",
    "optimism": "optimism",
    "recunoștință": "gratitude",
    "gratitudine": "gratitude",
    "liniște": "trust",              # old seed
    "pace": "trust",                 # old seed
    "relaxare": "trust",             # old seed
    "încredere": "trust",            # old seed

    # --- Love / Affection ---
    "dragoste": "love",
    "iubire": "love",
    "afecțiune": "affection",
    "atașament": "attachment",
    "devotament": "devotion",
    "devoțiune": "devotion",
    "empatie": "empathy",
    "compasiune": "empathy",         # old seed
    "milă": "sadness",               # old seed

    # --- Shame / Embarrassment / Guilt ---
    "jenă": "embarrassment",
    "sfială": "embarrassment",
    "stinghereală": "embarrassment",
    "stânjeneală": "embarrassment",
    "stânjenire": "embarrassment",
    "sfiiciune": "shyness",
    "timiditate": "shyness",
    "rușine": "shame",
    "regret": "compunction",
    "remușcare": "guilt",
    "căință": "compunction",
    "pocăință": "repentance",
    "mustrare": "compunction",       # bridge salvaged
    "vinovăție": "guilt",            # old seed

    # --- Other ---
    "chin": "harassment",
    "pesimism": "pessimism",
    "nerăbdare": "eagerness",
    "fâstâcire": "confusion",
    "zăpăcire": "confusion",
    "înstrăinare": "alienation",
    "alienare": "alienation",        # bridge salvaged
    "izolare": "isolation",          # bridge salvaged
    "apăsare": "weight",            # bridge salvaged
    "detașare": "withdrawal",       # bridge salvaged
    "exuberanță": "exuberance",      # bridge salvaged
    "patimă": "harassment",         # bridge salvaged
    "isterie": "hysteria",          # bridge salvaged
    "așteptare": "anticipation",     # old seed
    "curiozitate": "anticipation",   # old seed
}

# Affective state ADVERBS
# Used with: "mă simt [X]"
ADVERBS = {
    # --- From bridge ---
    "drăgăstos": "love",
    "iubitor": "love",
    "indignat": "indignation",
    "isteric": "hysteria",
    "resemnat": "resignation",
    "sfios": "timidity",
    "timid": "timidity",
    "spășit": "repentance",
    "mohorât": "cheerlessness",
    "posomorât": "cheerlessness",
    "trist": "cheerlessness",
    "mâhnit": "attrition",
    "îndurerat": "attrition",
    "dureros": "regret-sorrow",      # bridge salvaged
    # --- From old seed ---
    "bine": "joy",
    "rău": "sadness",
    "prost": "sadness",
    "nasol": "sadness",
    "groaznic": "fear",
    "oribil": "fear",
    "îngrozitor": "fear",
    "teribil": "fear",
    "minunat": "joy",
    "grozav": "joy",
    "excelent": "joy",
    "fantastic": "joy",
    "extraordinar": "joy",
}


def build_seed():
    """Build the seed dictionary."""
    word_to_categ = {}
    word_to_categ.update(ADJECTIVES)
    word_to_categ.update(NOUNS)
    word_to_categ.update(ADVERBS)

    return {
        "source": "roemolex_wn_affect_bridge_merged_with_old_curated",
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
