#!/usr/bin/env python3
"""
Test bridge between WordNet-Affect 1.1 and RoEmoLex V3.

WordNet-Affect labels ~798 WordNet 1.6 synsets as affective (emotions, moods, traits).
RoEmoLex has ~9K Romanian words linked to WordNet 3.0 synset IDs.

This script bridges them using UPC/TALP offset mapping files (WN 1.6 → 3.0).

Usage:
    python -m pipeline.test_wn_affect_bridge
"""

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
ROEMOLEX_DIR = PROJECT_ROOT / "data" / "roemolex"
WN_AFFECT_XML = PIPELINE_DIR / "wn-affect-1.1" / "a-synsets.xml"
WN_MAPPING_DIR = PIPELINE_DIR / "wn-mappings"


def load_wn16_to_30_mapping():
    """
    Load UPC/TALP WordNet 1.6 → 3.0 offset mapping files.

    File format (space-separated):
        00001740 00001740 1
        00002086 00004258 0.219 00004475 0.781

    Returns {(pos, offset_16): [(offset_30, confidence), ...]}.
    """
    pos_map = {"noun": "n", "verb": "v", "adj": "a", "adv": "r"}
    mapping = {}

    for filename, pos in pos_map.items():
        fpath = WN_MAPPING_DIR / f"wn16-30.{filename}"
        if not fpath.exists():
            print(f"  WARNING: mapping file not found: {fpath}")
            continue

        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                offset_16 = int(parts[0])
                # Remaining parts are (offset_30, confidence) pairs
                targets = []
                for i in range(1, len(parts), 2):
                    offset_30 = int(parts[i])
                    conf = float(parts[i + 1])
                    targets.append((offset_30, conf))
                mapping[(pos, offset_16)] = targets

    return mapping


def parse_wn_affect():
    """Parse WN-Affect 1.1 XML. Returns {(pos, offset_16): categ}."""
    tree = ET.parse(WN_AFFECT_XML)
    root = tree.getroot()

    # Nouns have categ directly
    noun_categs = {}
    for elem in root.iter('noun-syn'):
        categ = elem.get('categ')
        if not categ:
            continue
        offset = int(elem.get('id').split('#')[1])
        noun_categs[offset] = categ

    # All synsets: adj/verb/adv inherit categ from their noun-id
    all_synsets = {('n', off): cat for off, cat in noun_categs.items()}
    for tag, pos in [('adj-syn', 'a'), ('verb-syn', 'v'), ('adv-syn', 'r')]:
        for elem in root.iter(tag):
            offset = int(elem.get('id').split('#')[1])
            noun_ref = elem.get('noun-id')
            if noun_ref:
                noun_offset = int(noun_ref.split('#')[1])
                categ = noun_categs.get(noun_offset)
                if categ:
                    all_synsets[(pos, offset)] = categ

    return all_synsets, noun_categs


def build_wn30_affect_set(affect_16_synsets, wn_mapping):
    """
    Map WN-Affect 1.6 synsets to WN 3.0 synset IDs using offset mapping.

    Returns {ENG30_synset_id: categ}, unmapped list.
    """
    affect_30 = {}
    unmapped = []

    for (pos, offset_16), categ in affect_16_synsets.items():
        targets = wn_mapping.get((pos, offset_16))
        if targets:
            # Use the highest-confidence mapping (or all of them)
            for offset_30, conf in targets:
                key = f"ENG30-{offset_30:08d}-{pos}"
                affect_30[key] = {"categ": categ, "confidence": conf}
        else:
            unmapped.append((pos, offset_16, categ))

    return affect_30, unmapped


VALID_POS = {'Adjective', 'Noun', 'Adverb'}


def load_roemolex():
    """Load RoEmoLex V3 entries with ENG30 synset IDs, excluding verbs and expressions."""
    entries = []
    skipped = 0
    for fname in sorted(ROEMOLEX_DIR.glob("*.csv")):
        with open(fname, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                sid = row.get('wn_synset_id', '').strip()
                pos = row.get('part_of_speech', '').strip()
                if not sid.startswith('ENG30-'):
                    continue
                if pos not in VALID_POS:
                    skipped += 1
                    continue
                entries.append({
                    'word': row['word'].strip(),
                    'synset_id': sid,
                    'pos': pos,
                })
    print(f"  Skipped {skipped} entries (verbs, expressions, etc.)")
    return entries


def main():
    print("=" * 60)
    print("WordNet-Affect 1.1 ↔ RoEmoLex V3 Bridge Test")
    print("(using UPC/TALP WN 1.6 → 3.0 offset mapping)")
    print("=" * 60)

    # Load offset mapping
    print("\nLoading WN 1.6 → 3.0 mapping...")
    wn_mapping = load_wn16_to_30_mapping()
    print(f"  Loaded {len(wn_mapping)} mappings")

    # Parse WN-Affect
    all_affect_16, noun_categs = parse_wn_affect()
    pos_dist = Counter(p for p, _ in all_affect_16)
    print(f"\nWN-Affect 1.6 synsets: {len(all_affect_16)}")
    for p, c in pos_dist.most_common():
        print(f"  {p}: {c}")

    # Build WN 3.0 bridge via offset mapping
    affect_30, unmapped = build_wn30_affect_set(all_affect_16, wn_mapping)
    pos_dist2 = Counter(k.split('-')[-1] for k in affect_30)
    print(f"\nWN 3.0 affect synsets (via offset mapping): {len(affect_30)}")
    for p, c in pos_dist2.most_common():
        print(f"  {p}: {c}")
    print(f"Unmapped synsets: {len(unmapped)} / {len(all_affect_16)}")
    if unmapped:
        print(f"  Sample unmapped: {unmapped[:5]}")

    # Load RoEmoLex
    roemolex = load_roemolex()
    print(f"\nRoEmoLex entries with ENG30 IDs: {len(roemolex)}")

    # Cross-reference
    hits = []
    for entry in roemolex:
        match = affect_30.get(entry['synset_id'])
        if match:
            hits.append({
                **entry,
                'affect_categ': match['categ'],
                'mapping_confidence': match['confidence'],
            })

    unique_words = set(h['word'] for h in hits)
    unique_categs = set(h['affect_categ'] for h in hits)

    print(f"\n{'=' * 40}")
    print(f"RESULTS")
    print(f"{'=' * 40}")
    print(f"Matched entries: {len(hits)} / {len(roemolex)} ({100*len(hits)/len(roemolex):.1f}%)")
    print(f"Unique words: {len(unique_words)}")
    print(f"Affect categories: {len(unique_categs)}")

    pos_hits = Counter(h['pos'] for h in hits)
    print(f"\nBy POS:")
    for p, c in pos_hits.most_common():
        print(f"  {p}: {c}")

    # Adjective matches (most useful for "mă simt X")
    adj_hits = sorted(set((h['word'], h['affect_categ']) for h in hits if h['pos'] == 'Adjective'))
    print(f"\nAdjective matches ({len(adj_hits)}):")
    for w, c in adj_hits:
        print(f"  {w} -> {c}")

    # Noun matches (for "am X", "mi-e X")
    noun_hits = sorted(set((h['word'], h['affect_categ']) for h in hits if h['pos'] == 'Noun'))
    print(f"\nNoun matches ({len(noun_hits)}):")
    for w, c in noun_hits:
        print(f"  {w} -> {c}")

    # Adverb matches
    adv_hits = sorted(set((h['word'], h['affect_categ']) for h in hits if h['pos'] == 'Adverb'))
    print(f"\nAdverb matches ({len(adv_hits)}):")
    for w, c in adv_hits:
        print(f"  {w} -> {c}")

    # Save results
    results = {
        "method": "UPC/TALP WN 1.6 → 3.0 offset mapping (verbs excluded)",
        "summary": {
            "wn_affect_16_synsets": len(all_affect_16),
            "wn_30_mapped_synsets": len(affect_30),
            "unmapped_synsets": len(unmapped),
            "roemolex_entries_with_eng30": len(roemolex),
            "matched_entries": len(hits),
            "unique_words": len(unique_words),
            "affect_categories": len(unique_categs),
            "match_rate": f"{100*len(hits)/len(roemolex):.1f}%",
        },
        "by_pos": dict(pos_hits.most_common()),
        "adjectives": [{"word": w, "affect_categ": c} for w, c in adj_hits],
        "nouns": [{"word": w, "affect_categ": c} for w, c in noun_hits],
        "adverbs": [{"word": w, "affect_categ": c} for w, c in adv_hits],
        "all_matched_words": sorted(unique_words),
    }

    out_path = PIPELINE_DIR / "wn_affect_bridge_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
