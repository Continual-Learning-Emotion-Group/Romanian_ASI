# Distributional Mining — Experiment Results

**Branch:** `strategy/distributional-mining`
**Date:** 2026-02-17
**Corpus:** RedditRoAP (26,517 posts) + PoPreRo (28,111 posts) = 54,628 posts

## Summary

Starting from only the **6 Ekman basic emotions** (18 normalized word forms: 12 adjective + 6 noun), the distributional mining pipeline:

1. **Discovered 235 unique candidate words** from 288 pattern matches in Reddit text
2. **Expanded the seed to 251 words** (18 Ekman + 233 discovered)
3. **Extracted 538 ASI candidates** (146 from Ekman words, 392 from new discoveries)

## Phase 1: Discovery — What Worked, What Didn't

### Pattern effectiveness

| Pattern | Words Found | Precision | Notes |
|---------|------------|-----------|-------|
| `plin(ă) de X` | 190 | **Low** | Mostly physical nouns (mașini, gândaci, purici, mucegai) |
| `o stare de X` | 19 | Medium | Mix of emotional (bine, anxietate) and non-emotional (urgență) |
| `un sentiment de X` | 12 | **High** | Mostly genuine emotions (siguranță, vină, neputință) |
| `starea de X` | 8 | Medium | Some good (bine, spirit), some noise |
| `sentimentul de X` | 7 | **High** | Genuine emotions (siguranță, iubire, libertate) |
| `cuprins(ă) de X` | 4 | **High** | Good hits (vină, panică) |
| `senzație de X` | 3 | High | Small count but precise (siguranță) |
| `copleșit(ă) de X` | 3 | High | Small count but precise |
| `emoți(a/e) de X` | 0 | — | No matches found |

**Key finding:** `plin de` dominates with 190/235 words (81%) but has the worst precision. The pattern is too general — Romanian uses "plin de" for both emotional ("plin de furie") and physical ("plin de mașini") contexts. The high-precision patterns (`sentiment de`, `sentimentul de`, `cuprins de`, `copleșit de`) together found only 26 words but are far more reliable.

### Genuinely useful discoveries

Of the 235 discovered words, only **15 matched known emotions** from the existing NOUN_EMOTION_MAP:

| Word | Freq | Emotion | Discovered via |
|------|------|---------|----------------|
| siguranță | 4 | trust | sentiment_de, sentimentul_de, senzatie_de, stare_de |
| iubire | 3 | joy, trust | plin_de, sentimentul_de |
| vină | 3 | sadness | cuprins_de, sentiment_de, sentimentul_de |
| bucurie | 2 | joy | plin_de |
| nostalgie | 2 | sadness | plin_de, sentiment_de |
| ură | 2 | anger, disgust | plin_de |
| furie | 1 | anger | plin_de |
| tristețe | 1 | sadness | sentimentul_de |
| calm | 1 | trust | stare_de |
| durere | 1 | sadness | plin_de |
| teamă | 1 | fear | sentiment_de |
| neliniște | 1 | fear | stare_de |
| dezamăgire | 1 | sadness | sentimentul_de |
| nervozitate | 1 | anger, fear | stare_de |
| scârbă | 1 | disgust | plin_de |

Additional plausible emotion-adjacent words among the "unknown" discoveries: **afecțiune** (3), **regrete** (2), **emoții** (2), **disTracție** (2), **incertitudini** (2), **libertate** (2).

## Phase 3: Extraction — Candidate Quality

### Confidence distribution

| Confidence | Count | % | Description |
|-----------|-------|---|-------------|
| 0.9 | 146 | 27% | Ekman seed words (high trust) |
| 0.7 | 45 | 8% | New words via primary patterns (mă simt, simt) |
| 0.5 | 347 | 65% | New words via secondary patterns (sunt, am, mi-e) |

### Emotion distribution

| Emotion | Candidates |
|---------|-----------|
| unknown | 339 (63%) |
| fear | 139 (26%) |
| joy | 25 (5%) |
| surprise | 18 (3%) |
| sadness | 7 |
| disgust | 6 |
| trust | 5 |

The dominance of "unknown" reflects that most discovered words aren't in NOUN_EMOTION_MAP and would need manual annotation.

### Top false positive patterns

The biggest source of noise is discovered words matching in non-affective contexts:

- **"sunt persoane"** (72 hits) — "there are people", not "I am people". The `sunt` pattern is ambiguous (1st person / 3rd plural).
- **"sunt bine" / "mi-e bine"** (122 hits) — Actually legitimate ("I'm fine"), but "bine" is more of a general state than an emotion.
- **"sunt chestii"** (11 hits) — "there are things", pure noise.
- **"am jocuri/oportunități/autobuze"** — "I have games/opportunities/buses", noun pattern matching non-emotional nouns.

### Pattern breakdown

| Pattern | Candidates | Notes |
|---------|-----------|-------|
| sunt_adj_present | 272 (51%) | Most productive but noisiest |
| mie_short | 92 (17%) | Good: "mi-e frică", "mi-e bine" |
| am_noun_present | 50 (9%) | Noisy: "am X" matches non-emotions |
| imi_este_present | 38 (7%) | Decent quality |
| ma_simt_present | 21 (4%) | **Highest quality** — "mă simt X" is explicitly affective |
| Others | 65 (12%) | Generally good quality |

## Conclusions

### What this experiment shows

1. **Distributional discovery can find real emotion words**, but the signal-to-noise ratio is poor on small corpora. Only 15/235 (6.4%) of discovered words were confirmed emotions.

2. **Pattern quality varies dramatically.** The `plin de` pattern is too broad for emotion discovery (81% of matches, mostly physical nouns). High-precision patterns (`sentiment de`, `copleșit de`, `cuprins de`) are better but yield few matches on 55K posts.

3. **The extraction pipeline amplifies noise.** Adding 233 unvetted words to the seed (most with "unknown" emotion) generates many false positives, especially through the `sunt` and `am` patterns which are already ambiguous.

4. **Reddit corpus size is limiting.** With only 54K posts, even good patterns fire rarely (most words appear 1-2 times). The FULG dataset (150B tokens) would likely yield far more hits on the high-precision patterns.

### Recommendations for improvement

1. **Drop or restrict `plin de`** from discovery patterns — it produces mostly physical nouns. Could require co-occurrence with emotion context words.

2. **Use min_freq >= 2** to filter hapax legomena. Running with `--min-freq 2` would reduce discovered words from 235 to ~30, dramatically improving precision.

3. **Manual curation step** between Phase 2 and Phase 3 — review discovered words before adding to seed, discard obvious non-emotions.

4. **Run on FULG** — the 150B-token dataset would give the high-precision patterns (`sentiment de`, `copleșit de`) enough data to discover many more genuine emotion words.

5. **Two-pass approach** — first discover on FULG with strict patterns only, manually curate, then extract with curated expanded seed on all corpora.

### Bottom line

The distributional mining strategy is **viable but needs curation**. On this small Reddit corpus, it found 15 genuine emotion words and produced 538 candidates, but ~65% of candidates are low-confidence matches with unknown emotions. The approach would benefit most from (a) running on a larger corpus and (b) a manual filtering step before seed expansion.
