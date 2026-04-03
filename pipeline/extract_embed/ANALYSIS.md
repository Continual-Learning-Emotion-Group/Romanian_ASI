# Embedding Similarity Extraction — Analysis & Conclusions

## Experiment Setup

- **Model**: `intfloat/multilingual-e5-base` (278M params, 768-dim)
- **Anchors**: Synthetic — all combinations of 20 "I feel" verb templates x 524 seed words = 3,912 anchors
- **Candidates**: All sentences from merged_corpus.jsonl (409,157 sentences from 105,927 posts), no pre-filter
- **GPU**: Modal T4
- **Similarity**: Cosine (dot product on normalized embeddings), max over all anchors per sentence

## Key Numbers

| Metric | Value |
|--------|-------|
| Total sentences embedded | 409,157 |
| Hits >= 0.75 (threshold) | 408,802 (99.9%) |
| Hits >= 0.90 | 342 |
| Hits >= 0.90 after filtering bad noun anchors | 342 |
| Novel (not found by regex) >= 0.90 | 164 |
| Regex-known >= 0.90 | 178 |
| Max confidence | 0.9373 |

## Core Problem

Synthetic anchors ("mă simt fericit", "sunt trist") are too short and generic. The E5 model sees
most Romanian sentences as moderately similar to them (median ~0.836), so there is almost no
separation between genuine ASI expressions and random text. Only above 0.90 do we get meaningful
signal, and at that threshold most hits are near-exact matches to what regex already catches.

## Discovered Novel "I Feel" Patterns

These are verb constructions found in NOVEL hits that are NOT covered by our 20 regex patterns:

### 1. "îmi pare rău" (I'm sorry / I feel bad)
- "Îmi pare rău." (0.9219)
- "Îmi pare rău " (0.9094)
- "Îmi pare rău pentru tine." (0.9018)
- "chiar imi pare rau" (0.9008)
- **Status**: Common Romanian expression. Borderline ASI — more of an apology than a felt state. Could be a new pattern if we want to capture regret/sympathy.

### 2. "am rămas [adj]" (I remained [adj] / I was left [adj])
- "Am rămas perplex." (0.9230)
- "Am rămas perplex!" (0.9076)
- "Am rămas plăcut surprins de asta." (0.9004)
- **Status**: Genuine ASI pattern. "Am rămas" + adjective expresses a resulting emotional state. Worth adding as a new pattern.

### 3. "a fost [adj]" (it was [adj]) — 3rd person as felt evaluation
- "a fost minunat." (0.9143, 0.9139 x7)
- "a fost groaznic." (0.9133 x3)
- "a fost oribil." (0.9095 x2)
- "a fost trist .." (0.9018)
- "a fost grozav" (0.9074)
- "a fost excelent." (0.9130)
- "a fost teribil." (0.9057 x3)
- "a fost destul de rau." (0.9019)
- "a fost rau." (0.9010 x4)
- "a fost dureros." (0.9010 x2)
- "a fost plictisitor!" (0.9008)
- **Status**: NOT first-person ASI. These are evaluative statements about an experience, not "I feel" expressions. Should NOT be added as a pattern — noise from the embedding model conflating "am fost [adj]" (I was) with "a fost [adj]" (it was).

### 4. "mă ia cu [noun]" / "m-a luat [noun]" (it takes me with / it took me)
- "Mă ia cu rău." (0.9184)
- "M-a luat complet prin surprindere." (0.9068)
- **Status**: Genuine felt-state idiom. "Mă ia cu rău" = I feel nauseous/sick. "M-a luat prin surprindere" = I was taken by surprise. Worth considering as a new pattern.

### 5. "simt că cedez / simt ca înnebunesc" (I feel like I'm losing it)
- "Simt că cedez nervos." (0.9185)
- "Simt ca înnebunesc." (0.9082)
- **Status**: These are already caught by our "simt că" pattern in theory, but with verb phrases after "că" rather than adjectives. The regex expects "simt că sunt [SEED]" but these use "simt că + verb". Could extend the "simt că" pattern.

### 6. "rămân [adj]" (I remain [adj])
- "Rămân perplex." (0.9160)
- **Status**: Same as "am rămas" but present tense. Genuine ASI pattern.

### 7. "am făcut atac de panică" (I had a panic attack)
- "Am făcut atac de panică." (0.9172)
- "Am făcut atac de panică ." (0.9149)
- **Status**: Specific medical/emotional idiom. Not a general pattern, but expresses felt state.

### 8. "am ajuns la [noun]" (I reached [noun/state])
- "Am ajuns la disperare." (0.9143)
- **Status**: Expresses reaching an emotional state. Could be a pattern: "am ajuns la" + emotion noun.

### 9. "sunt la culmea [noun-ului]" (I'm at the peak of [noun])
- "Sunt la culmea disperării" (0.9092)
- **Status**: Intensified emotional expression. Very specific idiom.

### 10. "mă bucur" (I'm glad)
- "Bun mă bucur" (0.9044)
- **Status**: Very common Romanian "I'm happy/glad". Reflexive verb expressing joy. Could be its own pattern.

### 11. "mă relaxează / mă amețești" (it relaxes me / you dizzy me)
- "Mă relaxează." (0.9126)
- "Mă amețești." (0.9028)
- **Status**: Causative constructions — something causes an emotional state. Different from "I feel X" patterns.

### 12. "m-am plictisit" (I got bored)
- "m-am plictisit." (0.9097)
- **Status**: Already in seed as "plictisit" but the reflexive verb "a se plictisi" is a different construction than "m-am simțit plictisit". This should already be caught by regex if "plictisit" is in seed + "m-am simțit" pattern exists.

### 13. "aș fi [adj]" (I would be [adj]) — conditional of "a fi"
- "Aș fi foarte recunoscător." (0.9111)
- **Status**: Conditional mood of "to be". We have "m-aș simți" (conditional of "a simți") but not "aș fi" (conditional of "a fi"). Worth adding.

### 14. "să fiu [adj]" (to be [adj]) — subjunctive
- "să fiu optimistă!" (0.9113)
- "Să fiu mulțumită." (0.9158)
- "să fiu optimistă !" (0.9103)
- "imi place sa fiu surprins." (0.9033)
- "s-ar putea sa fiu sentimentalist." (0.9009)
- **Status**: We already have "o să fiu" (future) pattern, but "să fiu" (subjunctive without "o") is not explicitly covered. Partially overlaps.

### 15. "mi-a plăcut" (I liked it)
- "mi-a placut acest lucru." (0.9033 x2)
- **Status**: Very common. Not quite "I feel X" but expresses positive affect.

## Discovered Novel Seed Words

Words that appeared in NOVEL hits but are not in the current seed:

| Word | Context | Assessment |
|------|---------|------------|
| special | "Mă simt special." (0.9098) | Valid affective state. ADD. |
| insultat | "Mă simt foarte insultat" (0.9061) | Valid — feeling insulted. Already may be in seed? Check. |
| rupt | "m-am simtit rupt." (0.9017) | Colloquial "broken/shattered". Valid felt state. ADD. |
| murdară | "Mă simțeam murdară" (0.9007) | "Dirty" as felt state. Borderline — could be literal. |
| credul | "Da, am fost credul." (0.9051) | "Gullible/credulous". Not really an affective state. SKIP. |
| sentimentalist | "s-ar putea sa fiu sentimentalist" (0.9009) | Personality trait more than felt state. SKIP. |
| pesimist | "Sunt prea pesimist?" (0.9016) | Attitude, not felt state. Already in seed? Check. |

## False Positives / Noise

Even at >= 0.90, significant noise:
- **"sunt multumit/multumita"** — ~80 duplicate hits from product reviews. Regex-known, massively over-represented.
- **"Foarte multumit"** — marked NOVEL only because it lacks the verb "sunt", but it's just a truncated review title.
- **"a fost [adj]"** — 3rd person evaluative, NOT first-person ASI (see pattern #3 above).
- **"De curiozitate" / "Intreb din curiozitate"** — not ASI expressions, just "out of curiosity".
- **"There is hope" / "i felt filmul"** — English/garbage text leaking through.
- **"Cu încredere" / "Cu scârbă"** — prepositional phrases, not "I feel" expressions.
- **"Credeam că." → "aveam încredere"** — false semantic match.

## Conclusions

### The synthetic anchor approach has fundamental limitations:
1. **Poor discrimination** — short synthetic anchors produce a very narrow similarity band (0.80-0.87 for most text), making thresholding ineffective. 99.9% of sentences pass 0.75.
2. **Near-exact matching only** — at the useful threshold (0.90+), hits are almost exclusively near-exact textual matches to anchor phrases, with minor variations (added modifiers, missing diacritics, feminine forms).
3. **Limited novel discovery** — of 164 novel hits at 0.90+, most are trivial variations ("foarte multumit" without "sunt"), 3rd-person evaluatives ("a fost minunat"), or non-ASI phrases. Only ~15-20 are genuinely interesting new patterns.

### Useful outputs to carry forward:

**New patterns to consider adding:**
1. `am_ramas_adj` — "am rămas [adj]" / "rămân [adj]" (I remained/remain [state])
2. `as_fi_adj` — "aș fi [adj]" (I would be [state]) — conditional of "a fi"
3. `sa_fiu_adj` — "să fiu [adj]" (that I be [state]) — subjunctive of "a fi" without "o"
4. `imi_pare_rau` — "îmi pare rău" (I feel bad/sorry) — if we want regret/sympathy
5. `ma_ia_cu` — "mă ia cu [noun]" (idiom for feeling nauseous/overwhelmed)

**New seed words to consider:**
1. `special` — "Mă simt special"
2. `rupt` — "m-am simțit rupt" (colloquial "broken/shattered")

### Recommendation:
The embedding approach with synthetic anchors is not effective for discovery at scale. For the small
datasets (106K records), the regex pattern matcher with the 524-word enriched seed already captures
the vast majority of "I feel" expressions. The few novel patterns discovered here (am rămas, aș fi,
să fiu, îmi pare rău) are better added as new regex patterns than found via embeddings.

If embeddings are to be used in future, **corpus-mined anchors** (real sentences from regex matches)
would provide much better discrimination than synthetic ones, as the original experiment showed.
