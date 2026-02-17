# Baseline Pattern Matching: MASIVE 6-Emotion Seed

Baseline extraction using only the 6 basic MASIVE emotions on RedditRoAP + PoPreRo. No bootstrapping — this establishes the starting point that other strategies build upon.

## Seed

6 emotions mapped to Romanian adjective forms (masc/fem/masc-pl/fem-pl) + nouns, with diacritics-stripped variants auto-generated:

| Emotion | Adjectives | Nouns |
|---------|-----------|-------|
| happy | fericit/fericită/fericiți/fericite | fericire, bucurie |
| sad | trist/tristă/triști/triste | tristețe |
| angry | furios/furioasă/furioși/furioase, supărat/supărată/supărați/supărate | furie, mânie |
| afraid | speriat/speriată/speriați/speriate, înfricoșat/înfricoșată/înfricoșați/înfricoșate | frică, teamă, spaimă |
| disgusted | dezgustat/dezgustată/dezgustați/dezgustate, scârbit/scârbită/scârbiți/scârbite | dezgust, scârbă |
| surprised | surprins/surprinsă/surprinși/surprinse, uimit/uimită/uimiți/uimite | surpriză, uimire |

**Total: 85 words** (66 adjectives + 19 nouns, including diacritics variants)

## Results

Run date: 2026-02-17

| Metric | Value |
|--------|-------|
| Records processed | 54,623 |
| Unique texts matched | 219 |
| Total pattern matches | 234 |
| Overall match rate | 0.40% |
| Unique seed words found | 28 / 85 |

### By dataset

| Dataset | Records | Matches | Match rate |
|---------|---------|---------|------------|
| RedditRoAP | 26,517 | 174 | 0.66% |
| PoPreRo | 28,106 | 60 | 0.21% |

RedditRoAP yields ~3x the match rate — expected since Reddit posts are informal/emotional while PoPreRo is news articles.

### By pattern

| Pattern | Matches |
|---------|---------|
| mi-e [noun] | 87 |
| sunt [adj] | 58 |
| îmi este [noun] | 37 |
| am fost [adj] | 20 |
| eram [adj] | 9 |
| îmi era [noun] | 6 |
| am [noun] | 6 |
| suntem [adj] | 3 |
| m-am simțit [adj] | 3 |
| mă simt [adj] | 2 |
| mă simțeam [adj] | 2 |
| aveam [noun] | 1 |

### By emotion

| Emotion | Matches |
|---------|---------|
| fear | 144 |
| joy | 35 |
| surprise | 27 |
| anger | 13 |
| disgust | 8 |
| sadness | 7 |

Fear dominates due to the very common `mi-e frică` / `mi-e teamă` expressions in Romanian.

## Usage

```bash
python -m experiments.baseline_pattern_matching.extract_baseline
python -m experiments.baseline_pattern_matching.extract_baseline --sample 20
```

## Output files

- `data/reddit_baseline_candidates.jsonl` — extracted candidates
- `data/reddit_baseline_candidates.stats.json` — statistics

## Observations

- Only 28 of 85 seed words were actually found — many plural/feminine forms don't appear naturally in first-person expressions
- Noun patterns (`mi-e`, `îmi este`) are the most productive, not adjective patterns (`mă simt`, `sunt`)
- The `sunt [adj]` pattern is the top adjective pattern but is inherently noisy (1st person "I am" vs 3rd person plural "they are")
- Sadness is underrepresented despite `trist` being a common word — likely because people express sadness through other constructions not captured here
