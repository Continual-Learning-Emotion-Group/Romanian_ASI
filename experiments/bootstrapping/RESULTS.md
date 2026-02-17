# Strategy 2: MASIVE-Style Bootstrapping

Replicates the MASIVE paper's bootstrapping approach for Romanian: start with Ekman basic emotion seeds, then iteratively expand via "I feel X and Y" conjunction mining.

## Method

1. **Ekman seed** (24 adjective forms): 6 emotions x masculine/feminine pairs
2. **Conjunction mining**: Find patterns like `sunt X si Y`, `ma simt X dar Y` where X is a known seed word, capture Y as candidate
3. **Validation**: Min length, gender agreement, function word filter, co-occurrence threshold
4. **Iterate**: Add validated Y words to seed, repeat (up to 4 rounds)
5. **Final extraction**: Use expanded seed with full PatternMatcher (18 patterns)

### Conjunction Patterns

9 verb forms x conjunction variants:
- Primary: `ma simt`, `m-am simtit`, `ma simteam`
- Secondary: `sunt`, `eram`, `am fost`, `suntem`
- Plural: `ne simtim`, `ne-am simtit`

Conjunctions: `si`, `dar`, `insa`, `dar si`, `ba chiar`, `,` (comma lists)

### Validation Filters

1. Min length >= 3 characters
2. Not already in seed
3. Gender agreement (Romanian adjective morphology: feminine `-a`, `-ata`, `-oasa` etc.)
4. Not a function word (80+ Romanian determiners, pronouns, prepositions, common verbs)
5. Co-occurrence with >= N distinct X seeds (configurable, default 2)

## Results

**Corpus**: RedditRoAP (26,517) + PoPreRo (28,106) = 54,623 texts

### Bootstrapping (co-occurrence threshold = 1)

| Round | Seed Size | Conjunction Matches | Candidates | Accepted | New Words |
|-------|-----------|-------------------|------------|----------|-----------|
| 1     | 24        | 13                | 11         | 3        | mahnit, afectata, nervos |
| 2     | 27        | 16                | 14         | 1        | transpirat |
| 3     | 28        | 16                | 14         | 0        | (saturated) |

**Seed growth**: 24 -> 28 words (+4)

New words discovered:
- **mahnit** (grieving) -- from co-occurrence with `suparat`
- **afectata** (affected) -- from co-occurrence with `suparata`
- **nervos** (nervous) -- from co-occurrence with `trist`
- **transpirat** (sweaty/anxious) -- from co-occurrence with `nervos` (round 2 chain)

### Extraction

| Metric | Ekman Only (24 words) | Expanded (28 words) |
|--------|----------------------|---------------------|
| Texts matched | 391 | 391 |
| Total matches | ~424 | 428 |
| Unique seed words used | 37 | 39 |

**Total output**: 428 ASI candidates

### Matches by Pattern

| Pattern | Count |
|---------|-------|
| am_noun_present | 151 |
| mie_short | 119 |
| sunt_adj_present | 48 |
| imi_este_present | 47 |
| am_fost_adj_perfect | 22 |
| aveam_noun_imperfect | 12 |
| eram_adj_imperfect | 9 |
| imi_era_imperfect | 8 |
| simt_noun | 3 |
| mam_simtit_perfect | 3 |
| ma_simt_present | 2 |
| ma_simteam_imperfect | 2 |
| simteam_noun | 2 |

### Matches by Emotion

| Emotion | Count |
|---------|-------|
| fear | 160 |
| joy | 126 |
| trust | 61 |
| sadness | 54 |
| anticipation | 54 |
| surprise | 25 |
| anger | 8 |
| disgust | 8 |

## Limitations

- **Small corpus**: 54K texts yields only 13-16 conjunction matches per round. MASIVE used much larger Reddit data. The bootstrapping saturates quickly.
- **Co-occurrence threshold tradeoff**: Threshold=2 finds no new words (too sparse); threshold=1 accepts words with single evidence (lower confidence).
- **No chaining within lists**: "eram suparat, mahnit, alarmat" captures mahnit as Y from suparat, but alarmat requires mahnit to become a seed first (happens in next round).
- **Noun patterns dominate**: `am [noun]` and `mi-e [noun]` account for most matches since PatternMatcher uses the full EMOTION_NOUNS_ONLY set (~81 nouns) regardless of the adjective seed.

## Output Files

- `data/bootstrapped_asi_candidates.jsonl` -- ASI candidates (common schema)
- `data/bootstrap_expanded_seed.json` -- Expanded seed (28 words)
- `data/bootstrap_provenance.json` -- Full bootstrapping history per round
- `data/bootstrapped_asi_candidates.stats.json` -- Extraction statistics

## Usage

```bash
# Default (co-occurrence threshold = 2)
python -m experiments.bootstrapping.bootstrap_candidates

# Lower threshold for small corpora
python -m experiments.bootstrapping.bootstrap_candidates --co-occurrence-threshold 1

# Show more samples
python -m experiments.bootstrapping.bootstrap_candidates --co-occurrence-threshold 1 --sample 20
```
