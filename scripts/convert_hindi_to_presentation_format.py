"""Convert hindi_samples_final/*.csv to presentation_hi/{train,val,test}.csv.

Matches the format of presentation_{ro,en,es,fa}/*.csv: columns (id, input, labels),
where `input` has the emotion word replaced by [MASK] and `labels` is a stringified
Python list like ['word'].

Semantics: we replace the WHOLE token that contains the emotion_word (so [MASK] sits
on a word boundary, matching other languages), and set the label to that full token
(i.e. the inflected form as it appears in text). This aligns hindi with romanian's
convention where `fericit` and `fericită` are distinct labels.

Input pool sizes (1000 train + 5000 test, no val) are re-split to 1000/250/1000
to match the other languages: train untouched, val+test carved from the 5000 test pool.
"""
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "presentation_data" / "hindi_samples_final"
DST = ROOT / "presentation_data" / "presentation_hi"
SEED = 42
VAL_SIZE = 250
TEST_SIZE = 1000

_TOKEN_DELIM = re.compile(r'[\s।,.!?"\'()\[\]—–…।:;]+')


def mask_containing_token(sentence: str, word: str) -> tuple[str, str]:
    """Replace the whole whitespace/punct-delimited token containing `word` with [MASK].

    Returns (masked_sentence, containing_token). Picks the first token (in text order)
    that contains `word` as a substring.
    """
    pos = 0
    for match in _TOKEN_DELIM.finditer(sentence + " "):
        tok = sentence[pos:match.start()]
        if tok and word in tok:
            return sentence[:pos] + "[MASK]" + sentence[match.start():], tok
        pos = match.end()
    raise ValueError(f"word {word!r} not found as subword in any token of {sentence!r}")


def to_row(idx: int, row: dict) -> dict:
    sentence = row["sentence"]
    word = row["emotion_word"]
    masked, containing = mask_containing_token(sentence, word)
    source_short = row["source"].split("-", 1)[0].lower()
    return {
        "id": f"hindi_{source_short}_{idx}",
        "input": masked,
        "labels": repr([containing]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "input", "labels"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    with (SRC / "hindi_train_1000.csv").open(encoding="utf-8") as f:
        train_src = list(csv.DictReader(f))
    with (SRC / "hindi_test_5000.csv").open(encoding="utf-8") as f:
        test_src = list(csv.DictReader(f))

    train = [to_row(i, r) for i, r in enumerate(train_src)]
    train_labels = {r["labels"] for r in train}
    train_pairs = {(r["input"], r["labels"]) for r in train}

    rng = random.Random(SEED)
    rng.shuffle(test_src)

    val, test = [], []
    skipped_unseen, skipped_dup = 0, 0
    for i, r in enumerate(test_src):
        converted = to_row(100_000 + i, r)
        if converted["labels"] not in train_labels:
            skipped_unseen += 1
            continue
        if (converted["input"], converted["labels"]) in train_pairs:
            skipped_dup += 1
            continue
        if len(val) < VAL_SIZE:
            val.append(converted)
        elif len(test) < TEST_SIZE:
            test.append(converted)
        else:
            break

    assert len(val) == VAL_SIZE, f"only got {len(val)} val rows (pool exhausted)"
    assert len(test) == TEST_SIZE, f"only got {len(test)} test rows (pool exhausted)"

    write_csv(DST / "train.csv", train)
    write_csv(DST / "val.csv", val)
    write_csv(DST / "test.csv", test)

    from collections import Counter
    train_label_counts = Counter(r["labels"] for r in train)
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test → {DST}")
    print(f"Unique labels in train: {len(train_label_counts)}")
    print(f"Skipped from test pool: {skipped_unseen} unseen-label, {skipped_dup} dup-of-train")
    print(f"Top 10 train labels: {train_label_counts.most_common(10)}")
    print(f"Singleton train labels: {sum(1 for c in train_label_counts.values() if c == 1)}")


if __name__ == "__main__":
    main()
