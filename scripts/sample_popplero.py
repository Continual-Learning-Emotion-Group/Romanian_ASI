#!/usr/bin/env python3
"""Sample 5 random rows from PoPreRo train.csv for visualization and analysis."""

import csv
import random

CSV_PATH = "small_datasets/PoPreRo/Dataset/train.csv"

with open(CSV_PATH, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

samples = random.sample(rows, 5)

for i, r in enumerate(samples, 1):
    print(f"\n{'─' * 60}\nSample {i} (label={r['label']})")
    print(f"{'─' * 60}")
    text = r["full_text"].strip()
    if len(text) > 500:
        text = text[:497] + "..."
    print(text)
