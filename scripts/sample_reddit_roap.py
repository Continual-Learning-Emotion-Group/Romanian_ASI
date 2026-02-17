#!/usr/bin/env python3
"""Sample 5 random rows from RedditRoAP train.parquet for visualization and analysis."""

import random

import pandas as pd

PARQUET_PATH = "small_datasets/RedditRoAP/train.parquet"

df = pd.read_parquet(PARQUET_PATH)
samples = df.sample(5)

for i, (_, row) in enumerate(samples.iterrows(), 1):
    print(f"\n{'─' * 60}\nSample {i}")
    print(f"{'─' * 60}")
    labels = []
    if pd.notna(row.get("SUBDIALECT")):
        labels.append(f"subdialect={row['SUBDIALECT']}")
    if pd.notna(row.get("STATUS")):
        labels.append(f"status={row['STATUS']}")
    if pd.notna(row.get("LABELS")):
        labels.append(f"topic={row['LABELS']}")
    if pd.notna(row.get("PERSONAL INCLINATION")):
        labels.append(f"inclination={row['PERSONAL INCLINATION']}")
    if labels:
        print(" | ".join(labels))
    print()
    text = str(row["TEXT"]).strip()
    if len(text) > 500:
        text = text[:497] + "..."
    print(text)
