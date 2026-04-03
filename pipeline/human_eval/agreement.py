"""Inter-annotator agreement and LLM correlation analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_FILE = DATA_DIR / "human_eval_sample.jsonl"


def load_annotations(path: Path) -> dict[str, int]:
    """Load MTurk results CSV exported from Sandbox.

    Expected columns: id (Input.id), affect (Answer.affect).
    MTurk exports answers as Answer.<field_name>.

    Returns dict mapping candidate id -> score (0-3).
    """
    import csv
    annotations = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("Input.id", "")
            # Map MTurk radio values to numeric scores
            affect = row.get("Answer.affect", "")
            score_map = {
                "not_affect": 0,
                "unlike_affect": 1,
                "like_affect": 2,
                "is_affect": 3,
            }
            score = score_map.get(affect)
            if score is not None and cid:
                annotations[cid] = score
    return annotations


def load_llm_scores(path: Path) -> dict[str, int]:
    """Load LLM scores from the sampled JSONL."""
    scores = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            scores[c["id"]] = c["llm_affect_score"]
    return scores


def compute_agreement(ann1: dict[str, int], ann2: dict[str, int],
                      llm: dict[str, int]):
    """Compute all agreement metrics."""
    # Align by shared IDs
    shared = sorted(set(ann1) & set(ann2))
    if not shared:
        print("ERROR: No shared IDs between annotators!")
        return

    y1 = np.array([ann1[k] for k in shared])
    y2 = np.array([ann2[k] for k in shared])

    print(f"=== Inter-Annotator Agreement (n={len(shared)}) ===\n")

    # Percent agreement
    exact = np.mean(y1 == y2)
    print(f"Percent agreement (exact): {exact:.1%}")

    # Cohen's Kappa (unweighted)
    kappa = cohen_kappa_score(y1, y2)
    print(f"Cohen's Kappa (unweighted): {kappa:.3f}")

    # Weighted Kappa (quadratic)
    kappa_w = cohen_kappa_score(y1, y2, weights="quadratic")
    print(f"Cohen's Kappa (quadratic weighted): {kappa_w:.3f}")

    # Binary agreement: 0-1 = not affect, 2-3 = affect
    y1_bin = (y1 >= 2).astype(int)
    y2_bin = (y2 >= 2).astype(int)
    kappa_bin = cohen_kappa_score(y1_bin, y2_bin)
    agree_bin = np.mean(y1_bin == y2_bin)
    print(f"\nBinary (0-1 vs 2-3):")
    print(f"  Percent agreement: {agree_bin:.1%}")
    print(f"  Cohen's Kappa: {kappa_bin:.3f}")

    # Confusion matrix (annotator 1 vs 2)
    print(f"\nConfusion matrix (Annotator 1 rows vs Annotator 2 cols):")
    cm = confusion_matrix(y1, y2, labels=[0, 1, 2, 3])
    print(f"{'':>8} {0:>5} {1:>5} {2:>5} {3:>5}")
    for i, row in enumerate(cm):
        print(f"  {i:>5}: {row[0]:>5} {row[1]:>5} {row[2]:>5} {row[3]:>5}")

    # LLM correlation
    shared_llm = sorted(set(shared) & set(llm))
    if shared_llm:
        y1_l = np.array([ann1[k] for k in shared_llm])
        y2_l = np.array([ann2[k] for k in shared_llm])
        yl = np.array([llm[k] for k in shared_llm])
        y_mean = (y1_l + y2_l) / 2.0

        print(f"\n=== LLM vs Human (n={len(shared_llm)}) ===\n")

        rho, p = spearmanr(y_mean, yl)
        print(f"Spearman's rho (mean human vs LLM): {rho:.3f} (p={p:.4f})")

        # Per-annotator correlation
        rho1, p1 = spearmanr(y1_l, yl)
        rho2, p2 = spearmanr(y2_l, yl)
        print(f"Spearman's rho (Ann1 vs LLM): {rho1:.3f} (p={p1:.4f})")
        print(f"Spearman's rho (Ann2 vs LLM): {rho2:.3f} (p={p2:.4f})")

        # Confusion matrix (mean human binary vs LLM binary)
        yh_bin = (y_mean >= 1.5).astype(int)
        yl_bin = (yl >= 2).astype(int)
        kappa_hl = cohen_kappa_score(yh_bin, yl_bin)
        print(f"\nBinary agreement (human vs LLM): Kappa={kappa_hl:.3f}")

        cm_hl = confusion_matrix(yl_bin, yh_bin, labels=[0, 1])
        print(f"\nConfusion matrix (LLM rows vs Human cols, binary):")
        print(f"{'':>12} {'not':>6} {'affect':>6}")
        print(f"  {'not':>9}: {cm_hl[0][0]:>6} {cm_hl[0][1]:>6}")
        print(f"  {'affect':>9}: {cm_hl[1][0]:>6} {cm_hl[1][1]:>6}")


def main():
    parser = argparse.ArgumentParser(description="Compute agreement metrics")
    parser.add_argument("annotator1", type=Path, help="MTurk results CSV for annotator 1")
    parser.add_argument("annotator2", type=Path, help="MTurk results CSV for annotator 2")
    parser.add_argument("--sample", type=Path, default=SAMPLE_FILE,
                        help="Sampled JSONL with LLM scores")
    args = parser.parse_args()

    ann1 = load_annotations(args.annotator1)
    ann2 = load_annotations(args.annotator2)
    llm = load_llm_scores(args.sample)

    print(f"Annotator 1: {len(ann1)} annotations")
    print(f"Annotator 2: {len(ann2)} annotations")
    print(f"LLM scores: {len(llm)} candidates\n")

    compute_agreement(ann1, ann2, llm)


if __name__ == "__main__":
    main()
