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
            if not cid:
                continue

            # MTurk exports radio buttons as separate boolean columns:
            # Answer.affect.not_affect, Answer.affect.unlike_affect, etc.
            bool_map = {
                "Answer.affect.not_affect": 0,
                "Answer.affect.unlike_affect": 1,
                "Answer.affect.like_affect": 2,
                "Answer.affect.is_affect": 3,
            }
            score = None
            for col, val in bool_map.items():
                if row.get(col, "").lower() == "true":
                    score = val
                    break

            # Fallback: single Answer.affect column
            if score is None:
                affect = row.get("Answer.affect", "")
                fallback_map = {
                    "not_affect": 0, "unlike_affect": 1,
                    "like_affect": 2, "is_affect": 3,
                }
                score = fallback_map.get(affect)

            if score is not None:
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
                      llm: dict[str, int], save_path: Path | None = None):
    """Compute all agreement metrics."""
    # Align by shared IDs
    shared = sorted(set(ann1) & set(ann2))
    if not shared:
        print("ERROR: No shared IDs between annotators!")
        return

    y1 = np.array([ann1[k] for k in shared])
    y2 = np.array([ann2[k] for k in shared])

    results = {"n_shared": len(shared)}

    print(f"=== Inter-Annotator Agreement (n={len(shared)}) ===\n")

    # Percent agreement
    exact = float(np.mean(y1 == y2))
    print(f"Percent agreement (exact): {exact:.1%}")
    results["percent_agreement_exact"] = round(exact, 4)

    # Cohen's Kappa (unweighted)
    kappa = cohen_kappa_score(y1, y2)
    print(f"Cohen's Kappa (unweighted): {kappa:.3f}")
    results["cohens_kappa_unweighted"] = round(kappa, 4)

    # Weighted Kappa (quadratic)
    kappa_w = cohen_kappa_score(y1, y2, weights="quadratic")
    print(f"Cohen's Kappa (quadratic weighted): {kappa_w:.3f}")
    results["cohens_kappa_quadratic_weighted"] = round(kappa_w, 4)

    # Binary agreement: 0-1 = not affect, 2-3 = affect
    y1_bin = (y1 >= 2).astype(int)
    y2_bin = (y2 >= 2).astype(int)
    kappa_bin = cohen_kappa_score(y1_bin, y2_bin)
    agree_bin = float(np.mean(y1_bin == y2_bin))
    print(f"\nBinary (0-1 vs 2-3):")
    print(f"  Percent agreement: {agree_bin:.1%}")
    print(f"  Cohen's Kappa: {kappa_bin:.3f}")
    results["binary_percent_agreement"] = round(agree_bin, 4)
    results["binary_cohens_kappa"] = round(kappa_bin, 4)

    # Confusion matrix (annotator 1 vs 2)
    cm = confusion_matrix(y1, y2, labels=[0, 1, 2, 3])
    print(f"\nConfusion matrix (Annotator 1 rows vs Annotator 2 cols):")
    print(f"{'':>8} {0:>5} {1:>5} {2:>5} {3:>5}")
    for i, row in enumerate(cm):
        print(f"  {i:>5}: {row[0]:>5} {row[1]:>5} {row[2]:>5} {row[3]:>5}")
    results["confusion_matrix_annotators"] = cm.tolist()

    # Per-item merged annotations
    per_item = []
    for cid in shared:
        item = {
            "id": cid,
            "annotator1_score": int(ann1[cid]),
            "annotator2_score": int(ann2[cid]),
            "human_mean": round((ann1[cid] + ann2[cid]) / 2.0, 2),
        }
        if cid in llm:
            item["llm_score"] = int(llm[cid])
        per_item.append(item)

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
        results["spearman_mean_human_vs_llm"] = {"rho": round(rho, 4), "p": round(p, 6)}

        # Per-annotator correlation
        rho1, p1 = spearmanr(y1_l, yl)
        rho2, p2 = spearmanr(y2_l, yl)
        print(f"Spearman's rho (Ann1 vs LLM): {rho1:.3f} (p={p1:.4f})")
        print(f"Spearman's rho (Ann2 vs LLM): {rho2:.3f} (p={p2:.4f})")
        results["spearman_ann1_vs_llm"] = {"rho": round(rho1, 4), "p": round(p1, 6)}
        results["spearman_ann2_vs_llm"] = {"rho": round(rho2, 4), "p": round(p2, 6)}

        # Confusion matrix (mean human binary vs LLM binary)
        yh_bin = (y_mean >= 2.0).astype(int)
        yl_bin = (yl >= 2).astype(int)
        kappa_hl = cohen_kappa_score(yh_bin, yl_bin)
        print(f"\nBinary agreement (human vs LLM): Kappa={kappa_hl:.3f}")
        results["binary_human_vs_llm_kappa"] = round(kappa_hl, 4)

        # Human validation rate: of LLM-positive (>=2), how many are human-positive?
        llm_pos = (yl >= 2)
        human_pos_of_llm_pos = float(np.mean(yh_bin[llm_pos])) if llm_pos.sum() > 0 else None
        if human_pos_of_llm_pos is not None:
            print(f"Human validation rate (LLM>=2 confirmed by humans): {human_pos_of_llm_pos:.1%}")
            results["human_validation_rate"] = round(human_pos_of_llm_pos, 4)

        cm_hl = confusion_matrix(yl_bin, yh_bin, labels=[0, 1])
        print(f"\nConfusion matrix (LLM rows vs Human cols, binary):")
        print(f"{'':>12} {'not':>6} {'affect':>6}")
        print(f"  {'not':>9}: {cm_hl[0][0]:>6} {cm_hl[0][1]:>6}")
        print(f"  {'affect':>9}: {cm_hl[1][0]:>6} {cm_hl[1][1]:>6}")
        results["confusion_matrix_llm_vs_human_binary"] = cm_hl.tolist()

    # Save results
    if save_path:
        output = {"metrics": results, "per_item": per_item}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute agreement metrics")
    parser.add_argument("annotator1", type=Path, help="MTurk results CSV for annotator 1")
    parser.add_argument("annotator2", type=Path, help="MTurk results CSV for annotator 2")
    parser.add_argument("--sample", type=Path, default=SAMPLE_FILE,
                        help="Sampled JSONL with LLM scores")
    parser.add_argument("--save", type=Path, default=DATA_DIR / "human_eval_results.json",
                        help="Path to save JSON results (default: data/human_eval_results.json)")
    args = parser.parse_args()

    ann1 = load_annotations(args.annotator1)
    ann2 = load_annotations(args.annotator2)
    llm = load_llm_scores(args.sample)

    print(f"Annotator 1: {len(ann1)} annotations")
    print(f"Annotator 2: {len(ann2)} annotations")
    print(f"LLM scores: {len(llm)} candidates\n")

    compute_agreement(ann1, ann2, llm, save_path=args.save)


if __name__ == "__main__":
    main()
