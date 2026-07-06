"""Overlay one per-layer metric across languages, from the metrics_*.json files
that `analyze.py` already wrote. Same computation for every language (identical
method -> directly comparable), one shared axis.

Usage:
    python -m pipeline.viz_hidden.metric_sweep --metric circle_resid \
        --lang ro:pipeline/viz_hidden/out/figs/metrics_ro_canon.json \
        --lang en:pipeline/viz_hidden/masive/out/figs/metrics_en.json \
        --lang es:pipeline/viz_hidden/masive/out/figs/metrics_es.json \
        --out pipeline/viz_hidden/out/figs/fig_circle_resid_sweep.png
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG_COLOR = {"ro": "tab:red", "en": "tab:blue", "es": "tab:green"}
LABELS = {"circle_resid": "circle residual  (PC1-PC2, 0 = round)",
          "silhouette": "silhouette  (cluster separation)",
          "best_valence_corr": "|valence corr|"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="circle_resid")
    ap.add_argument("--lang", action="append", required=True, help="repeatable 'code:metrics.json'")
    ap.add_argument("--out", default="pipeline/viz_hidden/out/figs/fig_circle_resid_sweep.png")
    ap.add_argument("--ymax", type=float, default=0.6)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for spec in args.lang:
        code, path = spec.split(":", 1)
        rows = json.load(open(path))["layers"]
        L = [r["layer"] for r in rows]; V = [r[args.metric] for r in rows]
        ax.plot(L, V, "o-", color=LANG_COLOR.get(code), lw=2, label=code.upper())

    if args.metric == "circle_resid":
        # embedding (L0) and final (L32) roundness is a degeneracy, not a real ring
        ax.axvspan(-1, 1, color="0.85", alpha=0.5, zorder=0)
        ax.axvspan(31, 33, color="0.85", alpha=0.5, zorder=0)
        ax.text(0, args.ymax * 0.96, "embedding", fontsize=7, color="0.5", ha="center", va="top")
        ax.text(32, args.ymax * 0.96, "final", fontsize=7, color="0.5", ha="center", va="top")

    ax.set_ylim(0, args.ymax)
    ax.set_xlabel("layer")
    ax.set_ylabel(LABELS.get(args.metric, args.metric))
    ax.set_title(f"{LABELS.get(args.metric, args.metric)} by layer — same method & scale, all languages")
    ax.grid(alpha=0.2); ax.legend(title="language", loc="upper right")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140); plt.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
