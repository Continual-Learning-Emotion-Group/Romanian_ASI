"""Aggregate evaluation results into comparison tables.

Scans eval_results/ for *_metrics.json files and produces formatted tables
following MASIVE Table 4 format.

Usage:
    python -m pipeline.eval.report
    python -m pipeline.eval.report --format latex
    python -m pipeline.eval.report --breakdown source
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "eval_results"


def load_all_metrics(results_dir: Path) -> list[dict]:
    """Load all *_metrics.json files from results_dir."""
    metrics = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        with open(path) as f:
            m = json.load(f)
        m["_file"] = path.name
        metrics.append(m)
    return metrics


def build_main_table(metrics: list[dict]) -> pd.DataFrame:
    """Build main comparison table (rows=models, columns=metrics)."""
    rows = []
    for m in metrics:
        row = {
            "Model": m.get("model", "?").split("/")[-1],
            "Type": m.get("type", "?"),
            "Lang": m.get("language", "?").upper(),
            "Split": m.get("split", "?"),
            "Acc@1": m.get("acc@1", 0),
            "Acc@3": m.get("acc@3", 0),
            "Acc@5": m.get("acc@5", 0),
            "MRR": m.get("mrr", 0),
            "N": m.get("n_samples", 0),
        }
        # Add similarity if present
        if "sim@1" in m:
            row["Sim@1"] = m["sim@1"]
            row["Sim@3"] = m["sim@3"]
            row["Sim@5"] = m["sim@5"]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort: RO first, then EN; within language: MLM first, then generative
    type_order = {"mlm": 0, "generative": 1}
    lang_order = {"RO": 0, "EN": 1}
    df["_sort_lang"] = df["Lang"].map(lang_order).fillna(2)
    df["_sort_type"] = df["Type"].map(type_order).fillna(2)
    df = df.sort_values(["_sort_lang", "Split", "_sort_type", "Model"])
    df = df.drop(columns=["_sort_lang", "_sort_type"])

    return df


def build_breakdown_table(metrics: list[dict], breakdown: str) -> pd.DataFrame:
    """Build a breakdown table for a specific group (source, pattern, gender)."""
    key = f"by_{breakdown}"
    rows = []
    for m in metrics:
        groups = m.get(key, {})
        for group_val, group_metrics in groups.items():
            row = {
                "Model": m.get("model", "?").split("/")[-1],
                "Split": m.get("split", "?"),
                breakdown.capitalize(): group_val,
                "Acc@1": group_metrics.get("acc@1", 0),
                "Acc@3": group_metrics.get("acc@3", 0),
                "Acc@5": group_metrics.get("acc@5", 0),
                "MRR": group_metrics.get("mrr", 0),
                "N": group_metrics.get("n_samples", 0),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, fmt: str = "markdown") -> str:
    """Format DataFrame as markdown or LaTeX."""
    # Format percentages
    pct_cols = [c for c in df.columns if c.startswith("Acc@") or c == "MRR"]
    sim_cols = [c for c in df.columns if c.startswith("Sim@")]

    df_display = df.copy()
    for col in pct_cols:
        df_display[col] = df_display[col].apply(lambda x: f"{x*100:.1f}%")
    for col in sim_cols:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")

    if fmt == "latex":
        return df_display.to_latex(index=False, escape=True)
    else:
        return df_display.to_markdown(index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument(
        "--breakdown", choices=["source", "pattern", "gender"],
        default=None, help="Show breakdown by group",
    )
    args = parser.parse_args()

    metrics = load_all_metrics(args.results_dir)
    if not metrics:
        print(f"No *_metrics.json files found in {args.results_dir}")
        return

    print(f"Found {len(metrics)} result files\n")

    # Main table
    print("=" * 80)
    print("MAIN COMPARISON TABLE")
    print("=" * 80)
    df_main = build_main_table(metrics)
    print(format_table(df_main, args.format))
    print()

    # Breakdown
    if args.breakdown:
        print("=" * 80)
        print(f"BREAKDOWN BY {args.breakdown.upper()}")
        print("=" * 80)
        df_break = build_breakdown_table(metrics, args.breakdown)
        if not df_break.empty:
            print(format_table(df_break, args.format))
        else:
            print(f"  No {args.breakdown} breakdown data found.")
        print()

    # Save
    output_base = args.results_dir / "comparison_table"
    ext = "tex" if args.format == "latex" else "md"

    with open(f"{output_base}.{ext}", "w") as f:
        f.write(format_table(df_main, args.format))

    df_main_json = df_main.to_dict(orient="records")
    with open(f"{output_base}.json", "w") as f:
        json.dump(df_main_json, f, indent=2)

    print(f"Saved to {output_base}.{ext} and {output_base}.json")


if __name__ == "__main__":
    main()
