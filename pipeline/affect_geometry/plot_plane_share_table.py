"""Render the plane-share comparison table as a PNG, from the analysis JSONs.

Run: python -m pipeline.affect_geometry.plot_plane_share_table
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
NAMES = {"en": "English", "es": "Spanish", "ro": "Romanian"}


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.3g}"
    mantissa, exponent = f"{p:.0e}".split("e")
    return f"{mantissa}·10⁻{abs(int(exponent))}"


def main():
    rows = []
    for lang in ("en", "es", "ro"):
        s = json.loads(
            (PACKAGE / f"results/plane_share_russell_{lang}.json").read_text()
        )["summary"]
        name = f"{NAMES[lang]} (L{s['layer']})"
        rows += [
            (name, "Original anchors (in-sample)", str(s["n_anchor_lemmas"]),
             f"{s['anchor_in_sample_median_plane_share']:.1%}",
             f"{s['anchor_in_sample_mean_plane_share']:.1%}", ""),
            ("", "Anchors, held-out (LOO)", str(s["n_anchor_lemmas"]),
             f"{s['anchor_loo_median_plane_share']:.1%}",
             f"{s['anchor_loo_mean_plane_share']:.1%}",
             fmt_p(s["mannwhitney_p_two_sided"])),
            ("", "Broader states", str(s["n_broader_lemmas"]),
             f"{s['broader_median_plane_share']:.1%}",
             f"{s['broader_mean_plane_share']:.1%}", ""),
        ]

    headers = ["Language", "Group", "n", "Median\nplane share",
               "Mean plane share\n(avg EV)", "MW p\n(LOO vs broader)"]
    block_colors = ["#eaf1f8", "#fdf2e9", "#edf7ed"]

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                     loc="center",
                     colWidths=[0.14, 0.30, 0.06, 0.13, 0.17, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#c9c9c9")
        if r == 0:
            cell.set_facecolor("#3b5a77")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.16)
            continue
        block = (r - 1) // 3
        cell.set_facecolor(block_colors[block])
        if c == 0 and rows[r - 1][0]:
            cell.set_text_props(fontweight="bold")
        if c == 4:
            cell.set_text_props(fontweight="bold")
        if rows[r - 1][1] == "Broader states":
            cell.set_text_props(color="#8a3510",
                                fontweight="bold" if c in (3, 4) else "normal")

    ax.set_title("Circumplex plane share: classic Russell anchors vs broader "
                 "affective states\n(plane share = fraction of a lemma centroid's "
                 "squared displacement explained by the PC1+PC2 circumplex plane)",
                 fontsize=12, pad=18)
    fig.text(0.5, 0.045,
             "Chance level for a random 2D plane in 2560 dims: 0.08%.  In-sample "
             "anchors are inflated (PCA was fit on them); held-out (leave-one-out) "
             "anchors\nare the fair yardstick. Mann–Whitney compares held-out "
             "anchors vs broader states.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = PACKAGE / "figures/russell_plane_share_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
