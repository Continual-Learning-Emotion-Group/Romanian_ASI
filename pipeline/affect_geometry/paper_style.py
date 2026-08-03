"""Shared figure styling for paper figures, following the project's
Figure/Table Formatting Guidelines: scienceplots 'science' style (no-latex so
no TeX install is needed), Tol bright_extended color cycle, and a legend
helper (in-plot legends get a border and a semi-transparent white patch).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

BRIGHT_EXTENDED = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # magenta
    "#BBBBBB",  # gray
    "#000000",  # black
    "#EE7733",  # orange
    "#33BBEE",  # lighter cyan
    "#CC3311",  # deep red
    "#0077BB",  # strong blue
]

BLUE, RED, GREEN, YELLOW = BRIGHT_EXTENDED[:4]


def apply_style():
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
    plt.rcParams["axes.prop_cycle"] = cycler(color=BRIGHT_EXTENDED)


def inplot_legend(ax, **kwargs):
    defaults = dict(frameon=True, framealpha=0.6, facecolor="white",
                    edgecolor="0.4", fancybox=False)
    defaults.update(kwargs)
    legend = ax.legend(**defaults)
    legend.get_frame().set_linewidth(0.6)
    return legend
