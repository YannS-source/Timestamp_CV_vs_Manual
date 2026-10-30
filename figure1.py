"""Figure 1: the four-panel composite, as vector files (PDF, EPS, SVG).

Layout: A and B on top (Bland-Altman of each method against ground truth,
patient entry and exit), C/D below (MAE heatmap by shift and subspecialty,
manual entry half and computer vision half). Drawn directly in matplotlib:
no rasterized layers, heatmap and colorbar drawn as vector rectangles,
TrueType (Type 42) fonts.

EPS does not support transparency, so in the EPS the translucent colors are
replaced by their solid equivalents on white; PDF and SVG keep true
transparency.

Outputs: figures/figure1.pdf, figures/figure1.eps, figures/figure1.svg,
figures/figure1.png (preview) and figures/figure1.tiff (600 dpi raster).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from data_utils import load_data

OUT = Path(__file__).resolve().parent / "figures"

CV_C = "#1E8CEB"
MAN_C = "#D7301F"

# Heatmap rows: the three largest specialties, then all others pooled.
SPEC_COL = "Collapsed_Specialties"
MAIN_SPECS = ["General", "Urology", "Gynecology"]
OTHER = "Other specialties"

# Shift blocks by the event's own clock hour.
SHIFTS = [("AM", lambda h: (h >= 7) & (h < 15)),
          ("PM", lambda h: (h >= 15) & (h < 23)),
          ("Overnight", lambda h: (h < 7) | (h >= 23))]
EVENTS = [("Entry", "hour_in", "in"), ("Exit", "hour_out", "out")]
METHODS = [("Manual", "emr"), ("CV", "cv")]

CMAP = LinearSegmentedColormap.from_list(
    "mae", ["#FFFFFF", "#FEE6CE", "#FDAE6B", "#E6550D", "#A63603"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.hashsalt": "figure1",
    "savefig.bbox": "tight",
})


def fade(color, alpha, flatten):
    """Translucent color, or its solid equivalent on white if flatten."""
    if not flatten:
        return mcolors.to_rgba(color, alpha)
    rgb = mcolors.to_rgba(color)[:3]
    return tuple(alpha * c + (1 - alpha) for c in rgb) + (1.0,)


# ---------------------------------------------------------------- A and B
def ba_panel(ax, tod_hr, d_cv_min, d_man_min, title, flatten):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # small horizontal offset so the two methods' points do not overlap
    for d, c, dx in [(d_man_min, MAN_C, +0.06), (d_cv_min, CV_C, -0.06)]:
        ax.scatter(tod_hr + dx, d, color=fade(c, 0.45, flatten), s=24,
                   edgecolor="none", zorder=2)
        bias, sd = d.mean(), d.std(ddof=1)
        for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"),
                      (bias - 1.96 * sd, "--")]:
            ax.axhline(y, color=fade(c, 0.8, flatten), lw=1.1, ls=ls, zorder=1)
    ax.axhline(0, color="0.4", lw=0.8, zorder=0)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_ylim(-7, 7)
    ax.set_xlabel("Time of day of event (h)")
    ax.set_ylabel("Difference from ground truth (minutes)")
    ax.set_title(title)
    ax.grid(color="0.85", lw=0.5)
    ax.set_axisbelow(True)

    n = len(d_cv_min)
    w_cv = int((np.abs(d_cv_min) <= 1.0).sum())
    w_man = int((np.abs(d_man_min) <= 1.0).sum())
    ax.text(0.02, 0.03,
            f"Within ±1 min of ground truth:\n"
            f"CV {w_cv}/{n} ({w_cv / n:.0%}),  Manual {w_man}/{n} ({w_man / n:.0%})",
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=10.5, color="0.25",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "edgecolor": "0.85", "lw": 0.8})
    handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=CV_C,
               markersize=8, label="Computer vision"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=MAN_C,
               markersize=8, label="Manual entry"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5)


# ---------------------------------------------------------------- C/D
def heatmap_panel(ax, cax, m, flatten):
    rows = [(s, m[SPEC_COL] == s) for s in MAIN_SPECS]
    rows.append((OTHER, ~m[SPEC_COL].isin(MAIN_SPECS)))
    row_labels = [r[0] for r in rows]

    cols = []
    for me_name, me_key in METHODS:
        for sh_name, sh_fn in SHIFTS:
            for ev_name, hour_col, ev_key in EVENTS:
                cols.append((me_name, me_key, sh_name, sh_fn, ev_name,
                             hour_col, ev_key))

    # cell = MAE (seconds) of one method, one event, one shift, one row
    M = np.full((len(rows), len(cols)), np.nan)
    N = np.zeros_like(M, dtype=int)
    for i, (_, row_sel) in enumerate(rows):
        sub = m[row_sel]
        for j, (_, me_key, _, sh_fn, _, hour_col, ev_key) in enumerate(cols):
            sel = sh_fn(sub[hour_col].values)
            N[i, j] = int(sel.sum())
            if N[i, j]:
                M[i, j] = np.abs(sub[f"{me_key}_err_{ev_key}"].values[sel]).mean()

    # color scale capped at the 95th percentile (at least 60 s); cells above
    # the cap take the darkest color, and every cell shows its value
    vmin = float(np.nanmin(M))
    vmax = max(float(np.nanpercentile(M[np.isfinite(M)], 95)), 60.0)
    norm = Normalize(vmin=vmin, vmax=vmax)
    vrange = vmax - vmin if vmax > vmin else 1.0

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", color="0.55",
                        fontsize=10)
                continue
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=CMAP(norm(min(v, vmax))), lw=0))
            tc = "white" if (v - vmin) / vrange > 0.55 else "0.10"
            ax.text(j, i - 0.13, f"{v:.0f}", ha="center", va="center",
                    color=tc, fontsize=10, fontweight="bold")
            ax.text(j, i + 0.26, f"n={N[i, j]}", ha="center", va="center",
                    color=tc, fontsize=7.5)

    n_cols = len(cols)
    half = n_cols // 2
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(len(rows) - 0.5, -3.9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=10.5)
    ax.set_xticks([])

    # three header rows above the cells: event, shift, method
    yh1, yh2, yh3 = -0.8, -1.6, -2.4
    for j, c in enumerate(cols):
        ax.text(j, yh1, c[4], ha="center", va="center", fontsize=9,
                color="0.25")
    for k in range(0, n_cols, 2):
        ax.text(k + 0.5, yh2, cols[k][2], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="0.10")
        ax.plot([k - 0.5, k + 1.5], [yh2 + 0.35, yh2 + 0.35],
                color="0.55", lw=0.6)
    for k, (me_name, _) in enumerate(METHODS):
        x0 = k * half - 0.5
        color = MAN_C if me_name == "Manual" else CV_C
        ax.text(x0 + half / 2, yh3,
                "Manual EHR" if me_name == "Manual" else "Computer Vision",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=color)
        ax.add_patch(Rectangle((x0, yh3 - 0.35), half, 0.7,
                               facecolor=fade(color, 0.08, flatten), lw=0))
    ax.axvline(half - 0.5, color="0.20", lw=1.5)
    for k in range(2, n_cols, 2):
        if k == half:
            continue
        ax.axvline(k - 0.5, color="0.65", lw=0.6, ls="--")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # colorbar drawn as stacked rectangles, so it stays vector
    n_seg = 256
    ys = np.linspace(vmin, vmax, n_seg + 1)
    for k in range(n_seg):
        cax.add_patch(Rectangle((0, ys[k]), 1, ys[k + 1] - ys[k],
                                facecolor=CMAP(norm(ys[k])), lw=0))
    cax.set_xlim(0, 1)
    cax.set_ylim(vmin, vmax)
    cax.set_xticks([])
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Mean absolute error vs ground truth (s)", fontsize=10)
    cax.tick_params(labelsize=10)
    for s in cax.spines.values():
        s.set_color("0.4")
        s.set_linewidth(0.6)


def build(flatten: bool = False):
    m = load_data()

    def tod(col):
        return (m[col].dt.hour + m[col].dt.minute / 60).values

    fig = plt.figure(figsize=(14.2, 12.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08],
                          hspace=0.20, wspace=0.22,
                          left=0.10, right=0.90, top=0.95, bottom=0.03)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    ba_panel(axA, tod("rs_wheels_in"),
             m["cv_err_in"].values / 60.0, m["emr_err_in"].values / 60.0,
             "Patient Entry", flatten)
    ba_panel(axB, tod("rs_wheels_out"),
             m["cv_err_out"].values / 60.0, m["emr_err_out"].values / 60.0,
             "Patient Exit", flatten)
    LET_OFF = -0.07  # panel letter position, as a fraction of panel width
    for ax, letter in [(axA, "A"), (axB, "B")]:
        ax.text(LET_OFF, 1.04, letter, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="bottom", ha="left")

    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[40, 1], wspace=0.03)
    axH = fig.add_subplot(gsb[0, 0])
    axC = fig.add_subplot(gsb[0, 1])
    heatmap_panel(axH, axC, m, flatten)

    # C and D at the same horizontal position as A and B
    posH = axH.get_position()
    hx0, hx1 = axH.get_xlim()

    def heat_x(fig_x):
        return hx0 + (fig_x - posH.x0) / posH.width * (hx1 - hx0)

    for src_ax, letter in [(axA, "C"), (axB, "D")]:
        pos = src_ax.get_position()
        fig_x = pos.x0 + LET_OFF * pos.width
        axH.text(heat_x(fig_x), -3.35, letter, fontsize=22,
                 fontweight="bold", va="center", ha="left")
    return fig


def main() -> None:
    OUT.mkdir(exist_ok=True)

    fig = build()
    fig.savefig(OUT / "figure1.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "figure1.svg", metadata={"Date": None})
    fig.savefig(OUT / "figure1.png", dpi=200)
    fig.savefig(OUT / "figure1.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # the EPS writer always stamps a date; fix it so reruns are identical
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1790035200")  # 2026-09-22
    fig = build(flatten=True)
    fig.savefig(OUT / "figure1.eps")
    plt.close(fig)

    for f in ["figure1.pdf", "figure1.eps", "figure1.svg", "figure1.png",
              "figure1.tiff"]:
        print(f"[written] {OUT / f}")


if __name__ == "__main__":
    main()
