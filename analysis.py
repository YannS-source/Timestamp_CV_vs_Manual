"""Every number reported in the paper, in the order it appears.

Writes results.txt. P values are formatted in AMA style: P < .001, three
decimals below .01, two decimals otherwise.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy import stats

from data_utils import load_data

N_BOOT = 10_000
SEED = 1234
MARGIN = 30.0  # seconds, prespecified noninferiority margin


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "P < .001"
    return "P = " + (f"{p:.3f}" if p < 0.01 else f"{p:.2f}").lstrip("0")


def overnight(hour: np.ndarray) -> np.ndarray:
    """Overnight shift block of Figure 1C/D (11 PM to 7 AM)."""
    return (hour < 7) | (hour >= 23)


def main() -> None:
    m = load_data()
    n = len(m)
    err = {("CV", "entry"): m["cv_err_in"].values,
           ("CV", "exit"): m["cv_err_out"].values,
           ("Manual", "entry"): m["emr_err_in"].values,
           ("Manual", "exit"): m["emr_err_out"].values}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Timestamp validation: results (n = {n} cases)")
        print("=" * 70)

        print("\nTable. Cases by surgical specialty")
        for spec, k in m["Collapsed_Specialties"].value_counts().items():
            print(f"  {spec:28s} {k:4d} ({100 * k / n:.1f})")

        print("\nMean absolute error (MAE) against ground truth, seconds")
        for ev in ["entry", "exit"]:
            print(f"  {ev:5s}: CV {np.abs(err[('CV', ev)]).mean():.1f} "
                  f"vs manual {np.abs(err[('Manual', ev)]).mean():.1f}")

        print("\nWithin 1 minute of ground truth (Figure, A and B)")
        for ev in ["entry", "exit"]:
            for meth in ["CV", "Manual"]:
                w = int((np.abs(err[(meth, ev)]) <= 60).sum())
                print(f"  {ev:5s} {meth:6s}: {w}/{n} ({w / n:.0%})")

        print("\nNoninferiority of CV to manual entry (primary analysis)")
        print("  (paired bootstrap: cases resampled with replacement, keeping "
              "each case's CV and manual errors together)")
        for ev in ["entry", "exit"]:
            a_cv = np.abs(err[("CV", ev)])
            a_man = np.abs(err[("Manual", ev)])
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, n, size=(N_BOOT, n))
            boot = a_cv[idx].mean(axis=1) - a_man[idx].mean(axis=1)
            ub = float(np.percentile(boot, 95))
            p = stats.wilcoxon(a_cv - a_man - MARGIN, alternative="less",
                               zero_method="wilcox").pvalue
            print(f"  {ev:5s}: MAE difference {a_cv.mean() - a_man.mean():+.1f} s; "
                  f"1-sided 95% upper bound {ub:+.1f} s (margin {MARGIN:.0f} s); "
                  f"supportive 1-sided Wilcoxon {fmt_p(p)}")

        print("\nBias against ground truth, seconds (Bland-Altman; "
              "bias tested with a 1-sample Wilcoxon test)")
        for ev in ["entry", "exit"]:
            for meth in ["CV", "Manual"]:
                x = err[(meth, ev)]
                bias, sd = x.mean(), x.std(ddof=1)
                p = stats.wilcoxon(x, zero_method="wilcox").pvalue
                print(f"  {ev:5s} {meth:6s}: bias {bias:+.1f} (rounded {bias:+.0f}), "
                      f"{fmt_p(p)}; 95% limits of agreement "
                      f"{bias - 1.96 * sd:+.1f} to {bias + 1.96 * sd:+.1f}")

        # Hour of day is the clock hour of patient entry (computer vision
        # timestamp), for entry and exit errors alike.
        print("\nTrend of absolute error by hour of day (Spearman, hour of "
              "patient entry)")
        hr = m["hour_in"].values
        for ev in ["entry", "exit"]:
            for meth in ["CV", "Manual"]:
                r = stats.spearmanr(hr, np.abs(err[(meth, ev)]))
                print(f"  {ev:5s} {meth:6s}: rho = {r.statistic:.2f}, "
                      f"{fmt_p(r.pvalue)}")

        night = overnight(hr)
        a_cv = np.abs(err[("CV", "entry")])
        a_man = np.abs(err[("Manual", "entry")])
        p = stats.mannwhitneyu(a_man[night], a_man[~night]).pvalue
        print(f"\nOvernight subgroup at entry (n = {int(night.sum())}): manual MAE "
              f"{a_man[night].mean():.1f} s vs CV {a_cv[night].mean():.1f} s")
        print(f"  manual error overnight vs other shifts, Mann-Whitney {fmt_p(p)}")

        print("\nExit timing relative to each method's own entry timing "
              "(Discussion)")
        print("  (exit error minus entry error per case; negative = exit "
              "marked earlier; Wilcoxon)")
        for meth in ["CV", "Manual"]:
            d = err[(meth, "exit")] - err[(meth, "entry")]
            p = stats.wilcoxon(d, zero_method="wilcox").pvalue
            print(f"  {meth:6s}: mean {d.mean():+.1f} s, {fmt_p(p)}")

    txt = buf.getvalue()
    print(txt)
    out = Path(__file__).resolve().parent / "results.txt"
    out.write_text(txt, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
