
#!/usr/bin/env python3
"""Plot subject-level SSIM/PSNR distributions with Bonferroni post-hoc p-values.
Run MCFormer_subject_level_ANOVA_Bonferroni.py first.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["Corrupted", "MC-Net", "Restormer", "MC-Former"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject_csv", type=Path, default=Path(__file__).resolve().parent / "MCFormer_subject_level_results.csv")
    p.add_argument("--posthoc_csv", type=Path, default=Path(__file__).resolve().parent / "MCFormer_subject_level_Bonferroni.csv")
    p.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = p.parse_args()
    subjects = pd.read_csv(args.subject_csv)
    posthoc = pd.read_csv(args.posthoc_csv)
    required = {"Metric", "Method_A", "Method_B", "p_Bonferroni"}
    if not required.issubset(posthoc.columns):
        raise ValueError(f"Post-hoc file missing columns: {required - set(posthoc.columns)}")
    if subjects.subject_id.duplicated().any():
        raise ValueError("Duplicate subject IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("SSIM", "PSNR"):
        arrays = [subjects[f"{m}_{metric}"].to_numpy(dtype=float) for m in METHODS]
        if not np.isfinite(np.concatenate(arrays)).all():
            raise ValueError("Nonfinite metric values")

        fig, ax = plt.subplots(figsize=(9, 6))
        violins = ax.violinplot(arrays, showmedians=True, showextrema=False)
        color = "#2672A0" if metric == "SSIM" else "#B84B49"

        for item in violins["bodies"]:
            item.set_facecolor(color)
            item.set_edgecolor("black")
            item.set_alpha(.65)

        violins["cmedians"].set_color("black")

        for i, a in enumerate(arrays, 1):
            ax.scatter(
                np.full(len(a), i), a,
                s=12, alpha=.5, color="black", zorder=3
            )

        ax.set_xticks(range(1, 5), METHODS)
        ax.set_ylabel(
            "Subject-level mean " + metric +
            (" (dB)" if metric == "PSNR" else "")
        )
        ax.set_title(
            f"MC-Former subject-level {metric} "
            f"(n={len(subjects)} subjects)"
        )

        # Display MC-Former comparisons with Bonferroni p-values.
        # Bonferroni adjustment includes all six pairs per metric.
        lo, hi = min(map(np.min, arrays)), max(map(np.max, arrays))
        span = max(hi - lo, 1e-6)

        for j, baseline in enumerate(METHODS[:-1]):
            match = posthoc[
                (posthoc.Metric == metric) &
                (
                    (
                        (posthoc.Method_A == baseline) &
                        (posthoc.Method_B == "MC-Former")
                    ) |
                    (
                        (posthoc.Method_B == baseline) &
                        (posthoc.Method_A == "MC-Former")
                    )
                )
            ]

            if len(match) != 1:
                raise ValueError(
                    f"Missing Bonferroni comparison: {metric} {baseline}"
                )

            val = float(match.iloc[0]["p_Bonferroni"])
            y = hi + (j + 1) * span * .13

            ax.plot(
                [j + 1, j + 1, 4, 4],
                [y, y + span * .02, y + span * .02, y],
                color="black", linewidth=1
            )

            label = (
                "p < 0.001" if val < .001 else f"p = {val:.3g}"
            )
            ax.text(
                (j + 1 + 4) / 2,
                y + span * .025,
                label,
                ha="center",
                fontsize=10
            )

        ax.set_ylim(lo - span * .1, hi + span * .55)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()

        path = (
            args.output_dir /
            f"MCFormer_subject_level_violin_{metric}_Bonferroni.png"
        )
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved", path)


if __name__ == "__main__":
    main()

