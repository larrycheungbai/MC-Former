#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MC-Former: subject-level paired Wilcoxon signed-rank sensitivity analysis.

Input (in the same directory as this script):
    MCFormer_subject_level_results.csv

Outputs:
    MCFormer_subject_level_Wilcoxon_Bonferroni.csv
    MCFormer_subject_level_Wilcoxon_notes.txt

Analysis: 12 independent subjects; four methods; all six paired comparisons
for SSIM and PSNR separately. Two-sided Wilcoxon signed-rank tests with
Bonferroni correction within each metric (six comparisons per metric).
Includes matched-pairs rank-biserial correlation (direction A minus B).

Requires: numpy, pandas, scipy. No pingouin required.
"""

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Locate data relative to the script so Spyder --wdir does not matter.
BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "MCFormer_subject_level_results.csv"
OUTPUT_FILE = BASE / "MCFormer_subject_level_Wilcoxon_Bonferroni.csv"
NOTES_FILE = BASE / "MCFormer_subject_level_Wilcoxon_notes.txt"

METHODS = ["Corrupted", "MC-Net", "Restormer", "MC-Former"]
METRICS = ["SSIM", "PSNR"]
ALPHA = 0.05
PAIRS = list(combinations(METHODS, 2))
N_TESTS_PER_METRIC = len(PAIRS)  # 6


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon test and paired rank-biserial effect size.

    Differences are A minus B. Zero differences are excluded (wilcox).
    rank-biserial = (sum positive ranks - sum negative ranks) / total ranks.
    The value is in [-1, 1], with positive values favoring A.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Inputs must be matching one-dimensional subject arrays.")
    d = a - b
    if not np.isfinite(d).all():
        raise ValueError("NaN or infinite paired difference.")

    nonzero = d[d != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0, 0.0, 0.0, 0, len(d)

    # Passing differences directly avoids duplicate roundoff from separate arrays.
    result = stats.wilcoxon(
        nonzero, zero_method="wilcox", alternative="two-sided", method="auto"
    )
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    rank_biserial = (w_plus - w_minus) / (w_plus + w_minus)
    return (float(result.statistic), float(result.pvalue),
            float(rank_biserial), w_plus, len(nonzero), len(d) - len(nonzero))


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}\n"
            "Place this script alongside MCFormer_subject_level_results.csv."
        )

    df = pd.read_csv(INPUT_FILE)
    required = ["subject_id"] + [
        f"{method}_{metric}" for metric in METRICS for method in METHODS
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    if df["subject_id"].isna().any() or df["subject_id"].duplicated().any():
        raise ValueError("Subject IDs must be present and unique.")

    for column in required[1:]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite values in {column}")

    n = len(df)
    if n < 3:
        raise ValueError(f"Too few subjects: {n}")
    if n != 12:
        print(f"NOTE: expected 12 test subjects, found {n}.")

    print(f"Loaded {n} independent subjects: {df['subject_id'].tolist()}")
    print("Six two-sided paired Wilcoxon tests per metric; Bonferroni family size = 6")
    rows = []
    for metric in METRICS:
        print(f"\n{metric}")
        for a_name, b_name in PAIRS:
            a = df[f"{a_name}_{metric}"].to_numpy(dtype=float)
            b = df[f"{b_name}_{metric}"].to_numpy(dtype=float)
            w, p, r_rb, w_plus, n_nonzero, n_zero = paired_wilcoxon(a, b)
            p_adj = min(p * N_TESTS_PER_METRIC, 1.0)
            row = {
                "Metric": metric,
                "Method_A": a_name,
                "Method_B": b_name,
                "N_subjects": n,
                "N_nonzero_differences": n_nonzero,
                "N_zero_differences": n_zero,
                "Mean_A": float(np.mean(a)),
                "Mean_B": float(np.mean(b)),
                "Mean_Difference_A_minus_B": float(np.mean(a - b)),
                "Median_Difference_A_minus_B": float(np.median(a - b)),
                "Wilcoxon_W_min_rank_sum": w,
                "Wilcoxon_W_plus": w_plus,
                "p_raw_two_sided": p,
                "p_Bonferroni": p_adj,
                "Rank_Biserial_A_minus_B": r_rb,
                "Significant_adjusted_p_below_0_05": p_adj < ALPHA,
                "Bonferroni_family_size": N_TESTS_PER_METRIC,
            }
            rows.append(row)
            print(f"  {a_name:10s} vs {b_name:10s} "
                  f"W={w:5.1f} p={p:.4g} adj.p={p_adj:.4g} "
                  f"r_rb={r_rb:+.3f}")

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, float_format="%.12g")
    NOTES_FILE.write_text(
        "MC-Former: subject-level Wilcoxon signed-rank sensitivity analysis\n"
        f"Independent subjects: {n}\n"
        "Within-subject methods: Corrupted, MC-Net, Restormer, MC-Former.\n"
        "Outcomes SSIM and PSNR analyzed as SEPARATE six-comparison families.\n"
        "Two-sided paired Wilcoxon tests; zero differences excluded (wilcox).\n"
        "SciPy method='auto' chooses an appropriate p-value calculation; "
        "ties may affect exactness.\n"
        "Bonferroni-adjusted p = min(raw p * 6, 1); alpha=0.05.\n"
        "Difference sign and rank-biserial effect size use Method_A minus Method_B.\n"
        "Positive rank-biserial effect size favors Method_A.\n"
        "These tests are a sensitivity analysis of the subject-level paired t-tests; "
        "they do not independently validate the ANOVA normality or sphericity assumptions.\n"
        "Bonferroni here controls six comparisons within each metric, not across both metrics.\n",
        encoding="utf-8"
    )
    print(f"\nSUCCESS: {OUTPUT_FILE}\nNotes: {NOTES_FILE}")


if __name__ == "__main__":
    main()
