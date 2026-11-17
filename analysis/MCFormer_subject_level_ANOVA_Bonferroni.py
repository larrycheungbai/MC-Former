#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MC-Former: subject-level repeated-measures ANOVA + Bonferroni post-hoc tests.

Input: MCFormer_subject_level_results.csv (subject_id and columns such as
       Corrupted_SSIM, MC-Net_SSIM, Restormer_SSIM, MC-Former_SSIM, etc.)
Outputs (in the same directory as this script unless paths are edited):
    MCFormer_subject_level_ANOVA.csv
    MCFormer_subject_level_Bonferroni.csv

Four repeated conditions, N = independent subjects (not slices).
Six paired two-sided t-tests per outcome, Bonferroni-adjusted within each
outcome. If both PSNR and SSIM are treated as one family of hypotheses,
account for that additional multiplicity in the manuscript.
Dependencies: numpy, pandas, scipy. No pingouin needed.
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Use the folder containing this script so Spyder's working directory does not matter.
BASE_DIR = Path(__file__).resolve().parent
input_file = BASE_DIR / "MCFormer_subject_level_results.csv"
anova_output_file = BASE_DIR / "MCFormer_subject_level_ANOVA.csv"
posthoc_output_file = BASE_DIR / "MCFormer_subject_level_Bonferroni.csv"

methods = ["Corrupted", "MC-Net", "Restormer", "MC-Former"]
metrics = ["SSIM", "PSNR"]


def repeated_measures_anova(Y):
    """One-factor repeated-measures ANOVA, with GG-adjusted p and partial eta squared."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must have dimensions subjects x methods")
    n, k = Y.shape
    if n < 3 or k < 2 or not np.isfinite(Y).all():
        raise ValueError("Insufficient subjects/methods or non-finite values")

    grand_mean = Y.mean()
    method_means = Y.mean(axis=0)
    subject_means = Y.mean(axis=1)
    ss_method = n * np.sum((method_means - grand_mean) ** 2)
    residuals = Y - subject_means[:, None] - method_means[None, :] + grand_mean
    ss_error = np.sum(residuals ** 2)
    df_method, df_error = k - 1, (n - 1) * (k - 1)
    ms_error = ss_error / df_error
    if ms_error <= 0:
        raise ValueError("Residual variance must be positive")
    F = (ss_method / df_method) / ms_error
    p_unc = stats.f.sf(F, df_method, df_error)
    partial_eta_squared = ss_method / (ss_method + ss_error)

    S = np.cov(Y, rowvar=False, ddof=1)
    C = np.eye(k) - np.ones((k, k)) / k
    V = C @ S @ C
    denom = (k - 1) * np.trace(V @ V)
    if denom <= 0:
        raise ValueError("Cannot calculate Greenhouse-Geisser epsilon")
    epsilon_GG = float(np.clip(np.trace(V) ** 2 / denom, 1 / (k - 1), 1))
    df1_GG, df2_GG = epsilon_GG * df_method, epsilon_GG * df_error
    p_GG = stats.f.sf(F, df1_GG, df2_GG)
    return {
        "N_subjects": n, "N_methods": k,
        "df1": df_method, "df2": df_error, "F": F,
        "p_unc": p_unc, "GG_epsilon": epsilon_GG,
        "GG_df1": df1_GG, "GG_df2": df2_GG,
        "p_GG_corr": p_GG, "partial_eta_squared": partial_eta_squared,
    }


def bonferroni_pairwise(Y, method_names, metric):
    """All six paired comparisons; Bonferroni across six comparisons per metric."""
    n, k = Y.shape
    pairs = list(combinations(range(k), 2))
    m = len(pairs)
    results = []
    for a, b in pairs:
        x, y = Y[:, a], Y[:, b]
        difference = x - y
        sd_diff = float(np.std(difference, ddof=1))
        mean_diff = float(np.mean(difference))
        if sd_diff == 0:
            if mean_diff == 0:
                t_val, p_raw, dz = 0.0, 1.0, 0.0
            else:
                t_val = float(np.copysign(np.inf, mean_diff))
                p_raw = 0.0
                dz = float(np.copysign(np.inf, mean_diff))
        else:
            test = stats.ttest_rel(x, y, alternative="two-sided")
            t_val, p_raw = float(test.statistic), float(test.pvalue)
            dz = mean_diff / sd_diff
        p_adj = min(1.0, m * p_raw)
        results.append({
            "Metric": metric,
            "Method_A": method_names[a],
            "Method_B": method_names[b],
            "N_subjects": n,
            "Mean_A": float(np.mean(x)),
            "Mean_B": float(np.mean(y)),
            "Mean_Difference_A_minus_B": mean_diff,
            "SD_of_Paired_Differences": sd_diff,
            "t_statistic": t_val,
            "df": n - 1,
            "p_raw": p_raw,
            "p_Bonferroni": p_adj,
            "Cohens_dz": dz,
            "Significant_at_0.05": bool(p_adj < 0.05),
            "Bonferroni_family_size": m,
        })
    return results


def main():
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Cannot find {input_file}\nPlace the input CSV beside this script or change input_file."
        )
    subject_df = pd.read_csv(input_file)
    if "subject_id" not in subject_df.columns:
        raise ValueError("CSV must contain a subject_id column")
    subject_df = subject_df.set_index("subject_id")
    if subject_df.index.has_duplicates:
        raise ValueError("Duplicate subject IDs detected")

    expected = [f"{method}_{metric}" for metric in metrics for method in methods]
    missing = [c for c in expected if c not in subject_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(subject_df) != 12:
        print(f"WARNING: Found {len(subject_df)} subjects (expected 12 for original test set).")
    print("Loaded:", input_file)
    print("Independent subjects:", len(subject_df))

    anova_results, posthoc_results = [], []
    for metric in metrics:
        columns = [f"{method}_{metric}" for method in methods]
        Y = subject_df[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(Y).all():
            raise ValueError(f"Missing/non-finite values in {metric}")
        omnibus = repeated_measures_anova(Y)
        anova_results.append({"Metric": metric, **omnibus})
        posthoc_results.extend(bonferroni_pairwise(Y, methods, metric))

        print(f"\n=== {metric} ===")
        print(f"Repeated-measures ANOVA: F({omnibus['df1']}, {omnibus['df2']})="
              f"{omnibus['F']:.4f}; p_GG={omnibus['p_GG_corr']:.6g}; "
              f"partial eta squared={omnibus['partial_eta_squared']:.4f}")
        print("Bonferroni-corrected paired t-tests (six pairs):")
        for r in posthoc_results[-6:]:
            print(f"  {r['Method_A']:10s} vs {r['Method_B']:10s}: "
                  f"difference={r['Mean_Difference_A_minus_B']:.5f}, "
                  f"p_raw={r['p_raw']:.4g}, p_adj={r['p_Bonferroni']:.4g}, "
                  f"dz={r['Cohens_dz']:.3f}")

    pd.DataFrame(anova_results).to_csv(anova_output_file, index=False)
    pd.DataFrame(posthoc_results).to_csv(posthoc_output_file, index=False)
    print("\nSaved ANOVA:", anova_output_file)
    print("Saved post-hoc:", posthoc_output_file)
    print("Note: paired t-tests assume approximately normal SUBJECT-LEVEL paired differences.")
    print("Bonferroni adjustment is applied separately within SSIM and PSNR (6 comparisons each).")


if __name__ == "__main__":
    main()
