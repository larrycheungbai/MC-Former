#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MC-Former: Shapiro-Wilk normality checks on subject-level paired differences.

Input (placed in same folder as this script):
    MCFormer_subject_level_results.csv

Outputs (saved beside the input CSV):
    MCFormer_subject_level_Shapiro_Wilk.csv
    MCFormer_subject_level_Shapiro_Wilk_notes.txt

Requires: numpy, pandas, scipy

Purpose: assess the normality assumption of the six paired t-tests for each
metric. This is NOT a test of repeated-measures ANOVA sphericity, nor does a
non-significant Shapiro-Wilk result establish normality.
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Resolve paths relative to the script file, to work reliably with Spyder F5.
BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "MCFormer_subject_level_results.csv"
OUTPUT_FILE = BASE / "MCFormer_subject_level_Shapiro_Wilk.csv"
NOTES_FILE = BASE / "MCFormer_subject_level_Shapiro_Wilk_notes.txt"

METHODS = ("Corrupted", "MC-Net", "Restormer", "MC-Former")
METRICS = ("SSIM", "PSNR")
ALPHA = 0.05


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Missing input CSV: {INPUT_FILE}\n"
            "Put this script in the same folder as MCFormer_subject_level_results.csv "
            "or edit INPUT_FILE at the top of the script."
        )

    df = pd.read_csv(INPUT_FILE)
    if "subject_id" not in df.columns:
        raise ValueError("Expected a 'subject_id' column in the input CSV.")
    if df["subject_id"].isna().any() or df["subject_id"].duplicated().any():
        raise ValueError("Subject IDs must be unique and nonmissing.")
    df = df.set_index("subject_id")

    expected = [f"{method}_{metric}" for metric in METRICS for method in METHODS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) < 3:
        raise ValueError("Shapiro-Wilk testing requires at least 3 subjects.")
    if len(df) != 12:
        print(f"WARNING: Found {len(df)} subjects; expected 12 in the original test set.")

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite value detected in {col}")

    results = []
    print(f"Subject-level Shapiro-Wilk normality tests: N={len(df)} subjects")
    print("Tests are performed on within-subject paired differences.\n")

    for metric in METRICS:
        print(f"{metric}:")
        for first, second in combinations(METHODS, 2):
            # The sign does not affect the Shapiro-Wilk result.
            diff = (df[f"{second}_{metric}"] - df[f"{first}_{metric}"]).to_numpy(dtype=float)
            sd = float(np.std(diff, ddof=1))
            if sd == 0:
                w, p = np.nan, np.nan
                conclusion = "constant differences; test undefined"
            else:
                test = stats.shapiro(diff)
                w, p = float(test.statistic), float(test.pvalue)
                conclusion = (
                    "evidence against normality" if p < ALPHA
                    else "normality not rejected"
                )
            row = {
                "Metric": metric,
                "Comparison": f"{second} minus {first}",
                "N_subjects": len(diff),
                "Mean_paired_difference": float(np.mean(diff)),
                "SD_paired_difference": sd,
                "Shapiro_W": w,
                "Shapiro_p": p,
                "Alpha": ALPHA,
                "Interpretation": conclusion,
            }
            results.append(row)
            print(f"  {row['Comparison']:<31} W={w:.4f}, p={p:.5g}  {conclusion}")
        print()

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, float_format="%.10g")

    flagged = result_df[result_df["Shapiro_p"] < ALPHA]
    with NOTES_FILE.open("w", encoding="utf-8") as fh:
        fh.write("MC-Former subject-level Shapiro-Wilk normality diagnostics\n")
        fh.write(f"N subjects: {len(df)}\n")
        fh.write("Tests: six within-subject method-pair differences, separately for SSIM and PSNR.\n")
        fh.write(f"Unadjusted Shapiro-Wilk screening threshold: alpha={ALPHA}.\n")
        fh.write(f"Pairs with p<0.05: {len(flagged)} of {len(result_df)}.\n")
        fh.write("A p-value >= 0.05 does not prove that the differences are normally distributed.\n")
        fh.write("Small samples (n=12) limit the power of normality tests.\n")
        fh.write("Shapiro-Wilk on paired differences is diagnostic for paired t-test assumptions.\n")
        fh.write("This does not assess repeated-measures ANOVA sphericity; GG epsilon addresses\n")
        fh.write("potential sphericity violations, but is not a normality correction.\n")
        fh.write("For p<0.05 pairs, inspect the paired differences and consider a Wilcoxon\n")
        fh.write("signed-rank sensitivity analysis with multiplicity correction.\n")
        fh.write("The six paired t-test p-values must still receive Bonferroni correction,\n")
        fh.write("as in the separately generated Bonferroni analysis.\n")
    print(f"Saved results: {OUTPUT_FILE}")
    print(f"Saved interpretation notes: {NOTES_FILE}")
    print(f"Pairs with Shapiro p < {ALPHA}: {len(flagged)}/{len(result_df)}")


if __name__ == "__main__":
    main()
