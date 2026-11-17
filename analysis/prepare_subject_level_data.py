#!/usr/bin/env python3
"""Aggregate matched image-level SSIM/PSNR results to subject-level means.

Input: four whitespace-separated text files, without headers, containing
filename SSIM PSNR. Filename must contain a subject identifier matched by
--subject_regex (default: Dicom(\\d+)MemEffMotion).
Output: MCFormer_subject_level_results.csv, one row per independent subject.
No patient images or confidential data are included in this script.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = {"Corrupted": "corrupted", "MC-Net": "mc_net", "Restormer": "restormer", "MC-Former": "mc_former"}


def read_metrics(path, method, pattern):
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape[1] != 3:
        raise ValueError(f"{method}: expected 3 columns (filename SSIM PSNR)")
    data.columns = ["filename", "SSIM", "PSNR"]
    data["filename"] = data["filename"].astype(str)
    if data["filename"].duplicated().any():
        raise ValueError(f"{method}: duplicate image filenames")
    ids = data["filename"].str.extract(pattern, expand=False)
    if isinstance(ids, pd.DataFrame):
        raise ValueError("Subject regex must contain exactly one capturing group")
    if ids.isna().any():
        raise ValueError(f"{method}: could not extract subject IDs from some filenames")
    data["subject_id"] = ids
    for metric in ("SSIM", "PSNR"):
        data[metric] = pd.to_numeric(data[metric], errors="raise")
        if not np.isfinite(data[metric]).all():
            raise ValueError(f"{method}: nonfinite {metric} values")
    if not data["SSIM"].between(0, 1).all():
        raise ValueError(f"{method}: SSIM outside [0,1]")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for method, flag in METHODS.items():
        parser.add_argument("--" + flag, type=Path, required=True, help=f"Image-level results for {method}")
    parser.add_argument("--subject_regex", default=r"Dicom(\d+)MemEffMotion",
                        help="Regex with one capturing group for subject ID")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "MCFormer_subject_level_results.csv")
    parser.add_argument("--expected_subjects", type=int, default=12,
                        help="Expected independent subjects; use 0 to disable check")
    args = parser.parse_args()
    if re.compile(args.subject_regex).groups != 1:
        parser.error("--subject_regex must contain exactly one capturing group")

    results = {method: read_metrics(getattr(args, flag), method, args.subject_regex)
               for method, flag in METHODS.items()}
    reference = results["MC-Former"]
    names = set(reference.filename)
    merged = reference[["filename", "subject_id"]].copy()
    for method in METHODS:
        df = results[method]
        if set(df.filename) != names:
            raise ValueError(f"{method}: image filenames do not match MC-Former; check inputs")
        if not df.set_index("filename")["subject_id"].reindex(reference.filename).to_numpy().tolist() == reference.subject_id.to_list():
            raise ValueError(f"{method}: inconsistent subject assignments")
        cols = df[["filename", "SSIM", "PSNR"]].rename(columns={
            "SSIM": f"{method}_SSIM", "PSNR": f"{method}_PSNR"})
        merged = merged.merge(cols, on="filename", validate="one_to_one", how="inner")
    subject_df = merged.drop(columns="filename").groupby("subject_id", sort=True).mean()
    if args.expected_subjects and len(subject_df) != args.expected_subjects:
        raise ValueError(f"Expected {args.expected_subjects} subjects, found {len(subject_df)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subject_df.to_csv(args.output, index_label="subject_id")
    print(f"Matched {len(merged)} images across four conditions")
    print(f"Aggregated {len(subject_df)} independent subjects")
    print(f"Saved {args.output.resolve()}")

if __name__ == "__main__":
    main()
