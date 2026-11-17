# MC-Former: Subject-Level Statistical Analysis

This folder contains the analysis scripts supporting the subject-level SSIM and PSNR comparisons in the revised MC-Former manuscript. **Subjects, not individual MRI slices, are the independent statistical units.**

The simulated-motion test set comprises 4,680 slices from 12 subjects. The four evaluated conditions are **Corrupted**, **MC-Net**, **Restormer**, and **MC-Former**.

## Files and execution order

1. `prepare_subject_level_data.py`: matches image-level metric results from the four conditions by filename and averages SSIM and PSNR within each subject. Creates `MCFormer_subject_level_results.csv`.
2. `MCFormer_subject_level_ANOVA_Bonferroni.py`: calculates one-way repeated-measures ANOVA (four conditions) for each metric, reports Greenhouse-Geisser-adjusted p-values and partial eta-squared, and conducts all **six** paired two-sided t-tests per metric with **Bonferroni** adjustment separately for SSIM and PSNR.
3. `MCFormer_subject_level_Shapiro_Wilk.py`: tests normality of subject-level paired differences for all six condition pairs within each metric.
4. `MCFormer_subject_level_Wilcoxon_Bonferroni.py`: performs the six paired Wilcoxon signed-rank comparisons per metric as a nonparametric sensitivity analysis, with Bonferroni adjustment separately for each metric.
5. `plot_subject_level_violin.py`: plots the distributions of subject-level means and annotates MC-Former versus the three comparator conditions using the **Bonferroni-adjusted** paired t-test p-values. The family for adjustment still includes all six condition pairs per metric.

Dependencies: Python, NumPy, pandas, SciPy, and Matplotlib. These analysis scripts do not require a GPU, Restormer checkpoints, or MRI volumes.

## Input format

For aggregation, supply four **whitespace-separated, headerless** metric text files, one for each condition. Each line contains:

```text
image_filename.png  0.9123  30.54
```

The three columns are `filename`, `SSIM`, and `PSNR`. Each method must use the **same set of image filenames**, with no duplicates. The default subject identifier pattern is `Dicom(\d+)MemEffMotion`, with one capturing group. Supply `--subject_regex` to use another naming convention. Input files are **not included** with this repository.

Run the aggregation from the MC-Former repository root:

```bash
python analysis/prepare_subject_level_data.py \
  --corrupted /path/to/corrupted_metrics.txt \
  --mc_net /path/to/mc_net_metrics.txt \
  --restormer /path/to/restormer_metrics.txt \
  --mc_former /path/to/mc_former_metrics.txt
```

The script saves `analysis/MCFormer_subject_level_results.csv`, with one row for each subject and columns such as `Corrupted_SSIM`, `MC-Net_PSNR`, and `MC-Former_SSIM`.

Alternatively, if you already have an appropriately generated `MCFormer_subject_level_results.csv`, place it in this `analysis/` directory and skip aggregation.

## Run the statistics and figures

From the repository root:

```bash
python analysis/MCFormer_subject_level_ANOVA_Bonferroni.py
python analysis/MCFormer_subject_level_Shapiro_Wilk.py
python analysis/MCFormer_subject_level_Wilcoxon_Bonferroni.py
python analysis/plot_subject_level_violin.py
```

The scripts create these files next to the input CSV:

- `MCFormer_subject_level_ANOVA.csv`
- `MCFormer_subject_level_Bonferroni.csv`
- `MCFormer_subject_level_Shapiro_Wilk.csv` and notes
- `MCFormer_subject_level_Wilcoxon_Bonferroni.csv` and notes
- `figures/MCFormer_subject_level_violin_SSIM_Bonferroni.png`
- `figures/MCFormer_subject_level_violin_PSNR_Bonferroni.png`

## Multiplicity and interpretation

For **each metric separately**, four conditions give six unique pairwise comparisons. Bonferroni-adjusted p-values are calculated as `min(6 × p_raw, 1)`. This is **not** Holm correction. The Greenhouse-Geisser correction is used for the repeated-measures ANOVA, rather than for post-hoc paired tests. Wilcoxon tests serve as sensitivity analyses, not as replacements for the ANOVA.

The plots and analysis are based on **12 subject-level observations per condition**, not 4,680 statistically independent observations.

## Data availability and verification

Individual MRI scans and the original image-level metric text files are **not distributed here**. Obtain appropriate permission before publishing underlying subject-level or image-level data, and ensure file names do not disclose sensitive information.

The scripts have been checked using the study's previously computed 12-subject metric table in a local environment: the ANOVA and Bonferroni outputs matched the previously saved numeric results. This does **not** establish that the entire inference pipeline has been independently reproduced from a clean installation.
