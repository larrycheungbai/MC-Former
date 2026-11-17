# MC-Former: Motion Correction in Brain MRI Using Vision Transformers and Self-Supervised Learning

MC-Former is a research method for retrospective correction of motion artifacts in brain magnetic resonance imaging (MRI). It uses the **Restormer** image-restoration architecture with **masked image modeling (MIM)** for self-supervised pretraining, followed by supervised training on simulated motion-corrupted MRI.

This repository provides the **MC-Former inference implementation**, a trained MC-Former checkpoint, network configuration, subject-level statistical analysis scripts, and a GPU inference-time benchmark.

> **Research use only.** This method has not been validated for clinical diagnosis or clinical deployment.

## Publications

**Conference paper**

Zhang, L., Wang, X., Herskovits, E. H., Melhem, E. R., Chang, L., Wang, Z., and Ernst, T. *Reducing Motion Artifacts in Brain MRI Using Vision Transformers and Self-Supervised Learning*. IEEE International Conference on Image Processing (ICIP), 2024.

https://doi.org/10.1109/ICIP51287.2024.10647626

**Extended journal manuscript**

*MC-Former: Motion Correction in Brain MRI Using Vision Transformers and Self-Supervised Learning*. Submitted to *Journal of Imaging*; revised manuscript under review.

The journal extension includes additional evaluation of MIM pretraining, subject-level statistical analyses, exploratory testing on real motion-corrupted images, and inference-time measurements.

## Method overview

MC-Former uses two training stages:

1. **Self-supervised pretraining:** A Restormer-based network learns to reconstruct masked regions of motion-free brain MRI. The investigated MIM configuration uses 16 × 16 patches and a 50% masking ratio.
2. **Supervised motion correction:** The pretrained network is further trained using simulated motion-corrupted MRI paired with motion-free targets.

The comparison with a standard Restormer baseline uses the same backbone architecture, making it possible to study the effect of MIM pretraining without introducing a different inference network.

## Repository structure

```text
MC-Former/
├── README.md
├── test.py
├── environment_restormer.yml
├── Deblurring_V3_1200k_with_overlap_downstream_last_clean_img.yml
├── Model/
│   └── net_g_1028000.pth
├── analysis/
│   ├── README.md
│   ├── prepare_subject_level_data.py
│   ├── MCFormer_subject_level_ANOVA_Bonferroni.py
│   ├── MCFormer_subject_level_Shapiro_Wilk.py
│   ├── MCFormer_subject_level_Wilcoxon_Bonferroni.py
│   └── plot_subject_level_violin.py
├── benchmark/
│   ├── README.md
│   └── benchmark_mcformer.py
└── legacy/
    ├── Scripts/
    └── Figures/
```

The `legacy/` directory preserves earlier slice-level scripts and figures for historical reference. They are **not** the statistical analyses used in the revised journal manuscript.

## Requirements

- Linux environment with Python 3.11 and PyTorch 2.0.1 (the provided Conda environment uses the CUDA 11.8 PyTorch build)
- NVIDIA GPU with a compatible CUDA driver for the current `test.py`
- Original [Restormer source code](https://github.com/swz30/Restormer), including its `basicsr` architecture implementation and motion-deblurring image utilities
- Python packages specified in `environment_restormer.yml`

**The original Restormer checkpoint is not required.** This repository distributes only the MC-Former checkpoint.

### Installation

First clone Restormer and install its dependencies following the [official Restormer instructions](https://github.com/swz30/Restormer):

```bash
git clone https://github.com/swz30/Restormer.git
```

Then clone MC-Former:

```bash
git clone https://github.com/larrycheungbai/MC-Former.git
cd MC-Former
```

The original development environment is supplied as `environment_restormer.yml`. As an alternative to the official Restormer setup, it can be **attempted** with:

```bash
conda env create -f environment_restormer.yml
conda activate restormer
```

The environment file contains version-pinned dependencies from the development workstation. Package availability and CUDA-driver compatibility may require local adjustments. Do not install both environments on top of one another without checking their dependencies.

## Pretrained MC-Former model

The distributed MC-Former checkpoint is:

```text
Model/net_g_1028000.pth
```

The accompanying network configuration is:

```text
Deblurring_V3_1200k_with_overlap_downstream_last_clean_img.yml
```

`test.py` reads the `network_g` section of this YAML and expects a PyTorch checkpoint containing a `params` state dictionary.

The YAML contains historical training paths; **they are not used by the inference script**. Inference uses the checkpoint path supplied by `--weights`, which defaults to the file in `Model/`.

The published checkpoint has not yet been separately validated in a clean installation as part of this repository update.

## Inference

### Prepare images

Place the motion-corrupted MRI image slices in a folder containing **three-channel 8-bit PNG or JPG images**. The script processes a single image file or all supported images directly within one folder.

Example:

```text
MC-Former/
└── sample_images/
    ├── image_001.png
    └── image_002.png
```

The current script does **not** directly load DICOM or NIfTI volumes. If your source MRI data are in those formats, preprocessing and image conversion must be performed separately and consistently with the intended evaluation workflow.

### Run

From the MC-Former repository root, with Restormer cloned as a sibling folder:

```bash
python test.py \
  --restormer_root ../Restormer \
  --input_dir ./sample_images \
  --result_dir ./results
```

Change `../Restormer` if Restormer is installed at a different location.

Optional arguments:

| Argument | Purpose | Default |
|---|---|---|
| `--input_dir` | Input image file or image directory | **Required** |
| `--result_dir` | Output PNG directory | `./results` |
| `--weights` | MC-Former `.pth` checkpoint | `./Model/net_g_1028000.pth` |
| `--config` | YAML file containing `network_g` | Included root-level configuration |
| `--restormer_root` | Root of a cloned Restormer source tree | Uses importable Restormer modules if omitted |

The script loads the MC-Former checkpoint, performs CUDA inference on each image, and saves a corrected PNG with the same image dimensions and filename stem in the output directory.

For the example above:

```text
results/
├── image_001.png
└── image_002.png
```

**Verification status:** The underlying MC-Former workflow has been run on a laboratory workstation using an installed Restormer environment. However, this updated public `test.py` and the specific `net_g_1028000.pth` checkpoint still require a documented end-to-end smoke test after a fresh checkout. No independently verified inference result is claimed here.

## Subject-level statistical analysis

The [`analysis/`](analysis/) directory contains the scripts supporting the revised manuscript's statistical comparisons of **Corrupted**, **MC-Net**, **Restormer**, and **MC-Former**.

The in-house simulated-motion test set comprised 4,680 image slices from 12 subjects. For statistical inference, SSIM and PSNR values were first averaged **within each subject**. The subject, rather than an individual image slice, is the unit of analysis.

The scripts implement:

- Repeated-measures ANOVA with **Greenhouse-Geisser** adjustment and partial eta-squared
- All **six pairwise comparisons per metric**, using paired t-tests with **Bonferroni correction applied separately to SSIM and PSNR**
- Shapiro-Wilk tests of subject-level paired differences
- Wilcoxon signed-rank sensitivity analyses with Bonferroni adjustment
- Subject-level SSIM and PSNR violin plots

These scripts do **not** use Holm correction.

See the [analysis instructions](analysis/README.md) for required input formats, execution order, and generated files.

The original image-level metric files and in-house MRI volumes are not included. The statistical scripts can be run with compatible metric files or an appropriately prepared subject-level summary table, subject to data-sharing permissions.

## GPU inference-time benchmark

The [`benchmark/`](benchmark/) folder contains a **MC-Former-only** forward-pass timing script:

```bash
python benchmark/benchmark_mcformer.py \
  --restormer_root ../Restormer \
  --input_dir ./sample_images \
  --result_dir ./benchmark_results \
  --warmup 30 \
  --max_images 1000
```

The benchmark uses the public `Model/net_g_1028000.pth` checkpoint by default, a batch size of one, and CUDA synchronization before and after each timed forward pass. Image loading, preprocessing, GPU transfer, and output saving are excluded from the measured interval.

It writes:

- `MCFormer_inference_time_per_image.csv`: one latency measurement per image
- `MCFormer_inference_time_summary.txt`: checkpoint, GPU, PyTorch version, and descriptive latency statistics

For a closer methodological comparison with the manuscript's timing experiment, use 1,000 three-channel images at 256 × 256 pixels and 30 warm-up forwards.

**Important:** The manuscript's Table 4 timing experiment used a different MC-Former checkpoint (`net_g_latest.pth`) and a corresponding configuration. Therefore, this public benchmark supports new timing measurements but is **not asserted to exactly reproduce the numerical timing reported in Table 4**. The baseline checkpoints for MC-Net and Restormer are not distributed here.

See the [benchmark instructions](benchmark/README.md) for further details.

## Data availability and limitations

The original in-house MRI dataset is not publicly released due to data-sharing and privacy restrictions. The repository provides the model checkpoint and analysis code, but reproducing the original numerical experiments requires access to the corresponding evaluation inputs or precomputed metrics.

The model was developed primarily using **simulated motion artifacts**. Exploratory evaluation on real motion-corrupted MRI showed only modest improvement over the uncorrected inputs. Generalization across scanners, sequences, and patient populations remains to be established.

The software is provided for research purposes and has not been validated for clinical diagnosis, clinical decision-making, or surgical planning.

## Citation

If you use the model or code, please cite our IEEE ICIP 2024 paper:

```bibtex
@inproceedings{zhang2024reducing,
  title={Reducing Motion Artifacts in Brain MRI Using Vision Transformers and Self-Supervised Learning},
  author={Zhang, Lei and Wang, Xiaoke and Herskovits, Edward H. and Melhem, Elias R. and Chang, Linda and Wang, Ze and Ernst, Thomas},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)},
  year={2024},
  doi={10.1109/ICIP51287.2024.10647626}
}
```

MC-Former builds on the original Restormer architecture:

```bibtex
@inproceedings{zamir2022restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Zamir, Syed Waqas and Arora, Aditya and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5728--5739},
  year={2022}
}
```

Original Restormer project: https://github.com/swz30/Restormer
