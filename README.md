# MC-Former: Brain MRI Motion Correction Using Vision Transformers and Self-Supervised Learning

MC-Former is a deep learning approach for retrospective motion correction in brain magnetic resonance imaging (MRI). It combines a Restormer-based image restoration network with masked image modeling (MIM) as a self-supervised pretraining strategy.

The model is pretrained by reconstructing masked regions of motion-free MRI images, followed by supervised training to correct simulated motion artifacts.

## Publications

**Conference paper (ICIP 2024)**

*Reducing Motion Artifacts in Brain MRI Using Vision Transformers and Self-Supervised Learning*

[IEEE DOI](https://doi.org/10.1109/ICIP51287.2024.10647626)

**Extended journal manuscript**

*MC-Former: Motion Correction in Brain MRI Using Vision Transformers and Self-Supervised Learning*

Journal of Imaging, under revision.

The extended manuscript includes additional investigations of pretraining patch size, subject-level statistical analyses, external evaluation on real motion-corrupted MRI, and computational efficiency.

## Method Overview

MC-Former uses a two-stage training procedure:

1. **Self-supervised pretraining:** The model reconstructs randomly masked image patches from motion-free brain MRI. The primary configuration uses a masking ratio of 50% and patches of 16 × 16 pixels.
2. **Supervised motion correction:** The pretrained network is subsequently trained using paired simulated motion-corrupted and motion-free brain MRI images.

MC-Former and the Restormer baseline use the same network backbone, enabling a controlled comparison of the effects of MIM pretraining.

## Results

Performance on the in-house simulated motion test dataset:

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Corrupted input | 26.20 | 0.7840 |
| MC-Net | 29.72 | 0.9185 |
| Restormer | 31.53 | 0.9379 |
| MC-Former | **32.76** | **0.9474** |

An additional exploratory evaluation on the independent PMoC3D dataset, comprising 27 real motion-corrupted scans from eight subjects, showed modest improvements over the uncorrected input (PSNR: 21.17 vs. 21.06 dB; SSIM: 0.6747 vs. 0.6695).

## Installation

MC-Former is implemented using the Restormer architecture. The original Restormer source code is required for inference, but its pretrained model weights are not needed.

### Step 1. Install Restormer

Clone the original [Restormer repository](https://github.com/swz30/Restormer):

```bash
git clone https://github.com/swz30/Restormer.git
```

Follow the [official Restormer installation instructions](https://github.com/swz30/Restormer/blob/main/INSTALL.md) to install its dependencies.

The Conda environment used in our development is also provided as `environment_restormer.yml`.

### Step 2. Download MC-Former

```bash
git clone https://github.com/larrycheungbai/MC-Former.git
cd MC-Former
```

The repository includes the MC-Former checkpoint and its corresponding network configuration:

- **Checkpoint:** `Model/net_g_1028000.pth`
- **Configuration:** `Deblurring_V3_1200k_with_overlap_downstream_last_clean_img.yml`
- **Inference script:** `test.py`

## Inference

### Input Preparation

Prepare a directory containing motion-corrupted brain MRI images in PNG or JPG format.

The current implementation expects three-channel images with 8-bit intensity values. It does not directly read DICOM or NIfTI files.

For example:

```text
MC-Former/
├── test.py
├── Model/
│   └── net_g_1028000.pth
└── sample_images/
    ├── image_001.png
    └── image_002.png
```

### Run MC-Former

From the MC-Former repository root, execute:

```bash
python test.py \
    --restormer_root ../Restormer \
    --input_dir ./sample_images \
    --result_dir ./results
```

Replace `../Restormer` with the path to your local Restormer installation if the directory structure is different.

The script uses the included MC-Former checkpoint and YAML configuration by default.

### Output

The corrected images are saved as PNG files in the specified output directory:

```text
results/
├── image_001.png
└── image_002.png
```

Output image dimensions are preserved.

### Optional Arguments

The script supports the following command-line arguments:

| Argument | Description |
|---|---|
| `--input_dir` | Input image file or folder containing PNG/JPG images |
| `--result_dir` | Directory for corrected output images |
| `--weights` | Path to the MC-Former checkpoint |
| `--config` | Path to the network configuration YAML |
| `--restormer_root` | Path to the original Restormer repository |

The current implementation requires an NVIDIA GPU with CUDA support.

## Pretrained Model

The pretrained MC-Former checkpoint is available in this repository:

`Model/net_g_1028000.pth`

The model uses the Restormer architecture and was developed through masked image modeling (MIM) pretraining followed by supervised training for MRI motion correction.

The original Restormer baseline checkpoint is not required and is not distributed with this repository.

## Limitations

MC-Former was primarily trained on simulated motion artifacts. Its improvements on real motion-corrupted MRI were modest, and broader validation across scanners, acquisition protocols, and patient populations is needed.

This software is intended for research use and has not been validated for clinical diagnosis.

## Citation

If you use this repository, please cite our ICIP 2024 paper and the original Restormer work. The extended journal manuscript can be cited once formally published.

**MC-Former, ICIP 2024**

Zhang, L.; Wang, X.; Herskovits, E.H.; Melhem, E.R.; Chang, L.; Wang, Z.; Ernst, T. Reducing Motion Artifacts in Brain MRI Using Vision Transformers and Self-Supervised Learning. IEEE ICIP, 2024. https://doi.org/10.1109/ICIP51287.2024.10647626.

**Restormer, CVPR 2022**

Zamir, S.W.; Arora, A.; Khan, S.; Hayat, M.; Khan, F.S.; Yang, M.H. Restormer: Efficient Transformer for High-Resolution Image Restoration. IEEE/CVF CVPR, 2022, pp. 5728–5739.
