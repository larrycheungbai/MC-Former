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

The repository includes an environment configuration file:

`environment_restormer.yml`

Detailed installation instructions will be provided after verifying the environment configuration and dependencies.

## Inference

Inference instructions and pretrained model availability will be documented after verification of the existing testing scripts and model files.

## Limitations

MC-Former was primarily trained on simulated motion artifacts. Its improvements on real motion-corrupted MRI were modest, and broader validation across scanners, acquisition protocols, and patient populations is needed.

This software is intended for research use and has not been validated for clinical diagnosis.

## Citation

If you use this repository, please cite our ICIP 2024 paper and the original Restormer work. The extended journal manuscript can be cited once formally published.

**MC-Former, ICIP 2024**

Zhang, L.; Wang, X.; Herskovits, E.H.; Melhem, E.R.; Chang, L.; Wang, Z.; Ernst, T. Reducing Motion Artifacts in Brain MRI Using Vision Transformers and Self-Supervised Learning. IEEE ICIP, 2024. https://doi.org/10.1109/ICIP51287.2024.10647626.

**Restormer, CVPR 2022**

Zamir, S.W.; Arora, A.; Khan, S.; Hayat, M.; Khan, F.S.; Yang, M.H. Restormer: Efficient Transformer for High-Resolution Image Restoration. IEEE/CVF CVPR, 2022, pp. 5728–5739.
