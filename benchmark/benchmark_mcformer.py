#!/usr/bin/env python3
"""CUDA forward-pass latency benchmark for the publicly released MC-Former checkpoint.

Uses model loading and image preprocessing consistent with the repository's test.py.
This is a reproducible *new* measurement, not a guarantee of reproducing the
latency reported in the paper (which may use a different checkpoint/hardware).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test import (DEFAULT_CONFIG, DEFAULT_WEIGHTS, find_images,
                  import_restormer_dependencies)  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir', type=Path, required=True,
                   help='Folder containing 3-channel PNG/JPG MRI images (or one image).')
    p.add_argument('--result_dir', type=Path, default=ROOT / 'benchmark_results')
    p.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--restormer_root', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--max_images', type=int, default=1000,
                   help='Maximum measured images; 0 means all images.')
    args = p.parse_args()
    if args.warmup < 0 or args.max_images < 0:
        p.error('--warmup and --max_images must be nonnegative')
    if not torch.cuda.is_available():
        p.error('A CUDA-compatible NVIDIA GPU is required')

    files = find_images(args.input_dir)
    if args.max_images:
        files = files[:args.max_images]
    if not files:
        p.error('No PNG/JPG images found')
    for path in (args.weights, args.config):
        if not path.expanduser().is_file():
            p.error(f'File not found: {path}')

    utils, Restormer = import_restormer_dependencies(args.restormer_root)
    with args.config.expanduser().open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or not isinstance(cfg.get('network_g'), dict):
        p.error('Config must contain network_g mapping')
    params = dict(cfg['network_g'])
    if params.pop('type', None) != 'Restormer':
        p.error('Expected network_g.type: Restormer')

    model = Restormer(**params)
    state = torch.load(str(args.weights.expanduser()), map_location='cpu')
    if not isinstance(state, dict) or 'params' not in state:
        p.error('Checkpoint missing params state dictionary')
    model.load_state_dict(state['params'], strict=True)
    model = model.cuda().eval()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, file in enumerate(tqdm(files, desc='MC-Former benchmark')):
            image = np.float32(utils.load_img(str(file))) / 255.0
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'Expected 3-channel image: {file}')
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
            height, width = tensor.shape[-2:]
            pad_h = (8 - height % 8) % 8
            pad_w = (8 - width % 8) % 8
            if (pad_h and pad_h >= height) or (pad_w and pad_w >= width):
                raise ValueError(f'Input too small for reflect padding: {file}')
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

            if index == 0:
                for _ in range(args.warmup):
                    model(tensor)
                torch.cuda.synchronize()

            # Time only a batch-1 model forward pass, with CUDA synchronization.
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(tensor)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            records.append({'image': file.name, 'height': height,
                            'width': width, 'latency_ms': elapsed_ms})

    csv_file = args.result_dir / 'MCFormer_inference_time_per_image.csv'
    with csv_file.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    elapsed = np.asarray([r['latency_ms'] for r in records], dtype=float)
    mean = float(elapsed.mean())
    summary = (
        'Model: MC-Former\n'
        f'Checkpoint: {args.weights.resolve()}\n'
        f'Config: {args.config.resolve()}\n'
        f'GPU: {torch.cuda.get_device_name(0)}\n'
        f'PyTorch: {torch.__version__}\n'
        'Batch size: 1\n'
        f'Warmup forwards: {args.warmup}\n'
        f'Measured images: {len(elapsed)}\n'
        'Timing: synchronized GPU model forward pass only\n'
        'Excluded: image loading, padding, H2D transfer, output saving\n'
        f'Mean (ms/image): {mean:.4f}\n'
    )
    if len(elapsed) > 1:
        summary += f'SD (ms/image): {np.std(elapsed, ddof=1):.4f}\n'
    else:
        summary += 'SD (ms/image): N/A (one image)\n'
    summary += f'Median (ms/image): {np.median(elapsed):.4f}\n'
    summary += f'P95 (ms/image): {np.percentile(elapsed, 95):.4f}\n'
    summary_file = args.result_dir / 'MCFormer_inference_time_summary.txt'
    summary_file.write_text(summary, encoding='utf-8')
    print(summary)
    print(f'Per-image results: {csv_file}')
    print(f'Summary: {summary_file}')


if __name__ == '__main__':
    main()
