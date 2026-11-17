# MC-Former GPU inference-time benchmark

This folder contains a **MC-Former-only** latency benchmark. It uses the public
checkpoint `Model/net_g_1028000.pth` by default, and imports the original
Restormer architecture from a separately installed Restormer repository. No
Restormer baseline weights are needed.

## Running

From the root of the MC-Former repository:

```bash
python benchmark/benchmark_mcformer.py \
  --restormer_root ../Restormer \
  --input_dir ./sample_images \
  --result_dir ./benchmark_results \
  --warmup 30 \
  --max_images 1000
```

Provide PNG/JPG images with three channels. For comparability with the journal
benchmark, use 1,000 images of size 256 × 256. The command reports a batch size
of one and times only the CUDA model forward pass, synchronized immediately
before and after the call. Loading, padding, device transfer and image saving
are excluded from the timed region. The output directory contains a CSV of
per-image latencies and a TXT summary (GPU, software version, mean, SD, median,
P95).

## Relationship to the paper

The manuscript's Table 4 reports inference timing from a separate experiment.
That experiment used `net_g_latest.pth` and its paired configuration, whereas
this public example defaults to `net_g_1028000.pth`. The checkpoint version,
GPU hardware, software and image selection can affect results. **Do not claim
that this script reproduces the exact published latency value without running
the same configuration and checking the results.**

This script does not benchmark MC-Net or a separately trained Restormer baseline.
The paper reports those results, but their checkpoints are not part of this
MC-Former-only repository.

Note: Inference has not been tested in this environment with the actual GPU and
checkpoint. Run a smoke test on a compatible machine before marking the GitHub
workflow verified.
