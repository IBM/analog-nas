# Mixed-Precision Supernetwork (MPS) – Reference Implementation

This repository provides a runnable reference implementation of the training methodology from:
"Supernetwork-based efficient mapping of deep learning applications to mixed-precision hardware using model adaptation"
(Algorithm 1 + Methods section).

## What is implemented

A mixed-precision supernetwork where each Conv2d/Linear can choose among candidate operators:
- FP16 (simulated via autocast during training; weights remain fp32 params)
- INT8 (fake-quant QAT style, lightweight)
- ANALOG (adds Gaussian noise to the MAC output; progressive noise scaling)
- SKIP (identity when shape matches; otherwise a learned 1x1/linear projection)

Training has 3 phases:

1) Fairness Training:
   - strict fairness sampling (uniform coverage of operators)
   - warm-starts weights under stable training (paper warms at FP16)

2) Progressive Noise Scaling + QAT:
   - evaluates per-layer analog noise sensitivity
   - progressively increases analog noise for resilient layers
   - enables INT8 fake-quant path(s) and trains them (QAT)

3) Rank-Preserving Fine-Tuning:
   - samples N subnetworks
   - computes multi-objective Pareto ranks over:
       (maximize accuracy proxy, maximize analog MAC ratio, minimize digital weight size)
   - updates ONLY selection parameters (alphas) with a rank-preserving pairwise loss

Finally:
- extracts a single mapping via argmax over per-layer softmax(alphas)
- saves both supernet checkpoint and extracted mapping json

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Installation notes (AIHWKIT)
AIHWKIT installation can be platform/CUDA specific.
Try:
  pip install aihwkit

If it fails, consult the AIHWKIT docs for the correct wheel/build for your CUDA/PyTorch.
Keep torch/torchvision versions compatible with your AIHWKIT build.

## Train ResNet8 MPS on CIFAR-10

```bash
python train_mps.py --model resnet8 --data-dir ./data --out-dir ./runs/resnet8_mps \
  --epochs-phase1 20 --epochs-phase2 20 --epochs-phase3 10 \
  --batch-size 128 --lr 0.1
```


## Test extracted mapping (ResNet8)

```bash
python tests/test_resnet8.py --data-dir ./data --ckpt ./runs/resnet8_mps/supernet.pt
```