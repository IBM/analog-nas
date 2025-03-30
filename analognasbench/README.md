# AnalogNAS-Bench: A NAS Benchmark for Analog In-Memory Computing

## Overview
AnalogNAS-Bench is the first Neural Architecture Search (NAS) benchmark specifically designed for Analog In-Memory Computing (AIMC). It provides a comprehensive evaluation of 15,625 architectures from NAS-Bench-201 trained on the CIFAR-10 dataset under different conditions, including AIMC-specific hardware simulations.

This benchmark enables researchers to explore how different neural architectures perform on AIMC hardware and compare them against standard digital training methods.

## Features

- **Large-scale Benchmark:** 15,625 architectures from NAS-Bench-201.

- **Diverse Training Conditions:**
    - **Baseline Evaluation**: Full training on standard hardware.
    - **Noisy Evaluation**: Direct evaluation on AIMC hardware.
    - **Hardware-Aware Training (HWT)**: Fine-tuning trained models on AIMC hardware with noise injection.
    - **Post-Training Quantization (PTQ)**: Quantizing pre-trained models on standard hardware.
    - **Quantization-Aware Training (QAT)**: Training with quantization awareness on standard hardware.

- **Temporal Drift Simulation:** Evaluates performance degradation over time on AIMC hardware, for both noisy and hardware-aware training evaluations, at intervals of 1 minute, 1 hour, 1 day, and 1 month.

- **Ongoing Expansion:** Currently supports CIFAR-10, with CIFAR-100 and ImageNet16-120 integration in progress.

## Installation
- Clone the AnalogNAS repository
```bash
git clone https://github.com/IBM/analog-nas.git
```
- Move to analognasbench folder
```bash
cd analognasbench
```
- Run the setup script
```bash
pip install .
```


## Usage
### Initialize Benchmark
```python
from analognasbench.data import AnalogNASBench
benchmark = AnalogNASBench()
```

### Query a Specific Metric for an Architecture
Available metrics:
- `baseline_accuracy` - Standard training
- `noisy_accuracy` - Noisy evaluation
- `analog_accuracy` - Hardware-aware training (HWT)
- `ptq_accuracy` - Post-training quantization (PTQ)
- `qat_accuracy` - Quantization-aware training (QAT)
- `noisy_drift_60`, `analog_drift_60` - 1 min drift
- `noisy_drift_3600`, `analog_drift_3600` - 1 hour drift
- `noisy_drift_86400`, `analog_drift_86400` - 1 day drift
- `noisy_drift_2592000`, `analog_drift_2592000` - 1 month drift

Example:
```python
architecture = (0, 0, 0, 4, 3, 2)
analog_accuracy = benchmark.query_metric(architecture, 'analog_accuracy')
print(f"Analog Accuracy: {analog_accuracy}")
```

### Retrieve Full Architecture Details
```python
arch_details = benchmark.get_architecture_details(architecture)
print(arch_details)
```
