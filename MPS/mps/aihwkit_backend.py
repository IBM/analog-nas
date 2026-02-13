from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters import IOParameters, UpdateParameters
from aihwkit.simulator.rpu_base import cuda


@dataclass
class AnalogConfig:
    # These are intentionally conservative defaults; tune to match your hardware assumptions.
    # Progressive scaling will multiply out_noise.
    out_noise: float = 0.0          # output noise std
    adc_bits: int = 0               # 0 => ideal ADC in many configs; adjust as desired
    dac_bits: int = 0
    is_cuda: bool = True


def make_inference_rpu_config(cfg: AnalogConfig) -> InferenceRPUConfig:
    rpu = InferenceRPUConfig()

    # IO noise is typically controlled via IOParameters
    rpu.forward = IOParameters()
    rpu.forward.out_noise = cfg.out_noise
    rpu.forward.out_res = 0.0

    # Optional: model ADC/DAC bit constraints if supported by your AIHWKIT version
    # (some versions expose these differently; keep as best-effort)
    if hasattr(rpu.forward, "adc_bits"):
        rpu.forward.adc_bits = cfg.adc_bits
    if hasattr(rpu.forward, "dac_bits"):
        rpu.forward.dac_bits = cfg.dac_bits

    # Update parameters not used for inference config typically, but set safe defaults
    rpu.update = UpdateParameters()
    return rpu


def analog_conv2d_from_digital(digital_conv, rpu_config: InferenceRPUConfig):
    return AnalogConv2d(
        digital_conv.in_channels,
        digital_conv.out_channels,
        kernel_size=digital_conv.kernel_size,
        stride=digital_conv.stride,
        padding=digital_conv.padding,
        dilation=digital_conv.dilation,
        groups=digital_conv.groups,
        bias=(digital_conv.bias is not None),
        rpu_config=rpu_config,
    )


def analog_linear_from_digital(digital_fc, rpu_config: InferenceRPUConfig):
    return AnalogLinear(
        digital_fc.in_features,
        digital_fc.out_features,
        bias=(digital_fc.bias is not None),
        rpu_config=rpu_config,
    )


def maybe_move_analog_to_cuda(module, want_cuda: bool):
    # Some AIHWKIT analog modules require specific cuda context handling
    if want_cuda and cuda.is_compiled():
        module.cuda()
    return module
