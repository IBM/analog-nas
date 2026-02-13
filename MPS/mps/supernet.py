from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aihwkit_backend import AnalogConfig, make_inference_rpu_config, analog_conv2d_from_digital, analog_linear_from_digital, maybe_move_analog_to_cuda
from .ops_qat import make_qat_conv2d, make_qat_linear, SkipOp


@dataclass
class ChoiceConfig:
    enable_fp16: bool = True
    enable_int8: bool = True
    enable_analog: bool = True
    enable_skip: bool = True


class MixedPrecisionChoice(nn.Module):
    def __init__(
        self,
        name: str,
        base: nn.Module,
        is_conv: bool,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        choice_cfg: ChoiceConfig,
        analog_cfg: AnalogConfig,
        device: str,
    ):
        super().__init__()
        self.name = name
        self.is_conv = is_conv
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.choice_cfg = choice_cfg
        self.analog_cfg = analog_cfg
        self.device = device

        self.ops = nn.ModuleDict()
        # fp16 path = base module (executed under autocast by training loop)
        if choice_cfg.enable_fp16:
            self.ops["fp16"] = base

        # int8 path = QAT-prepared module (still float weights during training)
        if choice_cfg.enable_int8:
            self.ops["int8"] = make_qat_conv2d(base) if is_conv else make_qat_linear(base)

        # analog path = AIHWKIT analog module (weights copied from base)
        if choice_cfg.enable_analog:
            rpu = make_inference_rpu_config(analog_cfg)
            a = analog_conv2d_from_digital(base, rpu) if is_conv else analog_linear_from_digital(base, rpu)

            # Copy trained digital weights into analog weights container
            # AIHWKIT uses analog tiles; set_weights is the supported way.
            with torch.no_grad():
                w = base.weight.detach().cpu().numpy()
                b = None if (getattr(base, "bias", None) is None) else base.bias.detach().cpu().numpy()
                a.set_weights(w, b)

            a = maybe_move_analog_to_cuda(a, want_cuda=("cuda" in device))
            self.ops["analog"] = a

        if choice_cfg.enable_skip:
            self.ops["skip"] = SkipOp(in_shape, out_shape, is_conv=is_conv)

        self.op_names = list(self.ops.keys())
        self.alpha = nn.Parameter(torch.zeros(len(self.op_names), dtype=torch.float32))

        self.sampled_op: Optional[str] = None
        self.use_gumbel: bool = False
        self.gumbel_tau: float = 1.0

    def set_sampled_op(self, op: Optional[str]) -> None:
        self.sampled_op = op

    def set_gumbel(self, enabled: bool, tau: float) -> None:
        self.use_gumbel = enabled
        self.gumbel_tau = tau

    def softmax_probs(self) -> torch.Tensor:
        return torch.softmax(self.alpha, dim=0)

    def argmax_op(self) -> str:
        return self.op_names[int(torch.argmax(self.alpha).item())]

    def set_analog_out_noise(self, out_noise: float) -> None:
        # Progressive noise scaling: update rpu_config forward noise if possible
        if "analog" in self.ops:
            analog = self.ops["analog"]
            if hasattr(analog, "rpu_config") and hasattr(analog.rpu_config, "forward"):
                if hasattr(analog.rpu_config.forward, "out_noise"):
                    analog.rpu_config.forward.out_noise = float(out_noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sampled_op is not None:
            return self.ops[self.sampled_op](x)

        if self.use_gumbel:
            w = F.gumbel_softmax(self.alpha, tau=self.gumbel_tau, hard=False, dim=0)
            out = 0.0
            for i, k in enumerate(self.op_names):
                out = out + w[i] * self.ops[k](x)
            return out

        w = self.softmax_probs()
        out = 0.0
        for i, k in enumerate(self.op_names):
            out = out + w[i] * self.ops[k](x)
        return out


class SuperNet(nn.Module):
    def __init__(self, model: nn.Module, choices: List[MixedPrecisionChoice]):
        super().__init__()
        self.model = model
        self.choices = nn.ModuleList(choices)

    def named_choices(self):
        return [(c.name, c) for c in self.choices]

    def set_all_sampled(self, mapping: Optional[Dict[str, str]]) -> None:
        for c in self.choices:
            c.set_sampled_op(None if mapping is None else mapping.get(c.name, None))

    def set_gumbel_all(self, enabled: bool, tau: float) -> None:
        for c in self.choices:
            c.set_gumbel(enabled, tau)

    def extract_mapping(self) -> Dict[str, str]:
        return {c.name: c.argmax_op() for c in self.choices}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _infer_shapes(model: nn.Module, input_shape=(1, 3, 32, 32)):
    shapes = {}
    hooks = []

    def register(name, mod):
        def hook(m, inp, out):
            x = inp[0]
            shapes[name] = (tuple(x.shape[1:]), tuple(out.shape[1:]))
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(register(name, m)))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))

    for h in hooks:
        h.remove()
    return shapes


def convert_to_supernet(
    model: nn.Module,
    choice_cfg: ChoiceConfig,
    analog_cfg: AnalogConfig,
    device: str,
    input_shape=(1, 3, 32, 32),
) -> SuperNet:
    shapes = _infer_shapes(model, input_shape=input_shape)
    choices: List[MixedPrecisionChoice] = []

    for name, m in list(model.named_modules()):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]

            in_shape, out_shape = shapes[name]
            is_conv = isinstance(m, nn.Conv2d)
            wrapped = MixedPrecisionChoice(
                name=name,
                base=m,
                is_conv=is_conv,
                in_shape=in_shape,
                out_shape=out_shape,
                choice_cfg=choice_cfg,
                analog_cfg=analog_cfg,
                device=device,
            )
            setattr(parent, leaf, wrapped)
            choices.append(wrapped)

    return SuperNet(model=model, choices=choices)
