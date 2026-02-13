from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.ao.quantization as tq


def make_qat_conv2d(base: nn.Conv2d) -> nn.Module:
    # QAT uses fake-quant modules attached via qconfig
    m = copy.deepcopy(base)
    m.qconfig = tq.get_default_qat_qconfig("fbgemm")
    return tq.prepare_qat(m, inplace=False)

def make_qat_linear(base: nn.Linear) -> nn.Module:
    m = copy.deepcopy(base)
    m.qconfig = tq.get_default_qat_qconfig("fbgemm")
    return tq.prepare_qat(m, inplace=False)

def convert_qat_to_int8(module: nn.Module) -> nn.Module:
    module = copy.deepcopy(module)
    module.eval()
    return tq.convert(module, inplace=False)


class SkipOp(nn.Module):
    def __init__(self, in_shape, out_shape, is_conv: bool):
        super().__init__()
        self.proj = None
        if in_shape != out_shape:
            if is_conv:
                cin, _, _ = in_shape
                cout, _, _ = out_shape
                stride = 1
                if in_shape[1] != out_shape[1] or in_shape[2] != out_shape[2]:
                    stride = 2
                self.proj = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
            else:
                self.proj = nn.Linear(in_shape[0], out_shape[0], bias=False)

    def forward(self, x):
        return x if self.proj is None else self.proj(x)
