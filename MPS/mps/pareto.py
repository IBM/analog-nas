from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Obj:
    acc: float          # maximize
    analog_macs: float  # maximize
    dig_w_mb: float     # minimize

def dominates(a: Obj, b: Obj) -> bool:
    return (
        a.acc >= b.acc and a.analog_macs >= b.analog_macs and a.dig_w_mb <= b.dig_w_mb
        and (a.acc > b.acc or a.analog_macs > b.analog_macs or a.dig_w_mb < b.dig_w_mb)
    )

def pareto_ranks(objs: List[Obj]) -> List[int]:
    n = len(objs)
    ranks = [-1] * n
    remaining = set(range(n))
    r = 0
    while remaining:
        front = []
        for i in list(remaining):
            if not any(dominates(objs[j], objs[i]) for j in remaining if j != i):
                front.append(i)
        for i in front:
            ranks[i] = r
            remaining.remove(i)
        r += 1
    return ranks

def normalize_ranks(ranks: List[int]) -> np.ndarray:
    r = np.array(ranks, dtype=np.float32)
    return r * 0.0 if r.max() == 0 else r / r.max()
