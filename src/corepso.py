"""Convenience interface to the frozen CoRePSO controller.

The objective receives a batch of physical coordinates as a float64 torch
tensor and returns one value per row. Use the benchmark runners for the
paper's CUDA-graph execution and timing protocols.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class OptimizeResult:
    x: np.ndarray
    fun: float
    nfev: int
    history: np.ndarray


@torch.no_grad()
def minimize(
    objective: Callable[[torch.Tensor], torch.Tensor],
    lower,
    upper,
    *,
    max_fes: int,
    seed: int = 0,
    lineages: int = 16,
    particles: int = 40,
    device: str = "cpu",
    method: str = "corepso",
) -> OptimizeResult:
    """Minimize a deterministic, finite-valued box-constrained objective.

    Bounds are one-dimensional vectors. The budget counts candidates, including
    initialization, local probes and rejected proposals. This convenience path
    evaluates eagerly on the requested device; it does not capture CUDA graphs.
    Calls must be serial because the underlying implementation uses PyTorch's
    default dtype and global random generator.
    """
    if not isinstance(max_fes, int) or isinstance(max_fes, bool) or max_fes < 1:
        raise ValueError("max_fes must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(lineages, int) or lineages < 4:
        raise ValueError("lineages must be an integer >= 4")
    if not isinstance(particles, int) or particles < 2:
        raise ValueError("particles must be an integer >= 2")
    methods = {
        "corepso": "cvbr_hybrid",
        "autopso": "original",
        "no_split": "cvbr_hybrid_no_split",
    }
    if method not in methods:
        raise ValueError(f"method must be one of {tuple(methods)}")
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if (
        lo.ndim != 1 or len(lo) == 0 or hi.shape != lo.shape
        or not np.isfinite(lo).all() or not np.isfinite(hi).all()
        or not np.all(hi > lo)
    ):
        raise ValueError("Bounds must be finite, equally sized vectors with upper > lower")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    from cuda_search import CudaOracle, optimize

    class BoxObjective:
        dim = len(lo)
        device = target

        def __init__(self):
            self.lower = torch.as_tensor(lo, device=target)
            self.width = torch.as_tensor(hi - lo, device=target)

        def physical(self, z):
            return self.lower + (z + 5.0) * self.width / 10.0

        def value(self, z):
            values = objective(self.physical(z))
            if not isinstance(values, torch.Tensor) or values.shape != (len(z),):
                raise ValueError("objective must return a torch tensor of shape (batch,)")
            return values.to(device=target, dtype=torch.float64)

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        adapted = BoxObjective()
        oracle = CudaOracle(adapted, max_fes)
        optimize(oracle, seed, methods[method], B=lineages, N=particles)
        x = lo + (oracle.best_x.numpy() + 5.0) * (hi - lo) / 10.0
        return OptimizeResult(x=x, fun=float(oracle.best), nfev=oracle.used,
                              history=np.asarray(oracle.trace, dtype=np.float64))
    finally:
        torch.set_default_dtype(previous_dtype)
