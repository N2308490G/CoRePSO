"""Value-only NumPy interface for MetaBox; uses the existing CEC protocol.

Coordinates are normalized to [-5, 5], as in the completed comparisons.
Returned values are objective errors, so MetaBox's optimum is zero.
This is an objective adapter, not a trained baseline or training runner.
"""
import time
import copy
import numpy as np
import torch
from cuda_objectives import NumericalObjective, PROTOCOL


class EvaluationBudgetExceeded(RuntimeError):
    pass


class MetaBBOCECProblem:
    def __init__(self, fid, dim=20, *, max_fes, device="cpu"):
        if fid not in range(1, 13) or dim not in (10, 20):
            raise ValueError("Supported scope: CEC2022 F1-F12, 10D or 20D")
        if not isinstance(max_fes, int) or isinstance(max_fes, bool) or max_fes <= 0:
            raise ValueError("max_fes must be a positive integer")
        self.dim, self.lb, self.ub, self.optimum = dim, -5., 5., 0.
        self.case = dict(group="cec2022", name=f"F{fid}", fid=fid, dim=dim, instance=0)
        self.objective_protocol = PROTOCOL
        self.max_fes = max_fes
        self._objective = NumericalObjective(self.case, device)
        self.ledger = dict(fes=0, best=float('inf'), best_z=None)
        self.reset()

    def __deepcopy__(self, memo):
        result = object.__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, value if key == 'ledger' else copy.deepcopy(value, memo))
        return result

    def reset(self):
        self.fes, self.T1 = 0, 0.
        self.best, self.best_z = float("inf"), None

    def eval(self, x):
        points = np.asarray(x, dtype=np.float64)
        if points.ndim not in (1, 2) or points.shape[-1] != self.dim:
            raise ValueError("Expected a vector (D,) or population (N,D)")
        batch = points.reshape(-1, self.dim)
        if not len(batch) or not np.isfinite(batch).all() or (np.abs(batch) > 5.).any():
            raise ValueError("Candidates must be finite and within [-5,5]")
        if self.fes + len(batch) > self.max_fes:
            raise EvaluationBudgetExceeded("Whole batch exceeds remaining FE budget")
        start = time.perf_counter()
        with torch.no_grad():
            values = self._objective.value(torch.as_tensor(batch, dtype=torch.float64,
                device=self._objective.device)).cpu().numpy()
        self.fes += len(batch)
        self.ledger['fes'] += len(batch)
        self.T1 += (time.perf_counter() - start) * 1000
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite objective values")
        i = int(values.argmin())
        if values[i] < self.best:
            self.best, self.best_z = float(values[i]), batch[i].copy()
        if values[i] < self.ledger['best']:
            self.ledger.update(best=float(values[i]), best_z=batch[i].copy())
        return float(values[0]) if points.ndim == 1 else values

    def __str__(self):
        return f"CEC2022_{self.case['name']}_{self.dim}D"
