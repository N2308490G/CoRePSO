"""Swarm-state operations and the pinned AutoPSO update implementation.

CoRePSO delegates particle update equations to the verified upstream module.
"""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import hashlib
import importlib.util
import os
import sys
import time
import numpy as np
import torch

PINNED_SHA = '0e0158e6c58f335f363af92a4f1fc5f4b4aa697f'

def load_upstream():
    path = Path(os.environ.get('AUTOPSO_CORE', Path(__file__).parent / 'vendor' / 'autopso.py'))
    raw = path.read_bytes()
    digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if digest != PINNED_SHA:
        raise RuntimeError(f'Original implementation checksum mismatch: {digest}')
    spec = importlib.util.spec_from_file_location('pinned_autopso', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

upstream = load_upstream()
State = upstream.InnerState
STATE_KEYS = tuple(f.name for f in fields(State))

class Objective:
    """Independent, shifted/rotated diagnostic suite, NOT CEC/BBOB."""
    def __init__(self, name: str, dim: int, instance: int, budget: int):
        self.name, self.dim, self.instance, self.budget = name, dim, instance, budget
        self.evaluations = 0
        self.best = float('inf')
        self.best_x = None
        self.trace = []
        rng = np.random.default_rng(76543 + dim * 1000 + instance)
        self.shift = torch.tensor(rng.uniform(-2, 2, dim), dtype=torch.float64)
        q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
        q = q * np.sign(np.diag(r))
        self.rotation = torch.tensor(q, dtype=torch.float64)
        self.weights = torch.logspace(0, 6, dim, dtype=torch.float64)
        self.indices = torch.sqrt(torch.arange(1, dim + 1, dtype=torch.float64))
        self.powers = torch.linspace(2, 6, dim, dtype=torch.float64)

    def value(self, x):
        z = (x - self.shift) @ self.rotation
        if self.name == 'sphere':
            return z.square().sum(-1)
        if self.name == 'rastrigin':
            return (z.square() + 10 * (1-torch.cos(2*torch.pi*z))).sum(-1)
        if self.name == 'ackley':
            return (-20*torch.exp(-.2*z.square().mean(-1).sqrt())
                    -torch.exp(torch.cos(2*torch.pi*z).mean(-1)) + 20 + np.e)
        if self.name == 'rosenbrock':
            y = z + 1
            return (100*(y[...,1:]-y[...,:-1].square()).square() + (y[...,:-1]-1).square()).sum(-1)
        if self.name == 'ellipsoid':
            return (z.square()*self.weights).sum(-1)
        if self.name == 'discus':
            return 1e6*z[...,0].square() + z[...,1:].square().sum(-1)
        if self.name == 'bent_cigar':
            return z[...,0].square() + 1e6*z[...,1:].square().sum(-1)
        if self.name == 'griewank':
            y = z*20
            return y.square().sum(-1)/4000 - torch.cos(y/self.indices).prod(-1) + 1
        if self.name == 'schwefel12':
            return z.cumsum(-1).square().sum(-1)
        if self.name == 'different_powers':
            return (z.abs()**self.powers).sum(-1).sqrt()
        raise ValueError(f'Unknown diagnostic function {self.name!r}')

    def __call__(self, x):
        n = int(x.shape[0])
        if self.evaluations+n > self.budget:
            raise RuntimeError('Function evaluation budget exceeded')
        values = self.value(x)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Invalid objective values')
        if not bool((x >= -5).all() and (x <= 5).all()):
            raise AssertionError('Bounds violated')
        val, idx = values.min(0)
        if float(val) < self.best:
            self.best, self.best_x = float(val), x[int(idx)].detach().clone()
        self.evaluations += n
        self.trace.append((self.evaluations, self.best))
        return values


def replicate_state(state, count):
    return State(*(getattr(state,k).expand((count,)+getattr(state,k).shape[1:]).clone() for k in STATE_KEYS))


def choose_state(state, index):
    return State(*(getattr(state,k)[index:index+1].clone() for k in STATE_KEYS))


def make_same_noise_step(inner):
    def one_step(tensors, fitness, params):
        local = State(*(x.unsqueeze(0) for x in tensors))
        p = {k:v.unsqueeze(0) for k,v in params.items()}
        nxt = inner.step(local, fitness.unsqueeze(0), p)
        return tuple(getattr(nxt,k).squeeze(0) for k in STATE_KEYS)
    vmapped = torch.vmap(one_step, randomness='same')
    def apply(state, fitness, params):
        tensors = tuple(getattr(state,k) for k in STATE_KEYS)
        return State(*vmapped(tensors,fitness,params))
    return apply


@torch.no_grad()
def run(name: str, dim: int, instance: int, seed: int, method: str,
        outer_n=16, inner_n=40, horizon=100, rounds=6):
    if method not in ('original', 'smc', 'smc_independent_noise', 'smc_stale_memory', 'matched_restart'):
        raise ValueError(method)
    budget = outer_n*inner_n*horizon*rounds
    objective = Objective(name,dim,instance,budget)
    torch.manual_seed(seed)
    lo = torch.zeros(10,dtype=torch.float64)
    hi = torch.tensor([1,1,1,1,1,8.99999,8.99999,8.99999,8.99999,1],dtype=torch.float64)
    outer = upstream.TorchOuterPSO(lo,hi,outer_n)
    inner = upstream.TorchAutoPSO(torch.full((dim,),-5.,dtype=torch.float64),
                                 torch.full((dim,),5.,dtype=torch.float64),inner_n)
    common_step = make_same_noise_step(inner)
    parent = None
    history = []
    started = time.perf_counter()
    for epoch in range(rounds):
        params = upstream.decode_autopso_parameters_torch(outer.population)
        if method == 'original':
            fitness = inner.evaluate_params(params,objective,horizon)
            outer.step(fitness)
        else:
            if parent is None or method == 'matched_restart':
                parent = inner.setup(1)
            state = replicate_state(parent,outer_n)
            parent_fitness = float(parent.gbest_fitness.min())
            for _ in range(horizon):
                fit = objective(state.population.reshape(-1,dim)).reshape(outer_n,1,inner_n)
                if method == 'smc_independent_noise':
                    state = inner.step(state,fit,params)
                else:
                    state = common_step(state,fit,params)
            fitness = state.gbest_fitness.amin(dim=1)
            winner = int(fitness.argmin())
            parent = choose_state(state,winner)
            winner_theta = outer.population[winner].clone()
            if method != 'smc_stale_memory':
                # Old scores came from a different starting state. Never rank
                # a new action against the historical absolute loss of another.
                outer.pbest_fitness.fill_(torch.inf)
                outer.gbest_fitness.fill_(torch.inf)
            outer.step(fitness)
            # Explicit one-candidate elitism, not a manually inserted PSO rule.
            outer.population[0] = winner_theta
            outer.velocity[0].zero_()
            history.append({'epoch':epoch,'parent_before':parent_fitness,
                            'parent_after':float(fitness[winner]),'winner':winner})
    elapsed = time.perf_counter()-started
    assert objective.evaluations == budget
    assert objective.best_x is not None
    verified = float(objective.value(objective.best_x.unsqueeze(0))[0])
    assert np.isclose(objective.best,verified,rtol=1e-12,atol=1e-12)
    return {'method':method,'function':name,'dim':dim,'instance':instance,'seed':seed,
            'error':objective.best,'fes':objective.evaluations,'seconds':elapsed,
            'best_x':objective.best_x.tolist(),'trace':objective.trace,'epochs':history}
