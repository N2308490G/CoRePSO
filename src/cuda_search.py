"""CoRePSO controller with CPU and CUDA swarm execution paths.

Local proposals use NumPy/SciPy; the common float64 oracle accounts for
every objective query. CUDA and CPU random streams differ.
"""
import time
from collections import OrderedDict
import numpy as np
import torch
import experiment_v3 as e
from cvbr import block_recombine
from alternatives import ray_search

ABLATIONS = ('no_inheritance', 'no_cvbr', 'no_curvature', 'no_polish', 'fitness_only')


class ValueGraphs:
    """Cache fixed-shape local evaluations; never cache objective values."""
    def __init__(self, objective, max_entries=16, max_bytes=1200000000):
        self.objective = objective
        self.entries = OrderedDict()
        self.max_entries, self.max_bytes = max_entries, max_bytes

    def __call__(self, cpu):
        n = len(cpu)
        if n not in self.entries:
            while self.entries and (len(self.entries) >= self.max_entries or
                                    sum(v[3] for v in self.entries.values()) >= self.max_bytes):
                self.entries.popitem(last=False)
            before = torch.cuda.memory_allocated()
            x = torch.zeros((n, self.objective.dim), dtype=torch.float64, device=self.objective.device)
            stream = torch.cuda.Stream(device=self.objective.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.objective.value(x)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                y = self.objective.value(x)
            cost = max(torch.cuda.memory_allocated()-before, 0)
            self.entries[n] = (x, y, graph, cost)
        self.entries.move_to_end(n)
        x, y, graph, _ = self.entries[n]
        x.copy_(cpu)
        graph.replay()
        return y.cpu()


class SwarmGraph:
    def __init__(self, objective, B, N, use_graph=True, policy=None):
        self.objective, self.B, self.N = objective, B, N
        self.device, d = objective.device, objective.dim
        inner_class = e.upstream.TorchAutoPSO if policy is None else policy.inner_class
        self.inner = inner_class(torch.full((d,), -5., device=self.device),
                                            torch.full((d,), 5., device=self.device), N)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        self.state = self.inner.setup(B)
        high = torch.tensor([1.]*5+[8.99999]*4+[1.], device=self.device)
        if policy is None:
            self.params = e.upstream.decode_autopso_parameters_torch(torch.rand(B, 10, device=self.device)*high)
        else:
            self.params = policy.decode(torch.rand(B, policy.dimension, device=self.device)*policy.high.to(self.device))
        self.best = torch.full((), torch.inf, device=self.device)
        self.best_x = torch.zeros(d, device=self.device)
        self.history = torch.empty(800, device=self.device)
        self.counter = torch.zeros(1, dtype=torch.long, device=self.device)
        self.valid = torch.ones((), dtype=torch.bool, device=self.device)
        self.graph = None
        if use_graph:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.step()
                self.counter.zero_()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.step()
        torch.cuda.synchronize(self.device)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, self.device)

    def step(self):
        s, d = self.state, self.objective.dim
        x = s.population.reshape(-1, d)
        f = self.objective.value(x)
        self.valid.logical_and_(torch.isfinite(x).all() & (x >= -5).all() & (x <= 5).all() & torch.isfinite(f).all())
        val, k = f.min(0)
        self.best_x.copy_(torch.where(val < self.best, x.index_select(0, k.reshape(1)).squeeze(0), self.best_x))
        self.best.copy_(torch.minimum(self.best, val))
        nxt = self.inner.step(s, f.reshape(self.B, 1, self.N), self.params)
        for field in e.STATE_KEYS:
            getattr(s, field).copy_(getattr(nxt, field))
        self.history.index_copy_(0, self.counter, self.best.reshape(1))
        self.counter.add_(1)

    def load(self, state, params, best=np.inf, best_x=None):
        for field in e.STATE_KEYS:
            getattr(self.state, field).copy_(getattr(state, field))
        for k in self.params:
            self.params[k].copy_(params[k])
        self.best.fill_(best)
        self.best_x.zero_() if best_x is None else self.best_x.copy_(best_x)
        self.counter.zero_()
        self.valid.fill_(True)

    def run(self, oracle, state, params, steps):
        if not 1 <= steps <= 800 or steps*self.B*self.N > oracle.remaining:
            raise ValueError('Invalid swarm-block budget.')
        self.load(state, params, oracle.best, oracle.best_x)
        for _ in range(steps):
            self.graph.replay() if self.graph is not None else self.step()
        history = self.history[:steps].cpu().tolist()
        if not bool(self.valid):
            raise FloatingPointError('Invalid GPU swarm candidate or objective.')
        start = oracle.used
        oracle.trace.extend((start+(i+1)*self.B*self.N, v) for i, v in enumerate(history))
        oracle.used += steps*self.B*self.N
        if history[-1] < oracle.best:
            oracle.best, oracle.best_x = history[-1], self.best_x.cpu().clone()
        return e.State(*(getattr(self.state, k).cpu().clone() for k in e.STATE_KEYS))


class CudaOracle:
    def __init__(self, objective, budget, value_cache=None):
        self.objective, self.budget, self.used = objective, int(budget), 0
        self.lb, self.ub = np.full(objective.dim, -5.), np.full(objective.dim, 5.)
        self.best, self.best_x, self.trace = np.inf, None, []
        self.value_cache = value_cache
    @property
    def remaining(self):
        return self.budget-self.used
    def __call__(self, x):
        is_torch = isinstance(x, torch.Tensor)
        cpu = x if is_torch else torch.from_numpy(np.asarray(x, dtype=np.float64))
        if cpu.ndim != 2 or cpu.shape[1] != len(self.lb) or not len(cpu):
            raise ValueError('Invalid candidate shape.')
        if len(cpu) > self.remaining:
            raise e.BudgetStop()
        if not bool(torch.isfinite(cpu).all() & (cpu >= -5.-1e-12).all() & (cpu <= 5.+1e-12).all()):
            raise ValueError('Invalid candidate or bounds.')
        values = (self.value_cache(cpu) if self.value_cache is not None else
                  self.objective.value(cpu.to(self.objective.device, dtype=torch.float64)).cpu())
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite objective.')
        val, k = values.min(0)
        if float(val) < self.best:
            self.best, self.best_x = float(val), cpu[int(k)].cpu().clone()
        self.used += len(cpu)
        self.trace.append((self.used, self.best))
        return values if is_torch else values.numpy()


@torch.no_grad()
def optimize(oracle, seed, method, B=100, N=100, graph=None, ablation=None, max_rounds=None,
             policy=None, fixed_policy=None, policy_observer=None, outer_factory=None):
    if method not in ('original', 'cc_four', 'cvbr_hybrid', 'cvbr_hybrid_no_split'):
        raise ValueError(method)
    if ablation is not None and (ablation not in ABLATIONS or method != 'cvbr_hybrid'):
        raise ValueError('Ablations apply only to the complete CoRePSO controller.')
    if max_rounds is not None and (not isinstance(max_rounds,int) or max_rounds<1):
        raise ValueError('The explicit cost-profile round limit must be a positive integer.')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+991)
    d = len(oracle.lb)
    high = torch.tensor([1.]*5+[8.99999]*4+[1.])
    decode = e.upstream.decode_autopso_parameters_torch if policy is None else policy.decode
    inner_class = e.upstream.TorchAutoPSO if policy is None else policy.inner_class
    if policy is not None:
        high = policy.high
    factory = e.upstream.TorchOuterPSO if outer_factory is None else outer_factory
    outer = factory(torch.zeros(len(high)), high, B)
    if fixed_policy is not None:
        fixed_policy = torch.as_tensor(fixed_policy, dtype=high.dtype)
        if fixed_policy.shape != high.shape or not bool(torch.isfinite(fixed_policy).all() & (fixed_policy>=0).all() & (fixed_policy<=high).all()):
            raise ValueError('Invalid frozen source policy')
        outer.population[:] = fixed_policy
        outer.velocity.zero_()
    inner = inner_class(torch.from_numpy(oracle.lb), torch.from_numpy(oracle.ub), N)
    s, block, epochs, logs = None, 0, [], []
    survivors = min(B, 4)
    def select(state, count):
        if ablation == 'fitness_only':
            return torch.argsort(state.gbest_fitness[:, 0], stable=True)[:count].tolist()
        return e.select_diverse(state, count)
    while oracle.remaining >= B*N and (max_rounds is None or block < max_rounds):
        if s is None or method == 'original':
            s = inner.setup(B)
        params = decode(outer.population)
        steps = min(800 if method == 'original' else 100, oracle.remaining//(B*N))
        start = oracle.used
        if graph is None:
            for _ in range(steps):
                fit = oracle(s.population.reshape(-1, d)).reshape(B, 1, N)
                s = inner.step(s, fit, params)
        else:
            s = graph.run(oracle, s, params, steps)
        cost, local_start = oracle.used-start, oracle.used
        quota = min(cost//4, oracle.remaining) if method != 'original' else 0
        if quota:
            ids = select(s, min(B, max(1, int(np.ceil(quota/5000)))))
            for i in ids:
                cap = min(5000, quota-(oracle.used-local_start), oracle.remaining)
                if cap < (2*d+1 if method == 'cc_four' else 100):
                    break
                x, before = s.gbest[i, 0].numpy().copy(), float(s.gbest_fitness[i, 0])
                detail = {}
                if method == 'cc_four':
                    quad_cap = min(1500, cap)
                    # The frozen function returns without queries in this case;
                    # avoid constructing its unused O(d^2) index list.
                    if 1+2*d+d*(d-1)//2+8 > quad_cap:
                        xx, ff, used = x, np.inf, 0
                    else:
                        xx, ff, used = e.polish_quad(oracle, x, quad_cap)
                    e.inject(s, i, xx, ff)
                    if ff < before:
                        x = xx
                    if cap-used >= 2*d+1:
                        xx, ff, n = e.polish_bfgs(oracle, x, min(1200, cap-used))
                        used += n
                        e.inject(s, i, xx, ff)
                    if cap-used >= 2*d+1:
                        xx, ff, n = ray_search(oracle, s.gbest[i, 0].numpy().copy(), rng, cap-used)
                        used += n
                        e.inject(s, i, xx, ff)
                else:
                    validate = method != 'cvbr_hybrid_no_split'
                    if (block+i) % 2 == 0 and ablation != 'no_curvature':
                        xx, ff, used = ray_search(oracle, x, rng, cap, polish=ablation != 'no_polish')
                        detail = {'operator': 'curvature_ray'}
                        if used == 0:
                            if ablation == 'no_cvbr':
                                xx, ff, used = e.polish_bfgs(oracle, x, cap)
                                detail = {'operator': 'polish_only_fallback'}
                            else:
                                xx, ff, used, detail = block_recombine(oracle, x, rng, cap, validate=validate, polish=ablation != 'no_polish')
                                detail['operator'] = 'block_fallback'
                    else:
                        if ablation == 'no_cvbr':
                            xx, ff, used = e.polish_bfgs(oracle, x, cap)
                            detail = {'operator': 'polish_only'}
                        else:
                            xx, ff, used, detail = block_recombine(oracle, x, rng, cap, validate=validate, polish=ablation != 'no_polish')
                            detail['operator'] = 'block'
                    e.inject(s, i, xx, ff)
                logs.append(dict(block=block, lineage=i, before=before,
                                 after=float(s.gbest_fitness[i, 0]), fes=used, **detail))
        scores = s.gbest_fitness[:, 0].clone()
        if method != 'original':
            outer.pbest_fitness.fill_(torch.inf)
            outer.gbest_fitness.fill_(torch.inf)
        theta = outer.population.clone()
        if policy_observer is not None:
            policy_observer(theta, scores, block, oracle.used)
        if fixed_policy is None:
            outer.step(scores)
        ids = select(s, survivors)
        epochs.append(dict(block=block, fes=oracle.used, best=oracle.best, selected=ids,
                           search_fes=cost, local_fes=oracle.used-local_start))
        if method != 'original':
            s = e.get_state(s, torch.tensor([ids[j % survivors] for j in range(B)]))
            for j, i in enumerate(ids):
                outer.population[j] = theta[i]
                outer.velocity[j].zero_()
            fresh = B//4
            new = inner.setup(fresh)
            sl = slice(B-fresh, B)
            e.set_state(s, sl, new)
            if fixed_policy is None:
                outer.population[sl] = torch.rand((fresh, len(high)))*high
            else:
                outer.population[:] = fixed_policy
            outer.velocity[sl].zero_()
            if ablation == 'no_inheritance':
                # Preserve policy selection/update but discard all particle memory.
                s = inner.setup(B)
        block += 1
    stopped_by_rounds = max_rounds is not None and block >= max_rounds
    if not stopped_by_rounds and oracle.remaining:
        if s is None:
            s = inner.setup(B)
        oracle(s.population.reshape(-1, d)[:oracle.remaining])
    if not stopped_by_rounds and oracle.used != oracle.budget:
        raise RuntimeError('Incorrect function-evaluation accounting.')
    return epochs, logs
