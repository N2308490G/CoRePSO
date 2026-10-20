"""Fixed-FE EvoX baselines on the shared numerical objectives; no learned agents."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/evox/src'))
from evox import algorithms
from evox.core import Problem, Mutable
from evox.workflows import StdWorkflow
from cuda_objectives import NumericalObjective, independent_verify, PROTOCOL

METHODS = ('PSO', 'CSO', 'CLPSO', 'SLPSOGS', 'SLPSOUS')


class ObjectiveBridge(Problem):
    def __init__(self, case, device):
        super().__init__()
        self.objective = NumericalObjective(case, device)
        self.best = Mutable(torch.full((), torch.inf, device=device))
        self.best_z = Mutable(torch.zeros(case['dim'], device=device))
        self.fes = Mutable(torch.zeros((), dtype=torch.int64, device=device))
        self.invalid = Mutable(torch.zeros((), dtype=torch.bool, device=device))

    def evaluate(self, z):
        values = self.objective.value(z)
        self.invalid.logical_or_(~torch.isfinite(z).all() | ~torch.isfinite(values).all()
                                 | (z.abs() > 5.00000001).any())
        idx = values.argmin().reshape(1)
        value = values.index_select(0, idx).squeeze(0)
        position = z.index_select(0, idx).squeeze(0)
        improve = value < self.best
        self.best_z.copy_(torch.where(improve, position, self.best_z))
        self.best.copy_(torch.where(improve, value, self.best))
        self.fes.add_(len(z))
        return values


def make_workflow(case, method, seed, population, device):
    torch.manual_seed(seed)
    problem = ObjectiveBridge(case, device)
    bound = torch.full((case['dim'],), 5., device=device)
    algorithm = getattr(algorithms, method)(pop_size=population, lb=-bound, ub=bound, device=device)
    flow = StdWorkflow(algorithm, problem, device=device)
    flow.init_step()
    return flow


class ReplaySteps:
    """Restore buffer identities after each functional EvoX state update."""
    def __init__(self, flow, steps=8):
        self.flow = flow
        self.steps = steps
        # Initial best/population tensors can share storage in upstream EvoX.
        # Give each named state its own storage before replay copies updates back.
        flow.load_state_dict({k:v.detach().clone() for k,v in flow.state_dict().items()}, assign=True)
        self.state = flow.state_dict(keep_vars=True)
        initial = {k: v.detach().clone() for k, v in self.state.items()}
        rng = torch.cuda.get_rng_state()
        cpu_rng = torch.get_rng_state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.eager(2)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.restore(initial)
        torch.cuda.set_rng_state(rng)
        torch.set_rng_state(cpu_rng)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.eager(steps)
        torch.cuda.synchronize()
        self.restore(initial)
        torch.cuda.set_rng_state(rng)
        torch.set_rng_state(cpu_rng)

    def restore(self, values):
        with torch.no_grad():
            for k, tensor in self.state.items():
                tensor.copy_(values[k])
        self.flow.load_state_dict(self.state, assign=True)

    def eager(self, steps):
        for _ in range(steps):
            self.flow.step()
            updated = self.flow.state_dict(keep_vars=True)
            with torch.no_grad():
                for k, tensor in self.state.items():
                    if tensor is not updated[k]:
                        tensor.copy_(updated[k])
            self.flow.load_state_dict(self.state, assign=True)

    def replay(self):
        self.graph.replay()


def run(case, method, seed, budget, population, device='cuda', graph=True, progress=None):
    step_fes = population // 2 if method == 'CSO' else population
    if population < 4 or population % 2 or budget < population or (budget-population) % step_fes:
        raise ValueError('Budget must cover initialization and an integer number of full updates')
    start = time.perf_counter()
    flow = make_workflow(case, method, seed, population, device)
    block = ReplaySteps(flow) if graph else None
    problem = flow.problem
    used = population
    capture_seconds = time.perf_counter()-start
    trace = [[used, float(problem.best)]]
    last_report = time.perf_counter()
    while used < budget:
        n = 8 if block is not None and budget-used >= 8*step_fes else 1
        if n == 8:
            block.replay()
        elif block is not None:
            block.eager(1)
        else:
            flow.step()
        used += n*step_fes
        if time.perf_counter()-last_report >= 60 or used == budget:
            if int(problem.fes) != used or bool(problem.invalid):
                raise RuntimeError('Invalid candidate/fitness or incorrect FE accounting')
            trace.append([used, float(problem.best)])
            if progress:
                progress(used, budget, float(problem.best))
            last_report = time.perf_counter()
    if bool(problem.invalid) or int(problem.fes) != budget:
        raise RuntimeError('Invalid completed run')
    z = problem.best_z.detach().cpu().numpy()
    verified = independent_verify(case, z)
    if not np.isclose(verified, float(problem.best), rtol=1e-9, atol=1e-6):
        raise RuntimeError('Independent verification failed')
    result = dict(**case, method=method, seed=seed, budget=budget, fes=used,
                  verified=True, verified_error=verified, best_z=z.tolist(), trace=trace,
                  population=population, elapsed_seconds=time.perf_counter()-start,
                  capture_seconds=capture_seconds, objective_protocol=PROTOCOL,
                  execution='evox_cuda_graph_8steps' if graph else 'evox_eager',
                  dtype='float64', device=device, paper_strict_replication=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if cfg['objective_protocol'] != PROTOCOL or any(m not in METHODS for m in cfg['methods']):
        raise ValueError('Unsupported protocol or method')
    if len(set(cfg['methods'])) != len(cfg['methods']) or len(set(cfg['reps'])) != len(cfg['reps']):
        raise ValueError('Duplicate method or seed')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    a.output.mkdir(parents=True, exist_ok=True)
    plan = a.output / 'plan.json'
    if plan.exists() and json.loads(plan.read_text()) != cfg:
        raise RuntimeError('Existing output has a different plan')
    plan.write_text(json.dumps(cfg, indent=2))
    (a.output / 'environment.json').write_text(json.dumps(dict(
        python=sys.version, torch=torch.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        baseline_parameters={m:{k:v.default for k,v in inspect.signature(getattr(algorithms,m)).parameters.items()
                                if isinstance(v.default,(int,float,str,bool))} for m in cfg['methods']}), indent=2))
    output = a.output / 'runs.jsonl'
    done = set()
    if output.exists():
        for line in output.open():
            r = json.loads(line)
            key = (r['case_index'], r['replicate'], r['method'])
            expected = cfg['cec_budget'] if cfg['cases'][key[0]]['group']=='cec2022' else cfg['other_budget']
            if key in done or not r['verified'] or r['fes'] != expected or r['objective_protocol'] != PROTOCOL:
                raise RuntimeError('Invalid saved run')
            done.add(key)
    status = dict(status='running', pid=os.getpid(), completed=len(done),
                  total=len(cfg['cases'])*len(cfg['reps'])*len(cfg['methods']))
    def save():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        (a.output / 'status.json').write_text(json.dumps(status, indent=2))
    try:
        save()
        for idx,c in enumerate(cfg['cases']):
            budget = cfg['cec_budget'] if c['group']=='cec2022' else cfg['other_budget']
            for j,rep in enumerate(cfg['reps']):
                methods = cfg['methods'][j % len(cfg['methods']):] + cfg['methods'][:j % len(cfg['methods'])]
                for method in methods:
                    key = idx,rep,method
                    if key in done:
                        continue
                    status['current'] = dict(case_index=idx, case=c, replicate=rep, method=method)
                    save()
                    print('RUN_START',idx,rep,method,flush=True)
                    def progress(used,total,best):
                        status['current'].update(fes=used,budget=total,best=best)
                        save()
                        print('PROGRESS',idx,rep,method,used,total,best,flush=True)
                    row = run(c,method,7010300+100003*rep+7919*idx,budget,cfg['population'],progress=progress)
                    row.update(case_index=idx,replicate=rep,completed_utc=datetime.now(timezone.utc).isoformat(),
                               runtime=dict(host=os.environ.get('COMPUTERNAME') or os.uname().nodename,
                                            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                            cuda=torch.version.cuda, python=sys.version))
                    with output.open('a') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                    done.add(key)
                    status['completed'] = len(done)
                    save()
                    print('RESULT',idx,rep,method,row['verified_error'],flush=True)
        status.update(status='completed',current=None)
    except BaseException as exc:
        status.update(status='failed',detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
