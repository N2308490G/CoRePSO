"""Reproducible, resumable corrected-objective experiments on one local GPU."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import scipy
import torch
from cuda_objectives import NumericalObjective, cases, independent_verify, PROTOCOL
from cuda_search import CudaOracle, SwarmGraph, ValueGraphs, optimize

METHODS = ('original','cc_four','cvbr_hybrid','cvbr_hybrid_no_split')
DEFAULT_METHODS = ('original','cvbr_hybrid','cvbr_hybrid_no_split')


def environment():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                threads=torch.get_num_threads())


def one_run(case, seed, method, budget, B, N, obj, graph, value_cache=None):
    oracle = CudaOracle(obj, budget, value_cache=value_cache)
    torch.cuda.synchronize()
    started = time.perf_counter()
    epochs, local = optimize(oracle, seed, method, B, N, graph)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    z = oracle.best_x.numpy()
    verified = independent_verify(case, z)
    if not np.isclose(oracle.best, verified, rtol=1e-9, atol=1e-6):
        raise RuntimeError(f'Independent verification failed: GPU={oracle.best}, scalar={verified}')
    return dict(**case, seed=seed, method=method, budget=budget, fes=oracle.used,
                outer_n=B, inner_n=N, inner_horizon=800 if method=='original' else 100,
                error=oracle.best, verified_error=verified, verified=True,
                seconds=seconds, best_z=z.tolist(), trace=oracle.trace,
                epochs=epochs, local=local, objective_protocol=PROTOCOL,
                dtype='float64', device='cuda', optimizer_control_device='cpu',
                rng_protocol='CPU_outer_setup_and_local_numpy; CUDA_native_inner',
                execution='cuda_graph_swarm_blocks', paper_strict_replication=False)


def profile(out):
    records = []
    # Fixed workloads, chosen before seeing algorithm performance; no protocol tuning.
    for idx, B, N in ((7,100,100), (28,16,40), (53,16,40)):
        case = cases()[idx]
        obj = NumericalObjective(case)
        t = time.perf_counter()
        graph = SwarmGraph(obj, B, N)
        value_cache = ValueGraphs(obj)
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter()-t
        for method in ('original','cvbr_hybrid'):
            r = one_run(case, 908817, method, 2048000 if case['group']=='cec2022' else 320000, B, N, obj, graph, value_cache)
            small = {k:v for k,v in r.items() if k not in ('trace','best_z','epochs','local')}
            small.update(stage='throughput_probe_not_performance_evidence',
                         capture_seconds=capture_seconds,
                         max_memory_allocated=torch.cuda.max_memory_allocated(),
                         local_calls=len(r['local']), fes_per_second=r['fes']/r['seconds'])
            records.append(small)
            print(json.dumps(small), flush=True)
        del value_cache, graph, obj
        gc.collect()
        torch.cuda.empty_cache()
    (out/'throughput.json').write_text(json.dumps(records,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--indices', nargs='+', type=int)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(DEFAULT_METHODS))
    p.add_argument('--reps', nargs='+', type=int, default=list(range(960701,960712)))
    p.add_argument('--cec-budget', type=int, default=24800000)
    p.add_argument('--other-budget', type=int, default=1024000)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    if len(a.reps) != len(set(a.reps)) or len(a.methods) != len(set(a.methods)):
        p.error('Duplicate seeds or methods are not allowed.')
    if a.cec_budget <= 0 or a.other_budget <= 0 or not 0 <= a.shard_index < a.shards:
        p.error('Invalid budget or shard.')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    if not torch.cuda.is_available():
        raise RuntimeError('This entry point requires CUDA; no CPU fallback.')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.profile:
        profile(a.output)
        return 0
    selected = cases() if a.indices is None else [cases()[i] for i in a.indices]
    if len({(c['group'],c['name'],c['dim']) for c in selected}) != len(selected):
        p.error('Duplicate cases.')
    plan = dict(objective_protocol=PROTOCOL, cases=selected, reps=a.reps, methods=a.methods,
                cec_budget=a.cec_budget, other_budget=a.other_budget,
                cec_populations=[100,100], other_populations=[16,40],
                primary_endpoint='Final independent scalar error; per-run error <1e-8 scored as zero',
                ties='1e-8+1e-9*max(abs(means))',
                selection=('Fixed CVBR-Hybrid, fixed CC-Four, and only-split-removal control; no per-case selection'
                           if 'cc_four' in a.methods else
                           'Fixed CVBR-Hybrid versus original AutoPSO, with only-split-removal control; no per-case selection'),
                correction='F3 rotated; F7 last segment restored; ordered CEC arithmetic; F4 follows continuous author code',
                scope='New CUDA equal-FE numerical experiment, not original RTX3090/JAX equal-time reproduction',
                total_runs=len(selected)*len(a.reps)*len(a.methods),
                total_fes=sum(a.cec_budget if c['group']=='cec2022' else a.other_budget for c in selected)*len(a.reps)*len(a.methods))
    # Each shard owns separate files; schema is identical for subsequent aggregation.
    suffix = f'_shard{a.shard_index}' if a.shards > 1 else ''
    plan_path = a.output/f'plan{suffix}.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('Existing output uses a different experimental protocol.')
    else:
        plan_path.write_text(json.dumps(plan,indent=2))
    (a.output/f'environment{suffix}.json').write_text(json.dumps(environment(),indent=2))
    output = a.output/f'runs{suffix}.jsonl'
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            r = json.loads(line)
            key = (r['group'],r['name'],r['dim'],r['replicate'],r['method'])
            if key in done or r['objective_protocol'] != PROTOCOL or not r['verified']:
                raise RuntimeError('Duplicate or incompatible saved result.')
            done.add(key)
    print('START', datetime.now(timezone.utc).isoformat(), json.dumps(plan), flush=True)
    completed_now = 0
    for case in selected:
        idx = cases().index(case)
        tasks = [(rep,m) for rep in a.reps for m in a.methods
                 if (idx*len(a.reps)+a.reps.index(rep)) % a.shards == a.shard_index
                 and (case['group'],case['name'],case['dim'],rep,m) not in done]
        if not tasks:
            continue
        B,N = (100,100) if case['group']=='cec2022' else (16,40)
        budget = a.cec_budget if case['group']=='cec2022' else a.other_budget
        obj = NumericalObjective(case)
        started = time.perf_counter()
        graph = SwarmGraph(obj, B, N)
        value_cache = ValueGraphs(obj)
        capture_seconds = time.perf_counter()-started
        for rep,method in tasks:
            seed = 7010300+100003*rep+7919*idx
            try:
                r = one_run(case,seed,method,budget,B,N,obj,graph,value_cache)
                r.update(replicate=rep, capture_seconds=capture_seconds,
                         completed_utc=datetime.now(timezone.utc).isoformat())
            except Exception as exc:
                failure = dict(case=case,replicate=rep,method=method,seed=seed,
                               error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
                               time=datetime.now(timezone.utc).isoformat())
                with (a.output/f'failures{suffix}.jsonl').open('a') as f:
                    f.write(json.dumps(failure)+'\n')
                raise
            with output.open('a') as f:
                f.write(json.dumps(r,allow_nan=False)+'\n')
                f.flush()
            completed_now += 1
            print('RESULT',case['group'],case['name'],case['dim'],rep,method,
                  r['verified_error'],round(r['seconds'],3),flush=True)
        del value_cache, graph, obj
        gc.collect()
        torch.cuda.empty_cache()
    print('COMPLETE',completed_now,datetime.now(timezone.utc).isoformat(),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
