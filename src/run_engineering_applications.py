"""Serial GPU-0 FM fitting and radar phase-code application study.

Completed-record resume only. Full runs and pipeline pilots use separate roots.
All solvers consume the exact same FE allowance; compilation is not an FE.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from engineering_objectives import CASES, PROTOCOL, SOURCE, EngineeringObjective, independent_verify, check_objectives, physical
from cuda_search import CudaOracle, SwarmGraph, ValueGraphs, optimize
from run_external_fe import ReplaySteps, ObjectiveBridge
from evox import algorithms
from evox.core import Problem, Mutable
from evox.workflows import StdWorkflow

METHODS = ['cvbr_hybrid', 'original', 'PSO', 'CSO', 'CLPSO', 'SLPSOGS', 'SLPSOUS',
           'cvbr_hybrid_no_split', 'no_inheritance']
REPS = list(range(981001, 981012))
BUDGET = 1024000
B, N, POPULATION = 16, 40, 100


def plan_for(mode):
    instances = CASES if mode != 'pilot' else CASES[-1:]
    seeds = REPS if mode=='formal' else [80102]
    budget = 100000 if mode=='smoke' else BUDGET
    jobs = [dict(case=case, seed=seed, method=m, budget=budget)
            for case in instances for seed in seeds for m in METHODS]
    return dict(protocol=PROTOCOL, mode=mode, jobs=jobs, total=len(jobs),
                outer_population=B, inner_population=N, baseline_population=POPULATION,
                objective_source=SOURCE,
                radar_definition='Technical report equations (31)-(34), both signs via max(abs(phi)); distributed Matlab case 7 omits the negative branches and is not the definition used here.',
                scope='Engineering application benchmarks, not field measurements or deployed systems; independent 11-run experiment, not a reproduction of the historical paper budgets or results.',
                budget_policy='Exact common FE budget including initialization, finite differences, local search and rejected trials; final scalar verification excluded. Wall time includes per-run construction and graph capture; fixed shared objective validation excluded. No wall-clock superiority claim.',
                initialization='Uniform physical bounds through affine [-5,5] coordinates for all algorithms; common seed identifiers do not imply identical initial populations.',
                statistics='Means and sample SD per instance; independent two-sided Mann-Whitney comparisons, Holm over the 3 instances separately for each comparator; 11 optimization repetitions are not 11 independent problem instances.',
                hardware='physical GPU0 NVIDIA RTX3090 only, float64; no concurrent jobs')


class Bridge(ObjectiveBridge):
    def __init__(self, objective):
        Problem.__init__(self)
        self.objective = objective
        self.best = Mutable(torch.full((), torch.inf, device='cuda'))
        self.best_z = Mutable(torch.zeros(objective.dim, device='cuda'))
        self.fes = Mutable(torch.zeros((), dtype=torch.int64, device='cuda'))
        self.invalid = Mutable(torch.zeros((), dtype=torch.bool, device='cuda'))


def run(job):
    start = time.perf_counter()
    case, method, seed, budget = (job[k] for k in ('case','method','seed','budget'))
    objective = EngineeringObjective(case)
    epochs, operators = [], []
    torch.manual_seed(seed)
    if method in ('cvbr_hybrid', 'original', 'cvbr_hybrid_no_split', 'no_inheritance'):
        oracle = CudaOracle(objective, budget, ValueGraphs(objective))
        graph = SwarmGraph(objective, B, N)
        setup_seconds = time.perf_counter()-start
        epochs, operators = optimize(oracle,seed,'cvbr_hybrid' if method=='no_inheritance' else method,
                                    B,N,graph,ablation='no_inheritance' if method=='no_inheritance' else None)
        best, z, used, trace = oracle.best, oracle.best_x.numpy(), oracle.used, oracle.trace
    else:
        bound = torch.full((case['dim'],), 5., device='cuda')
        algorithm = getattr(algorithms, method)(pop_size=POPULATION, lb=-bound, ub=bound, device='cuda')
        flow = StdWorkflow(algorithm, Bridge(objective), device='cuda')
        flow.init_step()
        blocks = ReplaySteps(flow)
        step_fes = POPULATION//2 if method=='CSO' else POPULATION
        used = POPULATION
        trace = [[used,float(flow.problem.best)]]
        setup_seconds = time.perf_counter()-start
        while used < budget:
            steps = 8 if budget-used >= 8*step_fes else 1
            blocks.replay() if steps==8 else blocks.eager(1)
            used += steps*step_fes
            if used-trace[-1][0]>=10000 or used==budget:
                trace.append([used,float(flow.problem.best)])
        if bool(flow.problem.invalid) or int(flow.problem.fes)!=budget:
            raise RuntimeError('Baseline invalid candidates or incorrect FE count')
        best, z = float(flow.problem.best), flow.problem.best_z.cpu().numpy()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    verified = independent_verify(case, z)
    if used!=budget or not np.isclose(verified,best,rtol=2e-10,atol=2e-8):
        raise RuntimeError('Budget or independent endpoint verification failed')
    if not trace or trace[-1][0]!=budget or np.any(np.diff(np.array(trace)[:,0])<=0):
        raise RuntimeError('Invalid FE trace')
    return dict(**job, protocol=PROTOCOL, verified=True, verified_value=verified, fes=used,
                normalized_best=z.tolist(), physical_best=physical(z,case['kind']).tolist(),
                trace=trace, epochs=epochs, operators=operators,
                elapsed_seconds=elapsed, setup_seconds=setup_seconds, completed_unix=time.time())


def key(row):
    return row['case']['name'],row['seed'],row['method']


def validate_row(row, job, mode):
    if (any(row.get(k)!=v for k,v in job.items()) or row.get('mode')!=mode
            or row.get('protocol')!=PROTOCOL or row.get('verified') is not True
            or row.get('fes')!=job['budget'] or not np.isfinite(row.get('verified_value',np.nan))
            or row.get('hardware',{}).get('physical_gpu')!=0
            or '3090' not in row.get('hardware',{}).get('gpu','')):
        raise ValueError('Invalid saved engineering result')
    value=independent_verify(job['case'],row['normalized_best'])
    if not np.isclose(value,row['verified_value'],rtol=2e-10,atol=2e-8):
        raise ValueError('Saved endpoint does not reproduce')
    trace=np.asarray(row['trace'])
    if (trace.ndim!=2 or trace.shape[1]!=2 or not np.isfinite(trace).all()
            or trace[-1,0]!=job['budget'] or np.any(np.diff(trace[:,0])<=0)
            or np.any(np.diff(trace[:,1])>1e-8)
            or not np.isclose(trace[-1,1],value,rtol=2e-10,atol=2e-8)):
        raise ValueError('Invalid saved trace')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',choices=['formal','smoke','pilot'],default='formal')
    ap.add_argument('--max-runs',type=int)
    a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('Explicit binding to physical GPU0 is required')
    if os.name!='posix':
        raise RuntimeError('This archived engineering protocol requires POSIX and an idle RTX3090 GPU0')
    import fcntl
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    busy=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:
        raise RuntimeError('GPU0 already has a compute process: '+busy)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    hardware=dict(gpu=torch.cuda.get_device_name(0),physical_gpu=0,host=os.uname().nodename,
                  python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__)
    if '3090' not in hardware['gpu']:
        raise RuntimeError('RTX3090 required')
    plan=plan_for(a.mode);pp=a.output/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:
        raise ValueError('Existing plan differs')
    pp.write_text(json.dumps(plan,indent=2))
    (a.output/'environment.json').write_text(json.dumps(hardware,indent=2))
    print('OBJECTIVE_CHECK',check_objectives(),flush=True)
    rp=a.output/'runs.jsonl'; expected={key(j):j for j in plan['jobs']};done=set()
    if rp.exists():
        for line in rp.open():
            row=json.loads(line);k=key(row)
            if k not in expected or k in done:
                raise ValueError('Unexpected or duplicate completed record')
            validate_row(row,expected[k],a.mode);done.add(k)
    status=dict(status='running',mode=a.mode,pid=os.getpid(),completed=len(done),total=len(expected))
    def save():
        status['updated_unix']=time.time()
        temp=a.output/'status.pending'
        temp.write_text(json.dumps(status,indent=2));temp.replace(a.output/'status.json')
    count=0
    try:
        for job in plan['jobs']:
            if key(job) in done:continue
            status['current']=job;save();print('START',job,flush=True)
            row=run(job);row.update(mode=a.mode,hardware=hardware)
            validate_row(row,job,a.mode)
            with rp.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            done.add(key(job));count+=1
            status['completed']=len(done);save()
            print('RESULT',key(job),row['verified_value'],row['elapsed_seconds'],flush=True)
            gc.collect();torch.cuda.empty_cache()
            if a.max_runs and count>=a.max_runs and len(done)<len(expected):
                status.update(status='pending',current=None);return
        status.update(status='completed',current=None)
    except BaseException as exc:
        status.update(status='failed',detail=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
