"""Equal-wall-time paired CUDA comparison; late evaluations never improve scores."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
from collections import Counter
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from cuda_objectives import NumericalObjective, cases, independent_verify, PROTOCOL
from cuda_search import CudaOracle, SwarmGraph, ValueGraphs, optimize
from run_scope_cuda import environment


class Deadline(Exception):
    pass


class TimedOracle(CudaOracle):
    def __init__(self, objective, seconds, cache=None, clock=time.perf_counter, started=None):
        super().__init__(objective, 2**60, cache)
        self.clock = clock
        self.started = clock() if started is None else started
        self.deadline = self.started + seconds
        self.late_fes = 0
        self.timed_trace = []

    def check(self):
        if self.clock() >= self.deadline:
            raise Deadline()

    @property
    def remaining(self):
        self.check()
        return self.budget-self.used

    def snapshot(self):
        self.check()
        return self.best, self.best_x, len(self.trace), self.used

    def accept(self, saved):
        now = self.clock()
        if now > self.deadline:
            self.late_fes += self.used-saved[3]
            self.best, self.best_x = saved[:2]
            del self.trace[saved[2]:]
            raise Deadline()
        self.timed_trace.append((now-self.started, self.used, self.best))

    def __call__(self, x):
        saved = self.snapshot()
        values = super().__call__(x)
        self.accept(saved)
        return values


class TimedGraph(SwarmGraph):
    """Check synchronized completion every eight steps, preserving search updates."""
    def run(self, oracle, state, params, steps):
        for i in range(0, steps, 8):
            saved = oracle.snapshot()
            state = super().run(oracle, state, params, min(8,steps-i))
            oracle.accept(saved)
        return state


def seconds_for(case):
    if case['group']=='cec2022':
        return 60 if case['dim']==10 else 120
    return 30 if case['group']=='canonical' else 300


def summarize(output, plan):
    from scipy.stats import wilcoxon
    values={}
    for line in (output/'runs.jsonl').open():
        r=json.loads(line)
        key=(r['group'],r['name'],r['dim'],r['replicate'],r['method'])
        if key in values or not r['verified'] or not np.isfinite(r['verified_error']) or r['verified_error'] < -1e-6:
            raise ValueError('Duplicate or invalid result')
        values[key]=0. if r['verified_error']<1e-8 else r['verified_error']
    expected={(c['group'],c['name'],c['dim'],rep,m) for c in plan['cases'] for rep in plan['reps'] for m in plan['methods']}
    if set(values)!=expected:
        raise ValueError('Incomplete comparison grid')
    rows=[]
    for c in plan['cases']:
        k=(c['group'],c['name'],c['dim'])
        a=np.array([values[(*k,rep,'cvbr_hybrid')] for rep in plan['reps']])
        b=np.array([values[(*k,rep,'original')] for rep in plan['reps']])
        diff=a-b
        diff[np.abs(diff)<=1e-8+1e-9*np.maximum(np.abs(a),np.abs(b))]=0
        tol=1e-8+1e-9*max(abs(a.mean()),abs(b.mean()))
        relation='win' if a.mean()<b.mean()-tol else 'loss' if a.mean()>b.mean()+tol else 'tie'
        rows.append(dict(group=k[0],name=k[1],dim=k[2],n=len(a),cvbr_mean=float(a.mean()),
                         cvbr_std=float(a.std(ddof=1)) if len(a)>1 else None,
                         original_mean=float(b.mean()),original_std=float(b.std(ddof=1)) if len(b)>1 else None,
                         mean_result=relation,wilcoxon_p=1. if np.all(diff==0) else float(wilcoxon(diff,method='auto').pvalue)))
    ps=np.array([r['wilcoxon_p'] for r in rows]); order=np.argsort(ps); adjusted=np.empty(len(ps))
    adjusted[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
    for r,v in zip(rows,adjusted):
        r['wilcoxon_p_holm']=float(v)
    with (output/'comparison_vs_original.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(dict(completed_runs=len(values),scope=plan['scope'],
        smoke=plan['smoke'],mean_comparisons=dict(Counter(r['mean_result'] for r in rows))),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',nargs='+',type=int)
    p.add_argument('--reps',nargs='+',type=int,default=list(range(970701,970712)))
    p.add_argument('--seconds',type=float,help='Override for explicitly separate smoke runs only')
    a=p.parse_args()
    if len(a.reps)!=len(set(a.reps)) or (a.seconds is not None and (a.seconds<=0 or not np.isfinite(a.seconds))):
        p.error('Invalid seeds or time limit')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    indices=list(range(54)) if a.indices is None else a.indices
    if len(indices)!=len(set(indices)) or any(i<0 or i>=54 for i in indices):
        p.error('Invalid case indices')
    selected=[cases()[i] for i in indices]
    plan=dict(objective_protocol=PROTOCOL,execution_protocol='cuda_wall_time_eight_step_checks_v1',
              cases=selected,reps=a.reps,methods=['original','cvbr_hybrid'],
              seconds=[seconds_for(c) if a.seconds is None else a.seconds for c in selected],
              populations={'cec2022':[100,100],'other':[16,40]},
              scope='Local RTX4060Ti equal-time comparison, 11 seeds; not paper JAX/RTX3090 replication',
              timing='Optimizer initialization, local graph compilation and all search work included; shared swarm capture and independent final verification excluded; late batches excluded from incumbent',
              method_order='Alternating within each paired replicate',smoke=a.seconds is not None,
              total_runs=len(selected)*len(a.reps)*2)
    a.output.mkdir(parents=True,exist_ok=True)
    pp=a.output/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:
        raise RuntimeError('Output plan differs; use a new output directory')
    pp.write_text(json.dumps(plan,indent=2))
    (a.output/'environment.json').write_text(json.dumps(environment(),indent=2))
    done=set(); results=a.output/'runs.jsonl'
    if results.exists():
        for line in results.open():
            r=json.loads(line); key=(r['case_index'],r['replicate'],r['method'])
            if key in done or not r['verified']:
                raise RuntimeError('Duplicate or invalid saved result')
            done.add(key)
    status=dict(status='running',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),completed=len(done),total=plan['total_runs'])
    sp=a.output/'status.json'
    sp.write_text(json.dumps(status,indent=2))
    try:
        for idx,c,limit in zip(indices,selected,plan['seconds']):
            obj=NumericalObjective(c)
            B,N=(100,100) if c['group']=='cec2022' else (16,40)
            graph=TimedGraph(obj,B,N)
            for j,rep in enumerate(a.reps):
                methods=plan['methods'] if j%2==0 else list(reversed(plan['methods']))
                for method in methods:
                    key=(idx,rep,method)
                    if key in done:
                        continue
                    cache=ValueGraphs(obj)
                    torch.cuda.synchronize()
                    oracle=TimedOracle(obj,limit,cache)
                    seed=7010300+100003*rep+7919*idx
                    try:
                        optimize(oracle,seed,method,B,N,graph)
                    except Deadline:
                        pass
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-oracle.started
                    if oracle.best_x is None:
                        raise RuntimeError('No candidate completed before deadline')
                    z=oracle.best_x.numpy()
                    verified=independent_verify(c,z)
                    if not np.isclose(verified,oracle.best,rtol=1e-9,atol=1e-6):
                        raise RuntimeError(f'Independent verification failed: {verified} versus {oracle.best}')
                    row=dict(**c,case_index=idx,replicate=rep,seed=seed,method=method,seconds_budget=limit,
                             elapsed_seconds=elapsed,attempted_fes=oracle.used,late_fes=oracle.late_fes,
                             on_time_fes=oracle.used-oracle.late_fes,error=oracle.best,verified_error=verified,
                             verified=True,best_z=z.tolist(),timed_trace=oracle.timed_trace,
                             objective_protocol=PROTOCOL,completed_utc=datetime.now(timezone.utc).isoformat())
                    with results.open('a') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                    done.add(key)
                    status['completed']=len(done)
                    sp.write_text(json.dumps(status,indent=2))
                    print('RESULT',idx,rep,method,verified,elapsed,flush=True)
                    del oracle,cache
            del graph,obj
            gc.collect()
            torch.cuda.empty_cache()
        summarize(a.output,plan)
        status.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status='failed',detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        sp.write_text(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
