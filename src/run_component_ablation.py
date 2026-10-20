"""One-factor CoRePSO controls, with completed-run resume and fixed FE budgets."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from cuda_objectives import NumericalObjective,cases,independent_verify,PROTOCOL
from cuda_search import CudaOracle,SwarmGraph,ValueGraphs,optimize,ABLATIONS
from run_scope_cuda import environment

DESCRIPTIONS={
 'no_inheritance':'Reset all particle states after each round; retain policy selection and update.',
 'no_cvbr':'Replace CVBR calls by budgeted L-BFGS-B at the input center; retain curvature calls and parity.',
 'no_curvature':'Use CVBR in all local-search slots, retaining their quotas.',
 'no_polish':'Disable L-BFGS-B in both local operators; retain proposals and acceptance.',
 'fitness_only':'Select centers and surviving states by stable fitness order, without spatial separation.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-runs',type=int);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    indices=[17]+[i for i in range(12,24) if i!=17]
    plan=dict(cases=[dict(case_index=i,**cases()[i]) for i in indices],methods=list(ABLATIONS),
        descriptions=DESCRIPTIONS,reps=list(range(960701,960712)),budget=24800000,B=100,N=100,
        objective_protocol=PROTOCOL,total=660,reference='Reuse full CoRePSO and no-splitting endpoints from results_cuda_20260906/main.',
        scope='All 12 CEC-derived 20D cases, fixed one-factor controls; F6 first for timing.',
        rng='7010300+100003*replicate+7919*case_index; same seed set as primary FE runs')
    a.output.mkdir(parents=True,exist_ok=True);pp=a.output/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise RuntimeError('Plan mismatch')
    pp.write_text(json.dumps(plan,indent=2));(a.output/'environment.json').write_text(json.dumps(environment(),indent=2))
    rp=a.output/'runs.jsonl';done=set()
    if rp.exists():
        for line in rp.open():
            r=json.loads(line);key=(r['case_index'],r['replicate'],r['ablation'])
            if key in done or not r['verified'] or r['fes']!=plan['budget'] or r['objective_protocol']!=PROTOCOL:raise RuntimeError('Invalid saved result')
            done.add(key)
    status=dict(status='running',pid=os.getpid(),completed=len(done),total=plan['total'])
    def save():
        status['updated_unix']=time.time();(a.output/'status.json').write_text(json.dumps(status,indent=2))
    count=0
    try:
        for idx in indices:
            c=cases()[idx]
            tasks=[(rep,m) for rep in plan['reps'] for m in plan['methods'] if (idx,rep,m) not in done]
            if not tasks:continue
            obj=NumericalObjective(c);graph=SwarmGraph(obj,100,100);cache=ValueGraphs(obj)
            for rep,ablation in tasks:
                status['current']=dict(case_index=idx,function=c['name'],replicate=rep,ablation=ablation);save()
                seed=7010300+100003*rep+7919*idx
                oracle=CudaOracle(obj,plan['budget'],cache)
                torch.cuda.synchronize();start=time.perf_counter()
                epochs,local=optimize(oracle,seed,'cvbr_hybrid',100,100,graph,ablation)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                z=oracle.best_x.numpy();value=independent_verify(c,z)
                if oracle.used!=plan['budget'] or not np.isclose(value,oracle.best,rtol=1e-9,atol=1e-6):raise RuntimeError('Budget or independent verification failed')
                row=dict(**c,case_index=idx,replicate=rep,seed=seed,method='cvbr_hybrid',ablation=ablation,
                    budget=plan['budget'],fes=oracle.used,verified=True,verified_error=value,best_z=z.tolist(),
                    seconds=seconds,epochs=epochs,local=local,objective_protocol=PROTOCOL,completed_unix=time.time())
                with rp.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                done.add((idx,rep,ablation));count+=1;status['completed']=len(done);save()
                print('RESULT',c['name'],rep,ablation,value,round(seconds,3),flush=True)
                if a.max_runs and count>=a.max_runs:
                    status['status']='pending';return
            del cache,graph,obj;gc.collect();torch.cuda.empty_cache()
        status.update(status='completed',current=None)
    except BaseException as exc:
        status.update(status='failed',detail=repr(exc));raise
    finally:save()

if __name__=='__main__':main()
