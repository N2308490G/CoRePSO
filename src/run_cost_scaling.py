"""Measured CoRePSO population/dimension costs, with explicit fixed-round work."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
import gc
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from cuda_objectives import NumericalObjective,independent_verify,PROTOCOL
from cuda_search import CudaOracle,SwarmGraph,ValueGraphs,optimize
from run_scope_cuda import environment

def configurations():
    configs=[]
    for N,B in [(10,10),(10,50),(50,10),(50,50),(50,100),(100,50),(100,100)]:
        for name in ['ackley','griewank','sphere']:
            configs.append(dict(kind='population',name=name,dim=200,B=B,N=N,device='cuda'))
    for dim in [16,32,64,128,256,1024,2048,4096,8192]:
        for name in ['ackley','griewank','sphere']:
            configs.append(dict(kind='dimension',name=name,dim=dim,B=16,N=40,device='cuda'))
        configs.append(dict(kind='dimension',name='ackley',dim=dim,B=16,N=40,device='cpu'))
    return configs

def measure(cfg,rep,rounds):
    case=dict(group='canonical',name=cfg['name'],dim=cfg['dim'],instance=0)
    setup=time.perf_counter();obj=NumericalObjective(case,cfg['device'])
    B,N=cfg['B'],cfg['N'];gpu=cfg['device']=='cuda'
    graph=SwarmGraph(obj,B,N) if gpu else None
    cache=ValueGraphs(obj) if gpu else None
    if gpu:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    setup=time.perf_counter()-setup
    # This upper bound allows every round its full population and local allowance.
    budget=rounds*100*B*N*5//4+B*N
    oracle=CudaOracle(obj,budget,cache)
    seed=20260920+rep*100003+cfg['dim']*7919+B*31+N
    start=time.perf_counter()
    epochs,local=optimize(oracle,seed,'cvbr_hybrid',B,N,graph,max_rounds=rounds)
    if gpu:torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    value=independent_verify(case,oracle.best_x.numpy())
    if len(epochs)!=rounds or not np.isclose(value,oracle.best,rtol=1e-9,atol=1e-6):raise RuntimeError('Workload or endpoint verification failed')
    return dict(**cfg,replicate=rep,rounds=rounds,seed=seed,seconds=elapsed,setup_seconds=setup,
        fes=oracle.used,search_fes=sum(e['search_fes'] for e in epochs),local_fes=sum(e['local_fes'] for e in epochs),
        verified=True,verified_error=value,peak_cuda_memory=torch.cuda.max_memory_allocated() if gpu else None,
        objective_protocol=PROTOCOL,completed_unix=time.time())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    if not torch.cuda.is_available():raise RuntimeError('CUDA comparison requires a GPU on the measurement host')
    cfgs=configurations()
    if a.smoke:cfgs=[dict(kind='dimension',name='ackley',dim=8192,B=16,N=40,device=d) for d in ['cuda','cpu']]
    reps=[0] if a.smoke else list(range(11));rounds=1 if a.smoke else 10
    plan=dict(configurations=cfgs,reps=reps,rounds=rounds,total=len(cfgs)*len(reps),smoke=a.smoke,
        objective_protocol=PROTOCOL,scope='Full CoRePSO, H=100; fixed coordination rounds, not equal-FE accuracy comparisons.',
        timing='Search/controller/local compilation included; objective and swarm graph preparation measured separately.',
        dimension_population='B=16,N=40 matches the primary non-CEC setting; reference used B=N=100 on an RTX3090.',
        cpu='One thread; CPU objective and population path; same round count, B, N, dtype and functions.')
    a.output.mkdir(parents=True,exist_ok=True);pp=a.output/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise RuntimeError('Plan mismatch')
    pp.write_text(json.dumps(plan,indent=2))
    env=environment()
    env.update(hostname=platform.node(),cpu_model=platform.processor())
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        models=[line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')]
        if models:env['cpu_model']=models[0]
    (a.output/'environment.json').write_text(json.dumps(env,indent=2))
    rp=a.output/'runs.jsonl';done=set()
    if rp.exists():
        for line in rp.open():
            r=json.loads(line);key=(r['configuration_index'],r['replicate'])
            if key in done or not r['verified'] or r['rounds']!=rounds:raise RuntimeError('Invalid saved result')
            done.add(key)
    status=dict(status='running',pid=os.getpid(),completed=len(done),total=plan['total'])
    def save():
        status['updated_unix']=time.time();(a.output/'status.json').write_text(json.dumps(status,indent=2))
    try:
        for i,cfg in enumerate(cfgs):
            for rep in reps:
                if (i,rep) in done:continue
                status['current']=dict(configuration_index=i,replicate=rep,**cfg);save()
                row=measure(cfg,rep,rounds);row['configuration_index']=i
                with rp.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                done.add((i,rep));status['completed']=len(done);save()
                print('RESULT',i,rep,cfg['device'],cfg['dim'],round(row['seconds'],3),flush=True)
                gc.collect();torch.cuda.empty_cache()
        status.update(status='completed',current=None)
    except BaseException as exc:
        status.update(status='failed',detail=repr(exc));raise
    finally:save()

if __name__=='__main__':main()
