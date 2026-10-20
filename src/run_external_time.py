"""Local equal-time EvoX baselines. Synchronized late blocks cannot improve results."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse,json,time,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from run_external_fe import METHODS,make_workflow,ReplaySteps,independent_verify,PROTOCOL

def run_timed(case,method,seed,seconds,population=100):
    torch.cuda.synchronize()
    start=time.perf_counter(); deadline=start+seconds
    flow=make_workflow(case,method,seed,population,'cuda')
    torch.cuda.synchronize()
    if time.perf_counter()>deadline: raise RuntimeError('Initialization exceeded budget')
    best=float(flow.problem.best); z=flow.problem.best_z.cpu().numpy().copy(); accepted=int(flow.problem.fes)
    block=ReplaySteps(flow)
    torch.cuda.synchronize()
    trace=[[time.perf_counter()-start,accepted,best]] if time.perf_counter()<=deadline else []
    while time.perf_counter()<deadline:
        block.replay(); torch.cuda.synchronize()
        now=time.perf_counter()
        if bool(flow.problem.invalid): raise RuntimeError('Invalid candidates')
        if now>deadline: break
        best=float(flow.problem.best); z=flow.problem.best_z.cpu().numpy().copy(); accepted=int(flow.problem.fes)
        if not trace or now-start-trace[-1][0]>=10: trace.append([now-start,accepted,best])
    attempted=int(flow.problem.fes)
    value=independent_verify(case,z)
    if not np.isclose(value,best,rtol=1e-9,atol=1e-6): raise RuntimeError('Independent verification failed')
    return dict(**case,method=method,seed=seed,seconds_budget=seconds,verified=True,verified_error=value,best_z=z.tolist(),on_time_fes=accepted,attempted_fes=attempted,late_fes=attempted-accepted,timed_trace=trace,elapsed_seconds=time.perf_counter()-start,objective_protocol=PROTOCOL)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    source=json.loads((Path(__file__).resolve().parents[1]/'configs/numerical_time.json').read_text())
    plan=dict(cases=source['cases'][:1] if a.smoke else source['cases'],reps=[970701] if a.smoke else source['reps'],seconds=[2.] if a.smoke else source['seconds'],methods=list(METHODS),population=100,objective_protocol=PROTOCOL,smoke=a.smoke,timing='Initialization and per-run CUDA graph capture included; synchronized eight-step deadlines; late blocks excluded; independent verification excluded from search budget',scope='Local RTX4060Ti 11-seed extension, not strict paper replication; compare with existing local equal-time results with disclosed capture timing differences')
    a.output.mkdir(parents=True,exist_ok=True); pp=a.output/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan: raise RuntimeError('Plan mismatch')
    pp.write_text(json.dumps(plan,indent=2))
    (a.output/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),indent=2))
    rp=a.output/'runs.jsonl';done=set()
    if rp.exists():
        for line in rp.open():
            r=json.loads(line);k=(r['case_index'],r['replicate'],r['method'])
            if k in done or not r['verified'] or r['seconds_budget']!=plan['seconds'][k[0]] or r['objective_protocol']!=PROTOCOL: raise RuntimeError('Invalid saved result')
            done.add(k)
    status=dict(status='running',pid=os.getpid(),completed=len(done),total=len(plan['cases'])*len(plan['reps'])*5)
    def save():
        status['updated_utc']=datetime.now(timezone.utc).isoformat();(a.output/'status.json').write_text(json.dumps(status,indent=2))
    try:
        for i,c in enumerate(plan['cases']):
            for j,rep in enumerate(plan['reps']):
                for m in list(METHODS[j%5:])+list(METHODS[:j%5]):
                    if (i,rep,m) in done: continue
                    status['current']=dict(case=c,replicate=rep,method=m);save();print('RUN_START',i,rep,m,flush=True)
                    r=run_timed(c,m,7010300+100003*rep+7919*i,plan['seconds'][i]);r.update(case_index=i,replicate=rep,completed_utc=datetime.now(timezone.utc).isoformat())
                    with rp.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
                    done.add((i,rep,m));status['completed']=len(done);save();print('RESULT',i,rep,m,r['verified_error'],flush=True)
        status.update(status='completed',current=None)
    except BaseException as e:
        status.update(status='failed',detail=repr(e));raise
    finally:save()
if __name__=='__main__':main()
