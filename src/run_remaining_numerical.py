"""Missing partition and frozen-configuration transfer experiments only.

Deadlines follow the recorded local time protocol, not the paused uniform-clock
campaign. Existing adaptive target results are reused by the manuscript builder.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from cuda_objectives import NumericalObjective,cases,independent_verify,PROTOCOL
from cuda_search import ValueGraphs,optimize
from run_time_cuda import TimedGraph,TimedOracle,Deadline
from run_scope_cuda import environment
from partition_policy import PartitionPolicy

REPS=list(range(970701,970712))
PARTITION_FIDS=[2,3,6,7,10,12]
VARIANTS={'one':(1,False),'fixed_half':(2,True),'three':(3,False),'four':(4,False)}
SOURCES=[17,19,21,23,40]
TARGETS=[12,13,14,15,16,18,20,22]


def jobs(stage,smoke=False):
    reps=REPS[:1] if smoke else REPS
    if stage=='partition':
        return [dict(case_index=11+f,replicate=r,variant=v,seconds_budget=120.)
                for f in (PARTITION_FIDS[:1] if smoke else PARTITION_FIDS)
                for r in reps for v in VARIANTS]
    if stage=='sources':
        return [dict(case_index=i,replicate=r,variant='source',seconds_budget=300. if i==40 else 120.)
                for i in SOURCES for r in reps]
    if stage=='transfer':
        pairs=[(source,target) for source in SOURCES[:4] for target in TARGETS]+[(17,40),(40,17)]
        return [dict(case_index=t,source_index=s,replicate=r,variant='frozen',
                     seconds_budget=300. if t==40 else 120.) for s,t in pairs for r in reps]
    if stage=='cross_baselines':
        return [dict(case_index=40,replicate=r,variant='baseline_'+m,seconds_budget=300.)
                for r in reps for m in ['original','PSO','CSO','CLPSO','SLPSOGS','SLPSOUS']]
    raise ValueError(stage)


def key(job):
    return (job['case_index'],job['replicate'],job['variant'],job.get('source_index'))


class SourceObserver:
    def __init__(self):
        self.selected=None
    def __call__(self,theta,scores,block,fes):
        i=int(scores.argmin()); value=float(scores[i])
        if self.selected is None or value < self.selected['selection_error']:
            self.selected=dict(theta=theta[i].tolist(),selection_error=value,
                               selected_block=block,selected_lineage=i,selected_fes=fes)


def execute(job,objective,graph,policy=None,frozen=None):
    idx,rep=job['case_index'],job['replicate']
    B,N=(100,100) if idx<24 else (16,40)
    method=job['variant'][9:] if job['variant'].startswith('baseline_') else 'cvbr_hybrid'
    if method not in ('cvbr_hybrid','original'):
        from run_external_time import run_timed
        row=run_timed(cases()[idx],method,7010300+100003*rep+7919*idx,job['seconds_budget'])
        row.update(job);row['completed_unix']=time.time();return row
    cache=ValueGraphs(objective)
    observer=SourceObserver() if job['variant']=='source' else None
    torch.cuda.synchronize()
    oracle=TimedOracle(objective,job['seconds_budget'],cache)
    seed=7010300+100003*rep+7919*idx
    try:
        optimize(oracle,seed,method,B,N,graph,policy=policy,
                 fixed_policy=None if frozen is None else frozen['theta'],policy_observer=observer)
    except Deadline:
        pass
    torch.cuda.synchronize()
    if oracle.best_x is None:
        raise RuntimeError('No completed candidate before deadline')
    value=independent_verify(cases()[idx],oracle.best_x.numpy())
    if not np.isclose(value,oracle.best,rtol=1e-9,atol=1e-6):
        raise RuntimeError('Independent objective verification failed')
    row=dict(**job,**cases()[idx],seed=seed,method=method,verified=True,
             verified_error=value,best_z=oracle.best_x.tolist(),timed_trace=oracle.timed_trace,
             attempted_fes=oracle.used,late_fes=oracle.late_fes,on_time_fes=oracle.used-oracle.late_fes,
             elapsed_seconds=time.perf_counter()-oracle.started,objective_protocol=PROTOCOL,
             completed_unix=time.time())
    if observer is not None:
        if observer.selected is None:
            raise RuntimeError('No full source round completed; cannot export an evaluated policy')
        row['frozen_configuration']=observer.selected
    if frozen is not None:
        row['frozen_configuration']=frozen
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['partition','sources','transfer','cross_baselines'],required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--max-runs',type=int)
    a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    todo=jobs(a.stage,a.smoke)
    if a.smoke:
        for j in todo:j['seconds_budget']=3.
    out=a.output/a.stage;out.mkdir(parents=True,exist_ok=True)
    plan=dict(jobs=todo,total=len(todo),smoke=a.smoke,objective_protocol=PROTOCOL,
              device_requirement='local RTX 4060 Ti; same device as existing adaptive time endpoints',
              timing='Recorded time protocol: shared swarm capture excluded, optimizer initialization and local captures included; synchronized eight-step checks; late values excluded.',
              policy='Shared inertia; two accelerations and exemplars per subgroup; K-1 sorted searched cuts. Full-population exemplars. Fixed-half removes the cut coordinate.',
              transfer='Per-replicate configuration selected only on its source, at a completed CoRePSO round. Target freezes the ten policy coordinates, disables outer policy updates and policy randomization, and retains CoRePSO state feedback and local operators.',
              reuse='Default two-group target endpoints reuse the completed local recorded-time series; no uniform-time rerun.')
    pp=out/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise RuntimeError('Plan mismatch')
    pp.write_text(json.dumps(plan,indent=2))
    env=environment()
    if not a.smoke and '4060 Ti' not in env['gpu']:
        raise RuntimeError('Formal targets must use the existing local comparison hardware')
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    rp=out/'runs.jsonl';done=set();expected={key(j):j for j in todo}
    if rp.exists():
        for line in rp.open():
            row=json.loads(line);k=key(row)
            if k not in expected or k in done or not row['verified'] or row['objective_protocol']!=PROTOCOL or row['seconds_budget']!=expected[k]['seconds_budget']:
                raise RuntimeError('Invalid saved result')
            done.add(k)
    frozen={}
    if a.stage=='transfer':
        for line in (a.output/'sources/runs.jsonl').open():
            row=json.loads(line)
            if not row['verified'] or row['objective_protocol']!=PROTOCOL:raise RuntimeError('Invalid source record')
            frozen[(row['case_index'],row['replicate'])]=row['frozen_configuration']
    status=dict(status='running',pid=os.getpid(),total=len(todo),completed=len(done))
    def save():
        status['updated_unix']=time.time();(out/'status.json').write_text(json.dumps(status,indent=2))
    graph=obj=policy=None;previous=None;count=0
    try:
        for job in todo:
            if key(job) in done:continue
            identity=(job['case_index'],job['variant'])
            if identity!=previous:
                del graph,obj;gc.collect();torch.cuda.empty_cache()
                policy=PartitionPolicy(*VARIANTS[job['variant']]) if a.stage=='partition' else None
                external=job['variant'].startswith('baseline_') and job['variant']!='baseline_original'
                obj=None if external else NumericalObjective(cases()[job['case_index']])
                B,N=(100,100) if job['case_index']<24 else (16,40)
                graph=None if external else TimedGraph(obj,B,N,policy=policy);previous=identity
            status['current']=job;save();print('START',job,flush=True)
            cfg=frozen[(job['source_index'],job['replicate'])] if a.stage=='transfer' else None
            row=execute(job,obj,graph,policy,cfg)
            with rp.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            done.add(key(job));count+=1;status['completed']=len(done);save()
            print('RESULT',key(job),row['verified_error'],flush=True)
            if a.max_runs and count>=a.max_runs:
                status['status']='pending';return
        status.update(status='completed',current=None)
    except BaseException as exc:
        status.update(status='failed',detail=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
