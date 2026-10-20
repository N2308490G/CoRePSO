"""Final-checkpoint evaluation with a hard oracle-call cap; completed-run resume."""
import argparse
import importlib
import json
import os
import pickle
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from metabbo_runtime import load_method
from metabbo_cec_bridge import MetaBBOCECProblem, EvaluationBudgetExceeded
from cuda_objectives import independent_verify, PROTOCOL


def evaluate(method, fid, seed, budget):
    _, Optimizer = load_method(method)
    checkpoint = Path('results_metabbo_20260915/train') / method / 'resume.pkl'
    with checkpoint.open('rb') as f:
        saved = pickle.load(f)  # Locally generated trusted checkpoint only.
    if saved['done'] != 1200:
        raise RuntimeError('Final 100-epoch checkpoint required')
    agent = saved['agent']
    agent.config.maxFEs = budget
    agent.config.log_interval = max(1, budget // 50)
    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
        elif isinstance(value, dict):
            for module in value.values():
                if isinstance(module, torch.nn.Module):
                    module.eval()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    problem = MetaBBOCECProblem(fid, max_fes=budget, device='cuda')
    optimizer = Optimizer(agent.config)
    if method == 'SYMBOL':
        optimizer.is_train = False
    Env = importlib.import_module('corepso_metabox.environment.basic_environment').PBO_Env
    start = time.perf_counter()
    reason = 'optimizer_stop'
    with torch.no_grad():
        try:
            agent.rollout_episode(Env(problem, optimizer), seed=seed, required_info={})
        except EvaluationBudgetExceeded:
            reason = 'next_batch_exceeds_cap'
    assert 0 < problem.fes <= budget and problem.ledger['fes'] == problem.fes
    value = independent_verify(problem.case, problem.best_z)
    if not np.isclose(value, problem.best, rtol=1e-9, atol=1e-6):
        raise RuntimeError('Independent verification failed')
    return dict(method=method, fid=fid, dim=20, seed=seed, budget=budget,
        actual_fes=problem.fes, unused_fes=budget-problem.fes, verified_error=value,
        best_z=problem.best_z.tolist(), verified=True, stop_reason=reason,
        seconds=time.perf_counter()-start, objective_protocol=PROTOCOL,
        execution=dict(host=platform.node(), platform=platform.system(),
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            cuda=torch.version.cuda, numpy=np.__version__,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-runs',type=int);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    a.output.mkdir(parents=True,exist_ok=True)
    plan=dict(methods=['DEDQN','RLHPSDE','RLEPSO','GLEET','SYMBOL'],
        functions=[1,7] if a.smoke else list(range(1,13)),
        replicates=[970701] if a.smoke else list(range(970701,970712)),
        budget=2107 if a.smoke else 24800000,smoke=a.smoke,
        selection='final 100-epoch checkpoint',scope='supplementary trained-baseline evaluation',
        budget_policy='reject whole over-budget batch; report unused FEs')
    pp=a.output/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise RuntimeError('Plan mismatch')
    pp.write_text(json.dumps(plan,indent=2))
    rp=a.output/'runs.jsonl';done=set()
    if rp.exists():
        for line in rp.open():
            r=json.loads(line);key=(r['method'],r['fid'],r['replicate'])
            if key in done or not r['verified'] or r['budget']!=plan['budget']:raise RuntimeError('Invalid saved run')
            done.add(key)
    status=dict(status='running',pid=os.getpid(),completed=len(done),total=len(plan['methods'])*len(plan['functions'])*len(plan['replicates']))
    def save():
        status['updated_unix']=time.time();(a.output/'status.json').write_text(json.dumps(status,indent=2))
    started_count=len(done)
    try:
        for fid in plan['functions']:
            for rep in plan['replicates']:
                for method in plan['methods']:
                    if (method,fid,rep) in done:continue
                    status.update(method=method,fid=fid,replicate=rep);save()
                    seed=(7010300+100003*rep+7919*(12+fid-1))%(2**32)
                    r=evaluate(method,fid,seed,plan['budget']);r['replicate']=rep
                    with rp.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
                    status['completed']+=1;save();print(method,fid,rep,r['actual_fes'],r['verified_error'],flush=True)
                    if a.max_runs and status['completed']-started_count>=a.max_runs:
                        status['status']='pending';return
        status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',detail=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
