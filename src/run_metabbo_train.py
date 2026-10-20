"""Supplementary 100-epoch training with episode-boundary recovery.

Uses only locally generated trusted checkpoints. No external pickle is loaded.
Training stops after a full upstream update at the default 20,000 FE threshold;
all oracle calls, including SYMBOL's teacher, are recorded separately.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib
import json
import pickle
import random
import time
from pathlib import Path
import numpy as np
import torch
from metabbo_runtime import load_method
from metabbo_cec_bridge import MetaBBOCECProblem
from cuda_objectives import independent_verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', required=True, choices=['DEDQN','RLHPSDE','RLEPSO','GLEET','SYMBOL'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    A, O = load_method(args.method)
    cfgmod = importlib.import_module('corepso_metabox.config')
    c = cfgmod.init_config(cfgmod.get_config([]))
    c.dim, c.device, c.maxFEs = 20, 'cuda', 20000
    c.log_interval, c.max_epoch, c.max_learning_step = 400, 100, 10**12
    c.agent_save_dir, c.train_name = str(out / 'upstream'), args.method
    c.no_tb = True
    plan = dict(method=args.method, epochs=1 if args.smoke else 100,
                functions=[1,7] if args.smoke else list(range(1,13)),
                dim=20, train_seed=3849, episode_stop_threshold=20000,
                checkpoint_selection='final epoch only', smoke=args.smoke,
                scope='Supplementary protocol, not exact AutoPSO pretraining reproduction',
                accounting='Full upstream update at threshold; all actual teacher and student queries recorded; per-oracle safety ceiling 100000',
                device='cuda',dtype='float64')
    pp = out / 'plan.json'
    if pp.exists() and json.loads(pp.read_text()) != plan:
        raise RuntimeError('Training plan mismatch')
    pp.write_text(json.dumps(plan, indent=2))
    cp = out / 'resume.pkl'
    if cp.exists():
        with cp.open('rb') as f:
            saved = pickle.load(f)
        agent, done, records = saved['agent'], saved['done'], saved['records']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        random.seed(3849); np.random.seed(3849); torch.manual_seed(3849)
        agent, done, records = A(c), 0, []
    Env = importlib.import_module('corepso_metabox.environment.basic_environment').PBO_Env
    total = plan['epochs'] * len(plan['functions'])
    status = dict(status='running', pid=os.getpid(), completed=done, total=total, method=args.method)
    def write_status():
        status['updated_unix'] = time.time()
        (out/'status.json').write_text(json.dumps(status, indent=2))
    try:
        for index in range(done, total):
            epoch, offset = divmod(index, len(plan['functions']))
            fid = plan['functions'][offset]
            status.update(epoch=epoch+1, fid=fid); write_status()
            problem = MetaBBOCECProblem(fid, max_fes=100000, device='cuda')
            optimizer = O(agent.config)
            start = time.perf_counter()
            _, info = agent.train_episode([Env(problem, optimizer)],
                [3849+epoch*100+offset*5], para_mode='dummy')
            losses = np.asarray(info['loss'], dtype=float)
            if not len(losses) or not np.isfinite(losses).all():
                raise RuntimeError('Missing or nonfinite learning update')
            ledger = problem.ledger
            ref = independent_verify(problem.case, ledger['best_z'])
            if not np.isclose(ref, ledger['best'], rtol=1e-9, atol=1e-6):
                raise RuntimeError('Independent objective check failed')
            record = dict(epoch=epoch+1, fid=fid, actual_fes=ledger['fes'],
                loss_mean=float(losses.mean()),learn_steps=int(info['learn_steps']),
                elapsed_seconds=time.perf_counter()-start, verified_error=ref)
            records.append(record)
            state = dict(agent=agent,done=index+1,records=records,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            tmp = out/'resume.tmp'
            with tmp.open('wb') as f:
                pickle.dump(state,f)
            tmp.replace(cp)
            # Checkpoint is authoritative; regenerate the small log after recovery.
            (out/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            status['completed']=index+1;write_status()
            print(json.dumps(record),flush=True)
        status['status']='completed'
    except BaseException as exc:
        status.update(status='failed',detail=repr(exc))
        raise
    finally:
        write_status()


if __name__ == '__main__':
    main()
