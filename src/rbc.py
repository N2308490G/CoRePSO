"""Progress-credited persistent AutoPSO: experimental candidate, not a novelty claim.

All benchmark observations arrive through a value-only budgeted oracle. This module
never imports a benchmark, function identifier, optimum, shift, or rotation matrix.
It retains the author's original Torch inner update. Only state retention, policy
allocation, and paid geometric population renewal are changed.
"""
from __future__ import annotations
import numpy as np
import torch
from legacy_core import upstream, State
from experiment_v3 import polish_bfgs, polish_quad, select_diverse, inject, get_state, set_state


def probe_shape(oracle, x: np.ndarray, cap: int):
    """Paid quadratic probe. Return an inverse-curvature sampling shape, not a Hessian oracle."""
    d = len(x)
    pairs = np.array([(i,j) for i in range(d) for j in range(i+1,d)], int)
    cost = 1+2*d+len(pairs)
    if min(cap, oracle.remaining) < cost:
        return None, 0
    step = np.where(x>0,-1.,1.)*0.002
    e = np.eye(d)*step
    points = np.concatenate([x[None,:],x+e,x+2*e,x+e[pairs[:,0]]+e[pairs[:,1]]])
    vals = oracle(points)
    f0 = vals[0]; f1 = vals[1:1+d]; f2 = vals[1+d:1+2*d]
    H = np.diag((f2-2*f1+f0)/(step*step))
    cross = (vals[1+2*d:]-f1[pairs[:,0]]-f1[pairs[:,1]]+f0)/(step[pairs[:,0]]*step[pairs[:,1]])
    H[pairs[:,0],pairs[:,1]]=cross; H[pairs[:,1],pairs[:,0]]=cross
    w, q = np.linalg.eigh(H)
    scale=max(float(np.max(np.abs(w))),1e-12)
    ev=np.maximum(w,scale*1e-6)
    axes=np.sqrt(ev.min()/ev)
    # Some isotropic support is retained; probe uncertainty cannot suppress a dimension.
    axes=np.maximum(axes,0.003)
    return q*axes[None,:],cost


def policy_mutation(theta: torch.Tensor, donors: np.ndarray, rng, hi):
    """Keep categorical exemplar identities categorical; do not interpolate their labels."""
    out=theta[torch.as_tensor(donors)].clone()
    con=[0,1,2,3,4,9]
    out[:,con] += torch.from_numpy(rng.normal(0,0.12,(len(donors),len(con))))
    for j in range(5,9):
        mask=rng.random(len(donors))<0.18
        if mask.any():out[torch.from_numpy(mask),j]=torch.from_numpy(rng.integers(0,9,mask.sum()).astype(float))
    return torch.maximum(torch.zeros_like(out),torch.minimum(out,hi))


@torch.no_grad()
def optimize_rbc(oracle,seed:int,variant:str='rbc_geom',outer_n:int=100,inner_n:int=100,horizon:int=100):
    if variant not in ('rbc','rbc_geom','rbc_noprogress','rbc_nolocal'):
        raise ValueError(variant)
    torch.manual_seed(seed); rng=np.random.default_rng(seed+60191)
    B,N,d=outer_n,inner_n,len(oracle.lb)
    hi=torch.tensor([1.]*5+[8.99999]*4+[1.],dtype=torch.float64)
    outer=upstream.TorchOuterPSO(torch.zeros_like(hi),hi,B)
    theta=outer.population.clone()
    inner=upstream.TorchAutoPSO(torch.from_numpy(oracle.lb),torch.from_numpy(oracle.ub),N)
    state=inner.setup(B)
    stale=np.zeros(B,dtype=int);age=np.zeros(B,dtype=int)
    models={};local=[];epochs=[];credit=np.zeros(B);block=0
    while oracle.remaining>=B*N:
        before=state.gbest_fitness[:,0].numpy().copy()
        steps=min(horizon,oracle.remaining//(B*N))
        params=upstream.decode_autopso_parameters_torch(theta)
        for _ in range(steps):
            fit=oracle(state.population.reshape(-1,d)).reshape(B,1,N)
            state=inner.step(state,fit,params)
        age+=steps
        after=state.gbest_fitness[:,0].numpy().copy()
        finite=np.isfinite(before)
        gain=np.zeros(B)
        mad=max(float(np.median(np.abs(after-np.median(after)))),1e-12)
        gain[finite]=np.maximum(before[finite]-after[finite],0)/(np.abs(before[finite])+mad+1e-12)
        gain[~finite]=np.clip((np.median(after)-after[~finite])/(mad+1e-12),0,1)
        improved=~finite | ((before-after)>1e-10*np.maximum(1,np.abs(np.where(finite,before,0))))
        stale=np.where(improved,0,stale+1)
        credit=.5*credit+.5*gain
        chosen=select_diverse(state,min(8,B))
        if variant!='rbc_nolocal':
            for i in chosen:
                if oracle.remaining<2*d+1:break
                x=state.gbest[i,0].numpy().copy();f0=float(state.gbest_fitness[i,0])
                start=oracle.used
                # Refinement budget and model probes are real charged evaluations.
                xq,fq,_=polish_quad(oracle,x,min(1500,oracle.remaining))
                inject(state,i,xq,fq)
                if fq<f0:x=xq
                xb,fb,_=polish_bfgs(oracle,x,min(3500,oracle.remaining))
                inject(state,i,xb,fb)
                if variant=='rbc_geom':
                    shape,cost=probe_shape(oracle,state.gbest[i,0].numpy().copy(),min(1000,oracle.remaining))
                    if shape is not None:models[i]=shape
                local.append(dict(block=block,lineage=i,before=f0,after=float(state.gbest_fitness[i,0]),fes=oracle.used-start))
        current=state.gbest_fitness[:,0].numpy().copy()
        # Policy selection is based on within-state observed progress, not an inherited incumbent.
        if variant=='rbc_noprogress': donor_order=np.argsort(current)
        else: donor_order=np.lexsort((current,-credit))
        donors=donor_order[:max(4,B//5)]
        best_idx=int(current.argmin())
        change=np.flatnonzero((stale>=1)|(age>=400))
        if len(change):
            donor=rng.choice(donors,len(change),replace=True)
            theta[torch.as_tensor(change)]=policy_mutation(theta,donor,rng,hi)
        # Keep independently progressing lineages. Only old or stalled ones are renewed.
        renew=np.flatnonzero(((stale>=2)|(age>=600)) & (np.arange(B)!=best_idx))
        maxrenew=max(1,B//4)
        if len(renew)>maxrenew:renew=renew[np.argsort(current[renew])[-maxrenew:]]
        for i in renew:
            use_local=(rng.random()<.5)
            fresh=inner.setup(1)
            if use_local:
                # Re-enter this lineage at a random scale rather than cloning the global winner.
                centre=state.gbest[i,0].numpy().copy()
                shape=models.get(int(i),np.eye(d)) if variant=='rbc_geom' else np.eye(d)
                radius=10.**rng.uniform(-2.2,0.25)
                cloud=np.clip(centre+radius*(rng.normal(size=(N,d))@shape.T),oracle.lb,oracle.ub)
                cloud[0]=centre
                fresh.population[0,0]=torch.from_numpy(cloud)
                fresh.velocity[0,0]=torch.from_numpy(.1*radius*rng.normal(size=(N,d)))
                fresh.pbest=fresh.population.clone()
            else:
                theta[i]=torch.from_numpy(rng.random(10))*hi
            set_state(state,int(i),get_state(fresh,0))
            stale[i]=0;age[i]=0;credit[i]=0
            models.pop(int(i),None)
        epochs.append(dict(block=block,fes=oracle.used,best=oracle.best,renewed=len(renew),polished=len(chosen) if variant!='rbc_nolocal' else 0))
        block+=1
    if oracle.remaining:
        oracle(state.population.reshape(-1,d)[:oracle.remaining])
    assert oracle.used==oracle.budget
    return epochs,local
