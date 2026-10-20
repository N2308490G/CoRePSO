"""Additional research hypotheses after the persistent-bank variants failed development.

No first-in-literature claim: global line search, deflation and restarts are established
ideas. We test their budgeted interaction with the frozen AutoPSO/DCB controller.
"""
from __future__ import annotations
import numpy as np
import torch
import experiment_v3 as e
from rbc import probe_shape


def ray_search(oracle,x,rng,cap=2400,geometry='dither',basis='dither',polish=True):
    start=oracle.used;d=len(x);cap=min(cap,oracle.remaining)
    if cap<max(100,10*d):return x,np.inf,0
    q=np.eye(d)
    if basis=='mix':basis='axis' if rng.random()<.5 else 'dither'
    if basis=='dither':
        pairs=np.array([(i,j) for i in range(d) for j in range(i+1,d)],int)
        n=1+2*d+len(pairs)
        if cap<n+10*d:return x,np.inf,0
        centre=np.clip(x+.02*(oracle.ub-oracle.lb)*rng.standard_normal(d),oracle.lb,oracle.ub)
        h=np.where(centre>0,-1.,1.)*.002;e0=np.eye(d)*h
        pts=np.concatenate([centre[None],centre+e0,centre+2*e0,centre+e0[pairs[:,0]]+e0[pairs[:,1]]])
        f=oracle(pts);f1=f[1:1+d];f2=f[1+d:1+2*d]
        H=np.diag((f2-2*f1+f[0])/(h*h))
        cross=(f[1+2*d:]-f1[pairs[:,0]]-f1[pairs[:,1]]+f[0])/(h[pairs[:,0]]*h[pairs[:,1]])
        H[pairs[:,0],pairs[:,1]]=cross;H[pairs[:,1],pairs[:,0]]=cross
        _,q=np.linalg.eigh(H)
    rays=q.T
    left=np.full(d,-np.inf);right=np.full(d,np.inf)
    for j in range(d):
        nz=np.abs(rays[j])>1e-12
        a=(oracle.lb[nz]-x[nz])/rays[j,nz];b=(oracle.ub[nz]-x[nz])/rays[j,nz]
        left[j]=np.max(np.minimum(a,b));right[j]=np.min(np.maximum(a,b))
    gridsize=25
    if oracle.used-start+d*gridsize+2*d+10>cap:return x,np.inf,oracle.used-start
    ts=left[:,None]+(right-left)[:,None]*np.linspace(0,1,gridsize)
    points=np.clip(x+ts[:,:,None]*rays[:,None,:],oracle.lb,oracle.ub)
    vals=oracle(points.reshape(-1,d)).reshape(d,gridsize)
    pick=vals.argmin(1);ii=np.arange(d)
    best_t=ts[ii,pick];ray_f=vals[ii,pick]
    a=ts[ii,np.maximum(0,pick-1)];b=ts[ii,np.minimum(gridsize-1,pick+1)]
    # Refine each sampled trough in parallel using charged bounded ternary probes.
    for _ in range(12):
        if oracle.used-start+2*d+10>cap:break
        c=(2*a+b)/3;dd=(a+2*b)/3
        cp=np.clip(x+c[:,None]*rays,oracle.lb,oracle.ub)
        dp=np.clip(x+dd[:,None]*rays,oracle.lb,oracle.ub)
        ff=oracle(np.concatenate([cp,dp]));fc,fd=ff[:d],ff[d:]
        uc=fc<ray_f;best_t[uc]=c[uc];ray_f[uc]=fc[uc]
        ud=fd<ray_f;best_t[ud]=dd[ud];ray_f[ud]=fd[ud]
        low=fc<fd;b=np.where(low,dd,b);a=np.where(low,a,c)
    direction=best_t@rays
    joint=np.clip(x+np.array([1.,.5,.25,.125])[:,None]*direction,oracle.lb,oracle.ub)
    candidates=np.concatenate([joint,np.clip(x+best_t[:,None]*rays,oracle.lb,oracle.ub)])
    remaining=cap-(oracle.used-start)
    if remaining<len(candidates):candidates=candidates[:remaining]
    if len(candidates)==0:return x,np.inf,oracle.used-start
    f=oracle(candidates);j=int(f.argmin());bx=candidates[j].copy();bf=float(f[j])
    remain=cap-(oracle.used-start)
    if polish and remain>=2*d+1:
        rx,rf,_=e.polish_bfgs(oracle,bx,remain)
        if rf<bf:bx,bf=rx,rf
    return bx,bf,oracle.used-start


class MeritOracle:
    """Temporary merit landscape; original-value incumbent is always tracked by base.

    Changing landscapes starts a fresh search. No old merit score is carried across
    penalty changes. Cached observations are not silently credited as new evaluations.
    """
    def __init__(self,base,limit,archive):
        self.base=base;self.limit=limit;self.archive=archive;self.lb=base.lb;self.ub=base.ub
    @property
    def used(self):return self.base.used
    @property
    def budget(self):return self.limit
    @property
    def remaining(self):return min(self.base.remaining,self.limit-self.base.used)
    @property
    def best(self):return self.base.best
    def __call__(self,x):
        if len(x)>self.remaining:raise e.BudgetStop()
        y=self.base(x)
        if not self.archive:return y
        xx=x.numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
        penalty=np.zeros(len(xx))
        for centre,radius,amp,metric in self.archive:
            z=(xx-centre)@metric
            dist=(z*z).mean(1)/(radius*radius)
            penalty+=amp*np.exp(-.5*dist)
        return y+torch.from_numpy(penalty) if isinstance(y,torch.Tensor) else y+penalty


def run_alternative(oracle,seed,method,outer_n=100,inner_n=100):
    if method.startswith('ray_'):
        basis=method[4:];saved=e.curvature_jump
        def fn(o,x,rng,cap=2400,geometry='dither'):
            return ray_search(o,x,rng,cap,geometry,basis)
        e.curvature_jump=fn
        try:return e.optimize(oracle,seed,'ml_qbj',outer_n=outer_n,inner_n=inner_n)
        finally:e.curvature_jump=saved
    if method not in ('deflated_iso','deflated_shape','restart_control'):raise ValueError(method)
    rng=np.random.default_rng(seed+88939);archive=[];epochs=[];logs=[];d=len(oracle.lb)
    phases=4
    for phase in range(phases):
        limit=oracle.budget*(phase+1)//phases
        merit=MeritOracle(oracle,limit,archive if method!='restart_control' else [])
        ep,lc=e.optimize(merit,seed+phase*104729,'ml_qbj',outer_n=outer_n,inner_n=inner_n)
        epochs.extend([dict(z,phase=phase) for z in ep]);logs.extend([dict(z,phase=phase) for z in lc])
        if phase==phases-1:break
        centre=oracle.best_x.numpy().copy()
        # Radius grows if repeated trajectories still return to the same incumbent.
        radius=.5*(1.5**phase)
        pts=np.clip(centre+radius*rng.standard_normal((max(32,2*d),d)),oracle.lb,oracle.ub)
        vals=oracle(pts)
        amp=4*max(float(np.quantile(vals,.75)-np.min(vals)),1e-12)
        metric=np.eye(d)
        if method=='deflated_shape':
            shape,_=probe_shape(oracle,centre,min(1000,oracle.remaining))
            if shape is not None:
                inv=np.linalg.pinv(shape);sv=np.linalg.svd(inv,compute_uv=False)
                metric=inv.T/max(float(np.median(sv)),1e-12)
        archive.append((centre,radius,amp,metric))
    assert oracle.used==oracle.budget
    return epochs,logs
