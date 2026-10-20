"""Development candidates. No optimizer receives problem labels or analytic derivatives."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import time
import numpy as np
import torch
from legacy_core import upstream, State, STATE_KEYS, Objective

class BudgetStop(Exception): pass

class Oracle:
    """Black-box interface: only querying evaluated candidates is exposed."""
    def __init__(self, objective, budget):
        self.__f=objective
        self.budget=int(budget); self.used=0
        self.best=np.inf; self.best_x=None; self.trace=[]
        self.lb=np.full(objective.dim,-5.); self.ub=np.full(objective.dim,5.)
    @property
    def remaining(self): return self.budget-self.used
    def __call__(self,x):
        is_torch=isinstance(x,torch.Tensor)
        t=x if is_torch else torch.from_numpy(np.asarray(x,dtype=float))
        if t.ndim!=2 or t.shape[1]!=len(self.lb): raise ValueError('Wrong shape')
        n=len(t)
        if n>self.remaining: raise BudgetStop()
        if not bool(torch.isfinite(t).all()) or not bool((t>=-5.-1e-12).all() and (t<=5.+1e-12).all()):
            raise ValueError('Bad candidate or boundary violation')
        y=self.__f.value(t)
        if not bool(torch.isfinite(y).all()): raise ValueError('Nonfinite objective')
        k=int(y.argmin()); v=float(y[k])
        if v<self.best: self.best=v; self.best_x=t[k].clone()
        self.used+=n
        self.trace.append((self.used,self.best))
        return y if is_torch else y.numpy()


def get_state(s,idx):
    return State(*(getattr(s,k)[idx].clone() for k in STATE_KEYS))
def set_state(s,idx,other):
    for k in STATE_KEYS: getattr(s,k)[idx]=getattr(other,k)

def select_diverse(s,k):
    f=s.gbest_fitness[:,0].numpy(); x=s.gbest[:,0].numpy()
    order=np.argsort(f,kind='stable')
    dist=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)
    upper=dist[np.triu_indices(len(x),1)]
    threshold=max(1e-6,0.2*float(np.median(upper)))
    selected=[]
    for j in order:
        if not selected or min(dist[j,selected])>threshold: selected.append(int(j))
        if len(selected)==k: return selected
    return (selected+[int(j) for j in order if j not in selected])[:k]


def polish_bfgs(oracle,x,cap=2500):
    """Bounded local refinement using batched CENTRAL finite differences, not autograd.

    L-BFGS-B is an existing control operator, not a claimed invention.
    All finite-difference and line-search queries are charged to the caller.
    """
    from scipy.optimize import minimize
    start=oracle.used; cap=min(cap,oracle.remaining); bestx=x.copy(); bestf=np.inf
    d=len(x); eps=2e-7
    def fg(z):
        nonlocal bestx,bestf
        if oracle.used-start+2*d+1>cap: raise BudgetStop()
        plus=np.minimum(z[None,:]+eps*np.eye(d),oracle.ub)
        minus=np.maximum(z[None,:]-eps*np.eye(d),oracle.lb)
        pts=np.concatenate((z[None,:],plus,minus),axis=0)
        vals=oracle(pts)
        j=int(np.argmin(vals))
        if vals[j]<bestf: bestf=float(vals[j]);bestx=pts[j].copy()
        g=(vals[1:d+1]-vals[d+1:])/(np.diag(plus)-np.diag(minus))
        return float(vals[0]),g
    try:
        minimize(fg,x,jac=True,method='L-BFGS-B',bounds=list(zip(oracle.lb,oracle.ub)),
                 options={'maxiter':max(5,cap//(2*d+1)), 'maxfun':100000,
                          'maxcor':min(30,d),'ftol':1e-15,'gtol':1e-10,'maxls':25})
    except BudgetStop: pass
    return bestx,bestf,oracle.used-start


def polish_quad(oracle,x,cap=2500,radius=0.5):
    """Value-only full quadratic interpolation, regularized curvature and model checks.

    One-sided inward probes stay feasible; all probes are actual charged evaluations.
    """
    start=oracle.used; d=len(x); xx=x.copy(); bestx=x.copy();bestf=np.inf
    pairs=np.array([(i,j) for i in range(d) for j in range(i+1,d)],dtype=int)
    n=1+2*d+len(pairs); rad=radius
    while oracle.used-start+n+8<=min(cap,oracle.budget-start):
        h=max(1e-5,min(1e-2,rad*.02))
        sign=np.where(xx>0,-1.,1.); step=h*sign
        eye=np.eye(d)*step
        pts=np.concatenate([xx[None,:],xx+eye,xx+2*eye,xx+eye[pairs[:,0]]+eye[pairs[:,1]]])
        vals=oracle(pts); j=int(np.argmin(vals))
        if vals[j]<bestf:bestf=float(vals[j]);bestx=pts[j].copy()
        f0=vals[0]; f1=vals[1:d+1]; f2=vals[d+1:2*d+1]
        diag=(f2-2*f1+f0)/step**2
        H=np.diag(diag)
        cross=(vals[2*d+1:]-f1[pairs[:,0]]-f1[pairs[:,1]]+f0)/(step[pairs[:,0]]*step[pairs[:,1]])
        H[pairs[:,0],pairs[:,1]]=cross;H[pairs[:,1],pairs[:,0]]=cross
        g=(f1-f0)/step-0.5*step*diag
        w,q=np.linalg.eigh(H); floor=max(1e-8,np.max(np.abs(w))*1e-7)
        direction=-q@((q.T@g)/np.maximum(w,floor))
        norm=np.linalg.norm(direction)
        if norm>rad:direction*=rad/norm
        alpha=2.**(-np.arange(8))
        cand=np.clip(xx+alpha[:,None]*direction,oracle.lb,oracle.ub)
        fy=oracle(cand); j=int(np.argmin(fy))
        if fy[j]<bestf:bestf=float(fy[j]);bestx=cand[j].copy()
        pred=-(g@ (cand[j]-xx)+0.5*(cand[j]-xx)@H@(cand[j]-xx))
        rho=(f0-fy[j])/max(abs(pred),np.finfo(float).tiny)
        if fy[j]<f0:
            xx=cand[j].copy()
            rad=min(5.,rad*2) if rho>0.75 else rad
        else:rad*=0.25
        if rad<1e-7:break
    return bestx,bestf,oracle.used-start


def inject(s,i,x,f):
    if f<float(s.gbest_fitness[i,0]):
        s.gbest[i,0]=torch.from_numpy(x);s.gbest_fitness[i,0]=f
    j=int(s.pbest_fitness[i,0].argmax())
    if f<float(s.pbest_fitness[i,0,j]):
        s.pbest[i,0,j]=torch.from_numpy(x);s.pbest_fitness[i,0,j]=f
        s.population[i,0,j]=torch.from_numpy(x);s.velocity[i,0,j].zero_()


@torch.no_grad()
def optimize(oracle,seed,method,outer_n=16,inner_n=40,horizon=100):
    torch.manual_seed(seed);rng=np.random.default_rng(seed+991)
    d=len(oracle.lb); B=outer_n;N=inner_n
    lo=torch.zeros(10,dtype=torch.float64)
    hi=torch.tensor([1,1,1,1,1,8.99999,8.99999,8.99999,8.99999,1],dtype=torch.float64)
    outer=upstream.TorchOuterPSO(lo,hi,B)
    inner=upstream.TorchAutoPSO(torch.from_numpy(oracle.lb),torch.from_numpy(oracle.ub),N)
    epochs=[]; local_calls=[]; block=0; s=None
    original=method.startswith('original')
    original_h=800 if method in ('original','original_b','original_qbj') else horizon
    restart=('restart' in method or method in ('ml_b','ml_q','ml_qb','ml_qbj','ml_bj','ml_qbj_center','ml_qbj_random','ml_qbj_single','original_qbj'))
    single=(method in ('single_b','ml_qbj_single'))
    use_b=method in ('ml_b','single_b','original_b','ml_qb','ml_qbj','ml_bj','ml_qbj_center','ml_qbj_random','ml_qbj_single','original_qbj')
    use_q=method in ('ml_q','ml_qb','ml_qbj','ml_qbj_center','ml_qbj_random','ml_qbj_single','original_qbj')
    use_j=method in ('ml_qbj','ml_bj','ml_qbj_center','ml_qbj_random','ml_qbj_single','original_qbj')
    k=1 if single else 4
    while oracle.remaining>=B*N:
        if original or s is None: s=inner.setup(B)
        params=upstream.decode_autopso_parameters_torch(outer.population)
        steps=min(original_h if original else horizon,oracle.remaining//(B*N))
        for _ in range(steps):
            fit=oracle(s.population.reshape(-1,d)).reshape(B,1,N)
            s=inner.step(s,fit,params)
        scores=s.gbest_fitness[:,0].clone()
        ids=select_diverse(s,k)
        # Evaluation budget, not outer iterations, is the experimental constraint.
        if (use_b or use_q) and oracle.remaining>=2*d+1:
            for i in ids:
                x=s.gbest[i,0].numpy().copy()
                # Fixed local allowance; all pilot probes count against total budget.
                cap=min(5000 if use_j else 3000,oracle.remaining)
                old=float(s.gbest_fitness[i,0]); spent=0
                if use_q:
                    xq,fq,nq=polish_quad(oracle,x,cap if not use_b else min(1500,cap))
                    spent+=nq;inject(s,i,xq,fq)
                    if fq<old:x=xq
                if use_b and oracle.remaining>=2*d+1:
                    xb,fb,nb=polish_bfgs(oracle,x,min(1200 if use_j else cap-spent,cap-spent,oracle.remaining))
                    spent+=nb;inject(s,i,xb,fb)
                if use_j:
                    xj,fj,nj=curvature_jump(oracle,s.gbest[i,0].numpy().copy(),rng,min(cap-spent,oracle.remaining),geometry='center' if method.endswith('_center') else ('random' if method.endswith('_random') else 'dither'))
                    spent+=nj;inject(s,i,xj,fj)
                local_calls.append({'block':block,'lineage':i,'before':old,
                     'after':float(s.gbest_fitness[i,0]),'fes':spent})
        # Outer credit uses actual post-search fitness; refreshed when state changes.
        scores=s.gbest_fitness[:,0].clone()
        if not original:
            outer.pbest_fitness.fill_(torch.inf);outer.gbest_fitness.fill_(torch.inf)
        old_theta=outer.population.clone();outer.step(scores)
        ids=select_diverse(s,k)
        epochs.append({'block':block,'fes':oracle.used,'best':oracle.best,'selected':ids})
        if not original:
            parent_idx=torch.tensor([ids[j%k] for j in range(B)])
            s=get_state(s,parent_idx)
            # One unchanged policy per retained parent. Other policies follow outer PSO.
            for j,i in enumerate(ids):
                outer.population[j]=old_theta[i];outer.velocity[j].zero_()
            if restart:
                fresh=inner.setup(B//4);sl=slice(B-B//4,B);set_state(s,sl,fresh)
                outer.population[sl]=torch.rand((B//4,10))*hi
                outer.velocity[sl].zero_()
        block+=1
    if oracle.remaining:
        # Exact-budget tail: candidates are generated by the current swarm, not padded.
        if s is None:s=inner.setup(B)
        oracle(s.population.reshape(-1,d)[:oracle.remaining])
    assert oracle.used==oracle.budget
    return epochs,local_calls


def run_task(task):
    name,dim,instance,seed,method,budget,*rest=task
    options=rest[0] if rest else {}
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    objective=Objective(name,dim,instance,budget)
    oracle=Oracle(objective,budget)
    t=time.perf_counter();epochs,local=optimize(oracle,seed,method,**options);elapsed=time.perf_counter()-t
    actual=float(objective.value(oracle.best_x[None,:])[0])
    assert np.isclose(actual,oracle.best,rtol=1e-12,atol=1e-12)
    return {'function':name,'dim':dim,'instance':instance,'seed':seed,'method':method,
            'budget':budget,'options':options,'fes':oracle.used,'error':oracle.best,'best_x':oracle.best_x.tolist(),
            'seconds':elapsed,'trace':oracle.trace,'epochs':epochs,'local':local,
            'verification_error':actual}


def curvature_jump(oracle,x,rng,cap=2400,geometry='dither'):
    """Dithered-curvature basin probes: estimate geometry OFF the incumbent.

    Unlike a Hessian exactly at a near-isotropic minimum, the dithered model can
    expose useful directional structure. This is a hypothesis, not a novelty claim.
    Every model probe, jump candidate, and finite-difference refinement is charged.
    """
    start=oracle.used;d=len(x);cap=min(cap,oracle.remaining)
    pairs=np.array([(i,j) for i in range(d) for j in range(i+1,d)],int)
    nq=1+2*d+len(pairs);span=oracle.ub-oracle.lb
    scales=np.array([1/80,1/40,1/20,1/10,1/5])*float(np.mean(span))
    nj=2*d*len(scales)
    if nq+nj>cap:return x,np.inf,0
    if geometry=='random':
        q,_=np.linalg.qr(rng.standard_normal((d,d)))
        f=np.array([np.inf]);pts=x[None,:]
    else:
        centre=(np.clip(x+.02*span*rng.standard_normal(d),oracle.lb,oracle.ub) if geometry=='dither' else x.copy())
        step=np.where(centre>0,-1.,1.)*.0002*span
        eye=np.eye(d)*step
        pts=np.concatenate([centre[None,:],centre+eye,centre+2*eye,
                            centre+eye[pairs[:,0]]+eye[pairs[:,1]]])
        f=oracle(pts);f0=f[0];f1=f[1:d+1];f2=f[d+1:2*d+1]
        H=np.diag((f2-2*f1+f0)/step**2)
        z=(f[2*d+1:]-f1[pairs[:,0]]-f1[pairs[:,1]]+f0)/(step[pairs[:,0]]*step[pairs[:,1]])
        H[pairs[:,0],pairs[:,1]]=z;H[pairs[:,1],pairs[:,0]]=z
        _,q=np.linalg.eigh(H)
    rays=np.concatenate([q.T,-q.T])
    cand=np.clip(x+ (scales[:,None,None]*rays[None,:,:]).reshape(-1,d),oracle.lb,oracle.ub)
    val=oracle(cand)
    jm=int(f.argmin());j=int(val.argmin())
    bestx,bestf=(pts[jm].copy(),float(f[jm])) if f[jm]<val[j] else (cand[j].copy(),float(val[j]))
    # The best actual proposal is refined, even when it must first cross a ridge.
    remaining=cap-(oracle.used-start)
    if remaining>=2*d+1:
        bx,bf,used=polish_bfgs(oracle,cand[j].copy(),remaining)
        if bf<bestf:bestx,bestf=bx,bf
    return bestx,bestf,oracle.used-start
