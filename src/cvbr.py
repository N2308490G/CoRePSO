"""Coupling-validated block recombination: experimental value-only local operator.

Coordinate searches, block recombination, and trust checks have prior art.
This module is a testable hypothesis, not a claim of first-in-literature novelty.
No function identifiers, optimum locations, transforms, or derivatives are read.
"""
import numpy as np
import torch
import experiment_v3 as e


def block_recombine(oracle, x, rng, cap=5000, validate=True, polish=True):
    start=oracle.used; cap=min(int(cap),oracle.remaining); d=len(x)
    if cap<100: return x,np.inf,0,{}
    r=min(d,64,max(1,(cap-30)//62))
    ids=rng.choice(d,size=r,replace=False)
    f0=float(oracle(x[None])[0]); bx=x.copy();bf=f0
    # Randomized stratified probes, not a grid chosen to include the origin.
    g=31
    ts=(np.arange(g)[None,:]+rng.random((r,g)))/g
    ts=oracle.lb[ids,None]+ts*(oracle.ub[ids,None]-oracle.lb[ids,None])
    ts.sort(axis=1)
    pts=np.repeat(x[None,:],r*g,axis=0)
    pts[np.arange(r*g),np.repeat(ids,g)]=ts.ravel()
    vals=oracle(pts).reshape(r,g)
    k=vals.argmin(1);ii=np.arange(r)
    t=ts[ii,k].copy();fv=vals[ii,k].copy()
    a=ts[ii,np.maximum(k-1,0)].copy();b=ts[ii,np.minimum(k+1,g-1)].copy()
    for _ in range(12):
        if oracle.used-start+2*r+20>cap:break
        c=(2*a+b)/3;dd=(a+2*b)/3
        pp=np.repeat(x[None,:],2*r,axis=0)
        pp[ii,ids]=c;pp[r+ii,ids]=dd
        v=oracle(pp);fc,fd=v[:r],v[r:]
        u=fc<fv;t[u]=c[u];fv[u]=fc[u]
        u=fd<fv;t[u]=dd[u];fv[u]=fd[u]
        good=fc<fd;b=np.where(good,dd,b);a=np.where(good,a,c)
    # Do not recombine per-coordinate moves that are individually harmful.
    good=fv<f0;ids=ids[good];t=t[good];fv=fv[good]
    if not len(ids):return bx,bf,oracle.used-start,{'r':r,'improving_axes':0,'ratio':None}
    trial=x.copy();trial[ids]=t
    alphas=np.array([1.,.5,.25,.125])
    p=x+alphas[:,None]*(trial-x)
    vv=oracle(p);j=int(vv.argmin())
    if vv[j]<bf:bx=p[j].copy();bf=float(vv[j])
    pred=float(np.sum(f0-fv));ratio=(f0-float(vv[0]))/max(pred,1e-300)
    # Non-additive joint proposals are split and verified at the CURRENT incumbent.
    # The branch accepts only actual queried improvements; no free model fitness.
    splits=0
    if validate and ratio<.25:
        queue=[np.arange(len(ids))]
        while queue and oracle.used-start+1<=cap:
            gidx=queue.pop(0);p=bx.copy();p[ids[gidx]]=t[gidx]
            f=float(oracle(p[None])[0]);splits+=1
            if f<bf:bx,bf=p,f
            elif len(gidx)>1:
                mid=len(gidx)//2;queue.extend([gidx[:mid],gidx[mid:]])
    remain=cap-(oracle.used-start)
    if polish and remain>=2*d+1:
        xx,ff,_=e.polish_bfgs(oracle,bx,remain)
        if ff<bf:bx,bf=xx,ff
    return bx,bf,oracle.used-start,{'r':r,'improving_axes':len(ids),'ratio':ratio,'splits':splits}


@torch.no_grad()
def optimize_cvbr(oracle,seed,method='cvbr',outer_n=100,inner_n=100,horizon=100):
    torch.manual_seed(seed);rng=np.random.default_rng(seed+991)
    B,N=outer_n,inner_n;d=len(oracle.lb)
    hi=torch.tensor([1.,1.,1.,1.,1.,8.99999,8.99999,8.99999,8.99999,1.])
    outer=e.upstream.TorchOuterPSO(torch.zeros(10),hi,B)
    inner=e.upstream.TorchAutoPSO(torch.from_numpy(oracle.lb),torch.from_numpy(oracle.ub),N)
    s=None;epochs=[];logs=[];block=0;survivors=min(B,4)
    while oracle.remaining>=B*N:
        if s is None:s=inner.setup(B)
        params=e.upstream.decode_autopso_parameters_torch(outer.population)
        # Retains the previous CC controller's swarm budget block.
        steps=min(horizon,oracle.remaining//(B*N));u=oracle.used
        for _ in range(steps):
            fit=oracle(s.population.reshape(-1,d)).reshape(B,1,N);s=inner.step(s,fit,params)
        cost=oracle.used-u;quota=min(cost//4,oracle.remaining);u=oracle.used
        count=min(B,max(1,int(np.ceil(quota/5000))))
        ids=e.select_diverse(s,count)
        for i in ids:
            cap=min(5000,quota-(oracle.used-u),oracle.remaining)
            if cap<100:break
            x=s.gbest[i,0].numpy().copy();before=float(s.gbest_fitness[i,0])
            # Race an established curvature operator against the block operator,
            # with a deterministic split independent of problem identity.
            if method=='cvbr_hybrid' and (block+i)%2==0:
                from alternatives import ray_search
                xx,ff,used=ray_search(oracle,x,rng,cap)
                detail={'operator':'curvature_ray'}
                if used==0:
                    xx,ff,used,detail=block_recombine(oracle,x,rng,cap)
                    detail['operator']='block_fallback'
            else:
                xx,ff,used,detail=block_recombine(oracle,x,rng,cap,validate=method!='cvbr_no_split')
                detail['operator']='block'
            e.inject(s,i,xx,ff)
            logs.append(dict(block=block,lineage=i,before=before,after=float(s.gbest_fitness[i,0]),fes=used,**detail))
        scores=s.gbest_fitness[:,0].clone();outer.pbest_fitness.fill_(torch.inf);outer.gbest_fitness.fill_(torch.inf)
        theta=outer.population.clone();outer.step(scores);ids=e.select_diverse(s,survivors)
        epochs.append(dict(block=block,fes=oracle.used,best=oracle.best,selected=ids,search_fes=cost,local_fes=oracle.used-u))
        s=e.get_state(s,torch.tensor([ids[j%survivors] for j in range(B)]))
        for j,i in enumerate(ids):outer.population[j]=theta[i];outer.velocity[j].zero_()
        fresh=B//4;new=inner.setup(fresh);sl=slice(B-fresh,B);e.set_state(s,sl,new)
        outer.population[sl]=torch.rand((fresh,10))*hi;outer.velocity[sl].zero_();block+=1
    if oracle.remaining:
        if s is None:s=inner.setup(B)
        oracle(s.population.reshape(-1,d)[:oracle.remaining])
    assert oracle.used==oracle.budget
    return epochs,logs
