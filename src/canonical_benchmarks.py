"""Canonical functions from AutoPSO v3 Table S.I; no random shift/rotation.
Physical bounds are mapped bijectively to the frozen optimizer's [-5,5].
The algorithm sees only function values and bounds, never optimum coordinates.
"""
import numpy as np
import torch
BOUNDS={'ackley':(-32.,32.),'griewank':(-600.,600.),'rastrigin':(-5.12,5.12),
        'rosenbrock':(-5.,10.),'schwefel':(-500.,500.),'sphere':(-5.12,5.12)}

class Canonical:
    def __init__(self,name,dim):
        self.name=name;self.dim=dim; self.low,self.high=BOUNDS[name]
        self.indices=torch.arange(1,dim+1,dtype=torch.float64)
    def physical(self,z):return self.low+(z+5.)*(self.high-self.low)/10.
    def value(self,z):
        x=self.physical(z)
        if self.name=='ackley':
            return -20.*torch.exp(-.2*torch.sqrt((x*x).mean(1)))-torch.exp(torch.cos(2.*torch.pi*x).mean(1))+20.+np.e
        if self.name=='griewank':return (x*x).sum(1)/4000.-torch.cos(x/torch.sqrt(self.indices)).prod(1)+1.
        if self.name=='rastrigin':return (x*x+10.*(1.-torch.cos(2.*torch.pi*x))).sum(1)
        if self.name=='rosenbrock':return (100.*(x[:,1:]-x[:,:-1]**2)**2+(1.-x[:,:-1])**2).sum(1)
        if self.name=='schwefel':return 418.9828872724338*self.dim-(x*torch.sin(torch.sqrt(torch.abs(x)))).sum(1)
        if self.name=='sphere':return (x*x).sum(1)
        raise ValueError(self.name)

def independent_numpy(name,x):
    x=np.asarray(x,dtype=np.float64)
    if name=='ackley':return -20*np.exp(-.2*np.sqrt(np.mean(x*x)))-np.exp(np.mean(np.cos(2*np.pi*x)))+20+np.e
    if name=='griewank':return np.sum(x*x)/4000.-np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1))))+1.
    if name=='rastrigin':return np.sum(x*x+10*(1.-np.cos(2*np.pi*x)))
    if name=='rosenbrock':return np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)
    if name=='schwefel':return 418.9828872724338*len(x)-np.sum(x*np.sin(np.sqrt(np.abs(x))))
    if name=='sphere':return np.sum(x*x)
    raise ValueError(name)
