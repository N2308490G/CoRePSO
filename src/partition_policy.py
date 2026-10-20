"""Inner-subgroup policies; exemplars always use the entire inner population."""
import torch
from experiment_v3 import upstream


class PartitionInner(upstream.TorchAutoPSO):
    def step(self, state, fitness, params):
        improved = fitness < state.pbest_fitness
        pbest = torch.where(improved.unsqueeze(-1), state.population, state.pbest)
        pfit = torch.minimum(state.pbest_fitness, fitness)
        value, index = fitness.min(dim=2)
        gbest = torch.where((value < state.gbest_fitness).unsqueeze(-1),
                            self._gather_population(state.population, index), state.gbest)
        gfit = torch.minimum(value, state.gbest_fitness)
        population = state.population
        mean = population.mean(dim=2, keepdim=True).expand_as(population)
        random_x = self._random_population_rows(population)
        random_pbest = self._random_population_rows(pbest)
        new_random = torch.rand_like(population)*(self.ub-self.lb)+self.lb
        elite = self._random_elite_pbest(pbest, pfit)
        clpso = self._clpso_pbest(pbest, pfit)
        choice = torch.stack([population,pbest,gbest.unsqueeze(2).expand_as(population),
                              mean,random_x,random_pbest,new_random,elite,clpso])
        rp, rg = torch.rand_like(population), torch.rand_like(population)
        batch = len(population)
        cuts = torch.floor(params['cuts']*self.pop_size).long()
        particle = torch.arange(self.pop_size,device=population.device)[None,:,None]
        group = (particle >= cuts[:,None,:]).sum(-1)
        velocity = torch.zeros_like(population)
        for g in range(params['coefficients'].shape[1]):
            p1 = self._select_choice(choice,params['types'][:,g,0])
            p2 = self._select_choice(choice,params['types'][:,g,1])
            v = (params['weight'].view(batch,1,1,1)*state.velocity
                 + params['coefficients'][:,g,0].view(batch,1,1,1)*rp*(p1-population)
                 + params['coefficients'][:,g,1].view(batch,1,1,1)*rg*(p2-population))
            velocity = torch.where((group==g)[:,None,:,None],v,velocity)
        return upstream.InnerState(torch.clamp(population+velocity,self.lb,self.ub),
                                   velocity,pbest,pfit,gbest,gfit)


class PartitionPolicy:
    """K-1 sorted cuts; one shared inertia and two coefficients/exemplars per group.

    K=2 searched retains the original ten-coordinate decoder and implementation.
    The 0.5/0.5 control removes only its ratio coordinate from the search space.
    """
    def __init__(self, groups=2, fixed_half=False):
        if groups not in (1,2,3,4) or (fixed_half and groups!=2):
            raise ValueError('Unsupported inner partition')
        self.groups,self.fixed_half = groups,fixed_half
        self.high = torch.tensor([1.]*(1+2*groups)+[8.99999]*(2*groups)+
                                 ([1.]*(groups-1) if not fixed_half else []))
        self.dimension = len(self.high)
        self.inner_class = upstream.TorchAutoPSO if groups==2 else PartitionInner

    def decode(self,pop):
        if pop.shape[1]!=self.dimension:
            raise ValueError('Policy dimension mismatch')
        if self.groups==2:
            if self.fixed_half:
                pop=torch.cat([pop,torch.full_like(pop[:,:1],.5)],dim=1)
            return upstream.decode_autopso_parameters_torch(pop)
        k=self.groups
        return dict(weight=pop[:,0],coefficients=pop[:,1:1+2*k].reshape(-1,k,2),
                    types=pop[:,1+2*k:1+4*k].floor().long().clamp(0,8).reshape(-1,k,2),
                    cuts=pop[:,1+4*k:].sort(dim=1).values)
