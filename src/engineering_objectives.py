"""FM identification and radar minimax benchmarks, in normalized [-5, 5]^D.

Definitions: Das & Suganthan, CEC 2011 technical report, pp. 3-4, 12-13,
equations (1)-(3), (31)-(34). These are mathematical application benchmarks,
not measured signals or deployed radar systems. Radar follows the report's
positive AND negative branches; the distributed Matlab case 7 omits the
negative branches, so its scores are not interchangeable with this protocol.
"""
import math
import numpy as np
import torch

PROTOCOL = 'engineering_cec2011_equations_float64_v1'
SOURCE = 'https://github.com/P-N-Suganthan/CEC-2011--Real_World_Problems'
CASES = [dict(name='FM-6D', kind='fm', dim=6),
         dict(name='Radar-19D', kind='radar', dim=19),
         dict(name='Radar-20D', kind='radar', dim=20)]
FM_TARGET = np.array([1., 5., -1.5, 4.8, 2., 4.9])


def physical(z, kind):
    return -6.4 + (z + 5.) * 1.275 if kind == 'fm' else (z + 5.) * (math.pi / 5.)


class EngineeringObjective:
    def __init__(self, case, device='cuda'):
        if case not in CASES:
            raise ValueError('Unknown application instance')
        self.case, self.dim, self.device = dict(case), case['dim'], device
        if case['kind'] == 'fm':
            self.time = torch.arange(101, dtype=torch.float64, device=device) * (2 * math.pi / 100)
            t = self.time
            self.target = torch.sin(5*t - 1.5*torch.sin(4.8*t + 2*torch.sin(4.9*t)))
        else:
            # Precompute interval sums and group membership from report (32)-(33).
            # Each row of A selects x_{|2i-j-1|+1:j} or x_{|2i-j|+1:j}.
            a, groups = [], []
            offsets = np.zeros(2*self.dim-1)
            for i in range(1, self.dim+1):
                for j in range(i, self.dim+1):
                    v = np.zeros(self.dim); v[abs(2*i-j-1):j] = 1
                    a.append(v); groups.append(2*i-2)
                if i < self.dim:
                    offsets[2*i-1] = .5
                    for j in range(i+1, self.dim+1):
                        v = np.zeros(self.dim); v[abs(2*i-j):j] = 1
                        a.append(v); groups.append(2*i-1)
            reduce = np.zeros((len(a), 2*self.dim-1))
            reduce[np.arange(len(a)), groups] = 1
            self.intervals = torch.tensor(np.array(a).T.copy(), dtype=torch.float64, device=device)
            self.reduce = torch.tensor(reduce, dtype=torch.float64, device=device)
            self.offsets = torch.tensor(offsets, dtype=torch.float64, device=device)

    def value(self, z):
        x = physical(z, self.case['kind'])
        if self.case['kind'] == 'fm':
            a1, w1, a2, w2, a3, w3 = x.unbind(-1)
            t = self.time
            y = a1[:, None]*torch.sin(w1[:, None]*t + a2[:, None]*torch.sin(
                w2[:, None]*t + a3[:, None]*torch.sin(w3[:, None]*t)))
            return ((y-self.target)**2).sum(-1)
        phi = torch.cos(x @ self.intervals) @ self.reduce + self.offsets
        return phi.abs().amax(-1)


def independent_verify(case, z):
    """Scalar loop implementation; shares no tensor reduction/index matrices."""
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (case['dim'],) or not np.isfinite(z).all() or (np.abs(z)>5.000000000001).any():
        raise ValueError('Invalid final candidate')
    x = physical(z, case['kind'])
    if case['kind'] == 'fm':
        a1, w1, a2, w2, a3, w3 = x
        terms = []
        for t in range(101):
            q = t*2*math.pi/100
            y = a1*math.sin(w1*q+a2*math.sin(w2*q+a3*math.sin(w3*q)))
            target = math.sin(5*q-1.5*math.sin(4.8*q+2*math.sin(4.9*q)))
            terms.append((y-target)**2)
        return math.fsum(terms)
    phi = []
    for i in range(1, len(x)+1):
        phi.append(math.fsum(math.cos(math.fsum(x[k-1] for k in range(abs(2*i-j-1)+1, j+1)))
                            for j in range(i, len(x)+1)))
        if i < len(x):
            phi.append(.5+math.fsum(math.cos(math.fsum(x[k-1] for k in range(abs(2*i-j)+1, j+1)))
                                   for j in range(i+1, len(x)+1)))
    return max(abs(v) for v in phi)


def check_objectives(device='cuda'):
    rng = np.random.default_rng(80101)
    for case in CASES:
        obj = EngineeringObjective(case, device)
        points = np.vstack([rng.uniform(-5, 5, (13, case['dim'])),
                            np.full(case['dim'], -5.), np.full(case['dim'], 5.)])
        actual = obj.value(torch.tensor(points, dtype=torch.float64, device=device)).cpu().numpy()
        expected = np.array([independent_verify(case, z) for z in points])
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)
        if case['kind']=='radar':
            np.testing.assert_allclose(actual[-2:], case['dim'], rtol=0, atol=1e-12)
        else:
            z = (FM_TARGET+6.4)/1.275-5
            assert independent_verify(case,z)<1e-23
            assert float(obj.value(torch.tensor(z[None],dtype=torch.float64,device=device))[0])<1e-23
    return dict(status='passed', device=device, checks='45 asymmetric/boundary cases, scalar reference, FM target zero, radar constant-phase value')
