"""Float64 numerical objectives for the corrected CUDA experiment protocol.

CEC F3 uses rotation and F7 uses its own last segment, as in the pinned
AutoPSO Torch evaluator. F4 intentionally retains that evaluator's continuous
Rastrigin expression. This is an explicit author-code protocol, not a claim
that every formula in the inconsistent CEC technical report is reproduced.
CEC rotations and sensitive sums use a fixed order on CPU and CUDA.
"""
from pathlib import Path
import numpy as np
import torch
from canonical_benchmarks import BOUNDS

ROOT = Path(__file__).resolve().parents[1]
BIAS = (300., 400., 600., 800., 900., 1800., 2000., 2200., 2300., 2400., 2600., 2700.)
PROTOCOL = 'author_cec_correct_segments_ordered_float64_v1'


def ordered_sum(x):
    result = torch.zeros_like(x[:, 0])
    for j in range(x.shape[1]):
        result = result + x[:, j]
    return result


def basic(x, kind):
    d = x.shape[1]
    if kind == 'zakharov':
        s = ordered_sum(x * (.5 * torch.arange(1, d + 1, device=x.device, dtype=x.dtype)))
        return ordered_sum(x*x) + s*s + s**4
    if kind == 'rosenbrock':
        y = x + 1
        return ordered_sum(100*(y[:, :-1]**2-y[:, 1:])**2 + (y[:, :-1]-1)**2)
    if kind == 'schaffer':
        r = torch.sqrt(x[:, :-1]**2 + x[:, 1:]**2)
        v = torch.sqrt(r) * (1 + torch.sin(50*r**.2)**2)
        return (ordered_sum(v)/(d-1))**2
    if kind == 'rastrigin':
        return ordered_sum(x*x - 10*torch.cos(2*torch.pi*x) + 10)
    if kind == 'levy':
        w = 1+x/4
        return (torch.sin(torch.pi*w[:, 0])**2 + (w[:, -1]-1)**2 *
                (1+torch.sin(2*torch.pi*w[:, -1])**2) +
                ordered_sum((w[:, :-1]-1)**2*(1+10*torch.sin(torch.pi*w[:, :-1]+1)**2)))
    if kind == 'bent_cigar':
        return x[:, 0]**2 + 1e6*ordered_sum(x[:, 1:]*x[:, 1:])
    if kind in ('hgbat', 'happycat'):
        y = x-1
        r2, s = ordered_sum(y*y), ordered_sum(y)
        term = torch.sqrt(torch.abs(r2*r2-s*s)) if kind == 'hgbat' else torch.abs(r2-d)**.25
        return term + (.5*r2+s)/d + .5
    if kind == 'katsuura':
        powers = 2.**torch.arange(1, 33, device=x.device, dtype=x.dtype)
        v = x[:, :, None]*powers
        sums = ((v-torch.floor(v+.5)).abs()/powers).sum(-1)
        terms = (1+sums*torch.arange(1, d+1, device=x.device, dtype=x.dtype))**(10/d**1.2)
        return (terms.prod(1)-1)*10/d/d
    if kind == 'ackley':
        return (np.e-20*torch.exp(-.2*torch.sqrt(ordered_sum(x*x)/d)) -
                torch.exp(ordered_sum(torch.cos(2*torch.pi*x))/d)+20)
    if kind == 'schwefel':
        z = x+420.9687462275036
        a = 500-torch.fmod(z.abs(), 500)
        v = -z*torch.sin(torch.sqrt(z.abs()))
        v = torch.where(z > 500, -a*torch.sin(torch.sqrt(a)) + ((z-500)/100)**2/d, v)
        v = torch.where(z < -500, a*torch.sin(torch.sqrt(a)) + ((z+500)/100)**2/d, v)
        return ordered_sum(v)+418.9828872724338*d
    if kind == 'grie_rosen':
        a = x+1
        t = 100*(a*a-torch.roll(a, -1, 1))**2+(a-1)**2
        return ordered_sum(t*t/4000-torch.cos(t)+1)
    if kind == 'griewank':
        ids = torch.sqrt(torch.arange(1, d+1, device=x.device, dtype=x.dtype))
        return 1+ordered_sum(x*x)/4000-torch.cos(x/ids).prod(1)
    if kind == 'escaffer6':
        a = x*x+torch.roll(x, -1, 1)**2
        return ordered_sum(.5+(torch.sin(torch.sqrt(a))**2-.5)/(1+.001*a)**2)
    if kind == 'ellips':
        ids = torch.arange(d, device=x.device, dtype=x.dtype)
        return ordered_sum(10.**(6*ids/(d-1))*x*x)
    if kind == 'discus':
        return 1e6*x[:, 0]**2+ordered_sum(x[:, 1:]*x[:, 1:])
    raise ValueError(kind)


def cases():
    out = [dict(group='cec2022', name=f'F{f}', dim=d, fid=f, instance=0)
           for d in (10, 20) for f in range(1, 13)]
    out += [dict(group='canonical', name=n, dim=d, instance=0)
            for d in (50, 100, 200) for n in BOUNDS]
    out += [dict(group='shiftrot', name=n, dim=d, instance=0)
            for d in (50, 100, 200) for n in ('ackley', 'rastrigin')]
    out += [dict(group='highdim', name=n, dim=d, instance=0)
            for d in (500, 1000, 2000) for n in ('ackley', 'rastrigin')]
    return out


class NumericalObjective:
    def __init__(self, case, device='cuda'):
        self.case, self.dim, self.device = case, case['dim'], torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.protocol = PROTOCOL
        def tensor(x):
            return torch.as_tensor(np.asarray(x).copy(), dtype=torch.float64, device=self.device)
        if case['group'] == 'cec2022':
            f, d = case['fid'], self.dim
            data = ROOT/'Python/input_data'
            self.M = tensor(np.loadtxt(data/f'M_{f}_D{d}.txt').reshape(-1, d, d))
            sh = np.loadtxt(data/f'shift_data_{f}.txt')
            self.shift = tensor(sh[None, :d] if f < 9 else sh[:9, :d])
            self.S = None
            if 6 <= f <= 8:
                self.S = torch.tensor(np.loadtxt(data/f'shuffle_data_{f}_D{d}.txt', dtype=int)-1,
                                      device=self.device, dtype=torch.long)
        else:
            self.low, self.high = BOUNDS[case['name']]
            if case['group'] == 'shiftrot':
                with np.load(ROOT/'new_shiftrot_instances'/f"{case['name']}_{self.dim}D_instance{case['instance']}.npz") as data:
                    self.shift, self.rotation = tensor(data['shift']), tensor(data['rotation'])

    def _transform(self, x, i=0, scale=1., rotate=True):
        y = (x-self.shift[i])*scale
        if not rotate:
            return y
        # Same scalar multiplication/addition order regardless of batch length,
        # avoiding BLAS GEMV/GEMM differences at HappyCat's fourth-root cusp.
        result = torch.zeros_like(y)
        for j in range(self.dim):
            result = result + y[:, j:j+1]*self.M[i, :, j]
        return result

    def value(self, z):
        if z.dtype != torch.float64 or z.device != self.device:
            raise ValueError('Objective requires explicit float64 on its configured device.')
        if self.case['group'] != 'cec2022':
            x = self.low+(z+5)*(self.high-self.low)/10
            if self.case['group'] == 'shiftrot':
                x = (x-self.shift)@self.rotation
            n = self.case['name']
            if n == 'ackley':
                return -20*torch.exp(-.2*(x*x).mean(1).sqrt())-torch.exp(torch.cos(2*torch.pi*x).mean(1))+20+np.e
            if n == 'rastrigin':
                return (x*x+10*(1-torch.cos(2*torch.pi*x))).sum(1)
            if n == 'sphere':
                return (x*x).sum(1)
            if n == 'rosenbrock':
                return (100*(x[:, 1:]-x[:, :-1]**2)**2+(1-x[:, :-1])**2).sum(1)
            if n == 'schwefel':
                return 418.9828872724338*self.dim-(x*torch.sin(x.abs().sqrt())).sum(1)
            if n == 'griewank':
                ids = torch.sqrt(torch.arange(1, self.dim+1, device=x.device, dtype=x.dtype))
                return (x*x).sum(1)/4000-torch.cos(x/ids).prod(1)+1
            raise ValueError(n)
        x, f = z*20, self.case['fid']
        if f <= 5:
            kind = ('zakharov', 'rosenbrock', 'schaffer', 'rastrigin', 'levy')[f-1]
            return basic(self._transform(x, scale=(1., .02048, 1., .0512, 1.)[f-1]), kind)
        if f <= 8:
            gp = {6:(.4,.4,.2), 7:(.1,.2,.2,.2,.1,.2), 8:(.3,.2,.2,.1,.2)}[f]
            kinds = {6:('bent_cigar','hgbat','rastrigin'),
                     7:('hgbat','katsuura','ackley','rastrigin','schwefel','schaffer'),
                     8:('katsuura','happycat','grie_rosen','schwefel','ackley')}[f]
            scales = {6:(1.,.05,.0512), 7:(.05,.05,1.,.0512,10.,1.), 8:(.05,.05,.05,10.,1.)}[f]
            sizes = [int(np.ceil(p*self.dim)) for p in gp]
            sizes[-1] = self.dim-sum(sizes[:-1])
            y = self._transform(x)[:, self.S]
            result, offset = torch.zeros_like(x[:, 0]), 0
            for n, size, scale in zip(kinds, sizes, scales):
                result = result+basic(y[:, offset:offset+size]*scale, n)
                offset += size
            return result
        settings = {
            9:(('rosenbrock','ellips','bent_cigar','discus','ellips'),(.02048,1.,1.,1.,1.),(1.,1e-6,1e-26,1e-6,1e-6),(1,1,1,1,0),(10.,20.,30.,40.,50.),(0.,200.,300.,100.,400.)),
            10:(('schwefel','rastrigin','hgbat'),(10.,.0512,.05),(1.,1.,1.),(0,1,1),(20.,10.,10.),(0.,200.,100.)),
            11:(('escaffer6','schwefel','griewank','rosenbrock','rastrigin'),(1.,10.,6.,.02048,.0512),(.0005,1.,10.,1.,10.),(1,1,1,1,1),(20.,20.,30.,30.,20.),(0.,200.,300.,400.,200.)),
            12:(('hgbat','rastrigin','schwefel','bent_cigar','ellips','escaffer6'),(.05,.0512,10.,1.,1.,1.),(10.,10.,2.5,1e-26,1e-6,.0005),(1,1,1,1,1,1),(10.,20.,30.,40.,50.,60.),(0.,300.,500.,100.,400.,200.))}
        kinds, scales, multipliers, rotates, deltas, biases = settings[f]
        weights, vals = [], []
        for i, (n, scale, mul, rot, delta, bias) in enumerate(zip(kinds, scales, multipliers, rotates, deltas, biases)):
            vals.append(basic(self._transform(x, i, scale, rot), n)*mul+bias)
            dist = ordered_sum((x-self.shift[i])**2)
            safe = torch.where(dist == 0, torch.ones_like(dist), dist)
            weights.append(torch.where(dist == 0, 1e99, torch.exp(-dist/(2*self.dim*delta*delta))/torch.sqrt(safe)))
        w, v = torch.stack(weights, 1), torch.stack(vals, 1)
        s = ordered_sum(w)
        w = torch.where((s == 0)[:, None], torch.ones_like(w), w)
        return ordered_sum(w/ordered_sum(w)[:, None]*v)


def independent_verify(case, z):
    """Independent scalar reference with only the Schaffer workspace bug corrected."""
    if case['group'] != 'cec2022':
        from canonical_benchmarks import independent_numpy
        low, high = BOUNDS[case['name']]
        x = low+(np.asarray(z)+5)*(high-low)/10
        if case['group'] == 'shiftrot':
            with np.load(ROOT/'new_shiftrot_instances'/f"{case['name']}_{case['dim']}D_instance{case['instance']}.npz") as data:
                x = (x-data['shift'])@data['rotation']
        return float(independent_numpy(case['name'], x))
    import importlib.util, os, contextlib, io
    spec = importlib.util.spec_from_file_location('scalar_corrected', ROOT/'Python/CEC2022.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    def schaffer(x, nx, Os, Mr, s_flag, r_flag):
        y = ref.sr_func(x, nx, Os, Mr, 1., s_flag, r_flag)
        value = 0.
        for i in range(nx-1):
            r = (y[i]*y[i]+y[i+1]*y[i+1])**.5
            value += r**.5*(1+np.sin(50*r**.2)**2)
        return value*value/(nx-1)/(nx-1)
    ref.schaffer_F7_func = schaffer
    old = os.getcwd()
    try:
        os.chdir(ROOT/'Python')
        with contextlib.redirect_stdout(io.StringIO()):
            v = ref.cec22_test_func(np.asarray(z)*20, case['dim'], 1, case['fid'])[0]
    finally:
        os.chdir(old)
    return float(v)-BIAS[case['fid']-1]
