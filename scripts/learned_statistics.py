"""The joint 72-comparison post-hoc family used in manuscript Table III."""
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

def learned_significance(values):
    """Table III family: 72 two-sided tests, jointly Holm-adjusted."""
    comparisons = []
    for method in ['AutoPSO', 'DEDQN', 'SYMBOL', 'GLEET', 'RLHPSDE', 'RLEPSO']:
        for fid in range(1, 13):
            a, b = [np.where(np.asarray(values[(fid, m)]) < 1e-8, 0., values[(fid, m)])
                    for m in ['CoRePSO', method]]
            if len(a) != 11 or len(b) != 11:
                raise ValueError('Eleven observations per comparison required')
            if method == 'AutoPSO':
                delta = a-b
                delta[np.abs(delta) <= 1e-8+1e-9*np.maximum(np.abs(a),np.abs(b))] = 0.
                p = 1. if np.all(delta == 0.) else float(wilcoxon(delta, alternative='two-sided', method='auto').pvalue)
                test = 'paired Wilcoxon signed-rank'
            else:
                p = float(mannwhitneyu(a,b,alternative='two-sided',method='asymptotic',use_continuity=True).pvalue)
                test = 'independent Mann-Whitney U, tie-corrected normal approximation'
            diff = float(a.mean()-b.mean())
            tol = 1e-8+1e-9*max(abs(a.mean()),abs(b.mean()))
            comparisons.append(dict(fid=fid,method=method,test=test,p_raw=p,
                                    direction='win' if diff < -tol else 'loss' if diff > tol else 'tie'))
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]['p_raw'])
    previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(order)-rank)*comparisons[i]['p_raw']))
        comparisons[i]['p_holm'] = previous
        comparisons[i]['significance'] = comparisons[i]['direction'] if previous < .05 else 'not_rejected'
    return comparisons
