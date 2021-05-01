import math
from collections import OrderedDict
import numpy as np

from utils.stats import Statistics

try:
    from .loo_func import compute_loo
except:
    raise ImportError('Cython loo import error')

from .. import LOO_METRICS
# from evaluation.backend import LOO_METRICS

def compute_loo_metrics_cy(pred, target, ks):
    score_cumulator = OrderedDict()
    for metric in LOO_METRICS:
        score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in ks}
    
    results = compute_loo(pred.astype(np.int32), target, 2, np.array(ks))
    for idx, u in enumerate(target):
        user_results = results[idx].tolist()
        for i, metric in enumerate(LOO_METRICS):
            for j, k in enumerate(ks):
                score_cumulator[metric][k].update(user_results[i * len(ks) + j])

    return score_cumulator