import math
from collections import OrderedDict

import numpy as np
from utils.stats import Statistics
try:
    from .holdout_func import compute_holdout
except:
    raise ImportError('Holdout pyx import error')

from .. import HOLDOUT_METRICS
# HOLDOUT_METRICS = ['Prec', 'Recall', 'NDCG']

def compute_holdout_metrics_cy(pred, target, ks):
    score_cumulator = OrderedDict()
    for metric in HOLDOUT_METRICS:
        score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in ks}
        
    # Cython compute
    # | M1@10 | M1@100 | M2@10 | M2@100| ...
    # (top_k, ground_truth, num_eval_users, metrics_num, Ks):
    results = compute_holdout(pred.astype(np.int32), target, len(HOLDOUT_METRICS), np.array(ks, dtype=np.int32))
    for idx, u in enumerate(target):
        user_results = results[idx].tolist()
        for i, metric in enumerate(HOLDOUT_METRICS):
            for j, k in enumerate(ks):
                score_cumulator[metric][k].update(user_results[i * len(ks) + j])

    return score_cumulator