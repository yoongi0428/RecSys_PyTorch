import math
from collections import OrderedDict

import numpy as np
from utils.stats import Statistics
try:
    from .holdout_func import compute_holdout
except:
    raise ImportError('Holdout pyx import error')

HOLDOUT_METRICS = ['Prec', 'Recall', 'NDCG']

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

# class HoldoutEvaluator:
#     def __init__(self, top_k, eval_pos, eval_target, eval_neg_candidates=None):
#         self.top_k = top_k
#         self.max_k = max(top_k)
#         self.eval_pos = eval_pos
#         self.eval_target = eval_target
#         self.eval_neg_candidates = eval_neg_candidates

#     def init_score_cumulator(self):
#         score_cumulator = OrderedDict()
#         for metric in ['Prec', 'Recall', 'NDCG']:
#             score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in self.top_k}
#         return score_cumulator

#     def compute_metrics(self, topk, target, score_cumulator=None):
#         if score_cumulator is None:
#             score_cumulator = self.init_score_cumulator()
        
#         # Cython compute
#         # | M1@10 | M1@100 | M2@10 | M2@100| ...
#         # (top_k, ground_truth, num_eval_users, metrics_num, Ks):
#         results = compute_holdout(topk.astype(np.int32), target, 3, np.array(self.top_k, dtype=np.int32))
#         for idx, u in enumerate(target):
#             user_results = results[idx].tolist()
#             for i, metric in enumerate(['Prec', 'Recall', 'NDCG']):
#                 for j, k in enumerate(self.top_k):
#                     score_cumulator[metric][k].update(user_results[i * len(self.top_k) + j])

#         return score_cumulator