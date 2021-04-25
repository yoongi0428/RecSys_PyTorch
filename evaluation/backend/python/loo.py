import math
from collections import OrderedDict
import numpy as np

from utils.stats import Statistics

LOO_METRICS = ['HR', 'NDCG']

def compute_loo_metrics_py(pred, target, ks):
    score_cumulator = OrderedDict()
    for metric in LOO_METRICS:
        score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in ks}
    
    max_k = max(ks)
    for idx, u in enumerate(target):
        pred_u = pred[idx]
        target_u = target[u][0]
        
        hit_at_k = np.where(pred_u == target_u)[0][0] + 1 if target_u in pred_u else max_k + 1

        for k in ks:
            hr_k = 1 if hit_at_k <= k else 0
            ndcg_k = 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0

            score_cumulator['HR'][k].update(hr_k)
            score_cumulator['NDCG'][k].update(ndcg_k)

    return score_cumulator

# class LOOEvaluator:
#     def __init__(self, top_k, eval_pos, eval_target, eval_neg_candidates=None):
#         self.top_k = top_k
#         self.max_k = max(top_k)
#         self.eval_pos = eval_pos
#         self.eval_target = eval_target
#         self.eval_neg_candidates = eval_neg_candidates

#     def init_score_cumulator(self):
#         score_cumulator = OrderedDict()
#         for metric in ['HR', 'NDCG']:
#             score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in self.top_k}
#         return score_cumulator

#     def compute_metrics(self, topk, target, score_cumulator=None):
#         if score_cumulator is None:
#             score_cumulator = self.init_score_cumulator()

#         for idx, u in enumerate(target):
#             pred_u = topk[idx]
#             target_u = target[u][0]
            
#             hit_at_k = np.where(pred_u == target_u)[0][0] + 1 if target_u in pred_u else self.max_k + 1

#             for k in self.top_k:
#                 hr_k = 1 if hit_at_k <= k else 0
#                 ndcg_k = 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0

#                 score_cumulator['HR'][k].update(hr_k)
#                 score_cumulator['NDCG'][k].update(ndcg_k)

#         return score_cumulator