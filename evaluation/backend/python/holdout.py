import math
from collections import OrderedDict
import numpy as np

from utils.stats import Statistics
from .. import HOLDOUT_METRICS
# from evaluation.backend import HOLDOUT_METRICS

# HOLDOUT_METRICS = ['Prec', 'Recall', 'NDCG']

def compute_holdout_metrics_py(pred, target, ks):
    score_cumulator = OrderedDict()
    for metric in HOLDOUT_METRICS:
        score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in ks}
    
    hits = []
    for idx, u in enumerate(target):
        pred_u = pred[idx]
        target_u = target[u]
        num_target_items = len(target_u)
        for k in ks:
            pred_k = pred_u[:k]
            hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
            num_hits = len(hits_k)

            idcg_k = 0.0
            for i in range(1, min(num_target_items, k) + 1):
                idcg_k += 1 / math.log(i + 1, 2)

            dcg_k = 0.0
            for idx, item in hits_k:
                dcg_k += 1 / math.log(idx + 1, 2)
            
            prec_k = num_hits / k
            recall_k = num_hits / num_target_items
            ndcg_k = dcg_k / idcg_k

            score_cumulator['Prec'][k].update(prec_k)
            score_cumulator['Recall'][k].update(recall_k)
            score_cumulator['NDCG'][k].update(ndcg_k)
        
            hits.append(len(hits_k))
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

#         hits = []
#         for idx, u in enumerate(target):
#             pred_u = topk[idx]
#             target_u = target[u]
#             num_target_items = len(target_u)
#             for k in self.top_k:
#                 pred_k = pred_u[:k]
#                 hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
#                 num_hits = len(hits_k)

#                 idcg_k = 0.0
#                 for i in range(1, min(num_target_items, k) + 1):
#                     idcg_k += 1 / math.log(i + 1, 2)

#                 dcg_k = 0.0
#                 for idx, item in hits_k:
#                     dcg_k += 1 / math.log(idx + 1, 2)

#                 if num_hits:
#                     pass
                
#                 prec_k = num_hits / k
#                 recall_k = num_hits / min(num_target_items, k)
#                 ndcg_k = dcg_k / idcg_k

#                 score_cumulator['Prec'][k].update(prec_k)
#                 score_cumulator['Recall'][k].update(recall_k)
#                 score_cumulator['NDCG'][k].update(ndcg_k)
            
#                 hits.append(len(hits_k))
#         return score_cumulator