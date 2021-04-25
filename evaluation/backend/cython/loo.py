import math
from collections import OrderedDict
import numpy as np

from utils.Statistics import Statistics

try:
    from .loo import compute_loo
except:
    raise ImportError('Cython loo import error')

class LOOEvaluator:
    def __init__(self, top_k, eval_pos, eval_target, eval_neg_candidates=None):
        self.top_k = top_k
        self.max_k = max(top_k)
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.eval_neg_candidates = eval_neg_candidates

    def init_score_cumulator(self):
        score_cumulator = OrderedDict()
        for metric in ['HR', 'NDCG']:
            score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in self.top_k}
        return score_cumulator

    def compute_metrics(self, topk, target, score_cumulator=None):
        if score_cumulator is None:
            score_cumulator = self.init_score_cumulator()
        
        results = compute_loo(topk.astype(np.int32), target, 2, np.array(self.top_k, dtype=np.int32))
        for idx, u in enumerate(target):
            user_results = results[idx].tolist()
            for i, metric in enumerate(['HR', 'NDCG']):
                for j, k in enumerate(self.top_k):
                    score_cumulator[metric][k].update(user_results[i * len(self.top_k) + j])

        return score_cumulator