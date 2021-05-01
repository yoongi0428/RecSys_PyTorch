import time
import numpy as np
from typing import Iterable
from collections import OrderedDict

from .backend import eval_func_router, predict_topk_func
from data.data_batcher import DataBatcher
from utils.types import sparse_to_dict

class Evaluator:
    def __init__(self, eval_input, eval_target, protocol, ks, eval_batch_size=1024):
        """

        """
        self.top_k = sorted(list(ks)) if isinstance(ks, Iterable) else [ks]
        self.max_k = max(self.top_k)
        
        self.batch_size = eval_batch_size
        self.eval_input = eval_input
        self.eval_target = sparse_to_dict(eval_target)

        self.protocol = protocol

        self._register_eval_func()
    
    def evaluate(self, model, mean=True):
        # Switch to eval mode
        model.eval()

        # eval users
        eval_users = np.array(list(self.eval_target.keys()))
        num_users = len(eval_users)
        num_items = self.eval_input.shape

        output = model.predict(eval_users, self.eval_input, self.batch_size)

        pred = self.predict_topk(output.astype(np.float32), self.max_k)

        score_cumulator = self.eval_func(pred, self.eval_target, self.top_k)

        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                if mean:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
                else:
                    scores['%s@%d' % (metric, k)] = score_by_ks[k].history

        # return
        return scores
    
    def _register_eval_func(self):
        self.eval_func = eval_func_router[self.protocol]
        self.predict_topk = predict_topk_func