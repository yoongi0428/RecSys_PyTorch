import math
import time
import torch
import numpy as np
from collections import OrderedDict

from utils.Tools import RunningAverage as AVG

class Evaluator:
    def __init__(self, dataset, top_k, split_type):
        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        self.split_type = split_type
        self.target = dataset.test_dict

    def evaluate(self, model, dataset, test_batch_size):
        pred_matrix = model.predict(dataset, test_batch_size)
        topk = self.predict_topk(pred_matrix, max(self.top_k))

        if self.split_type == 'holdout':
            ret = self.eval_holdout(topk, self.target)
        elif self.split_type == 'loo':
            ret = self.eval_loo(topk, self.target)
        else:
            raise NotImplementedError
            
        scores = OrderedDict()
        for metric in ret:
            score_by_ks = ret[metric]
            for k in score_by_ks:
                scores['%s@%d' % (metric, k)] = score_by_ks[k].mean

        return scores

    def predict_topk(self, scores, k):
        # top_k item index (not sorted)
        relevant_items_partition = (-scores).argpartition(k, 1)[:, 0:k]
        
        # top_k item score (not sorted)
        relevant_items_partition_original_value = np.take_along_axis(scores, relevant_items_partition, 1)
        
        # top_k item sorted index for partition
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, 1)
        
        # sort top_k index
        topk = np.take_along_axis(relevant_items_partition, relevant_items_partition_sorting, 1)

        return topk

    def eval_loo(self, topk,  target):
        hr = {k: AVG() for k in self.top_k}
        ndcg = {k: AVG() for k in self.top_k}
        scores = {
            'HR': hr,
            'NDCG': ndcg
        }

        for idx, u in enumerate(target):
            pred_u = topk[idx]
            target_u = target[u][0]
            
            hit_at_k = np.where(pred_u == target_u)[0][0] + 1 if target_u in pred_u else self.max_k + 1

            for k in self.top_k:
                hr_k = 1 if hit_at_k <= k else 0
                ndcg_k = 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0

                scores['HR'][k].update(hr_k)
                scores['NDCG'][k].update(ndcg_k)

        return scores

    def eval_holdout(self, topk, target):
        prec = {k: AVG() for k in self.top_k}
        recall = {k: AVG() for k in self.top_k}
        ndcg = {k: AVG() for k in self.top_k}
        scores = {
            'Prec': prec,
            'Recall': recall,
            'NDCG': ndcg
        }

        for idx, u in enumerate(target):
            pred_u = topk[idx]
            target_u = target[u]
            num_target_items = len(target_u)
            for k in self.top_k:
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
                recall_k = num_hits / min(num_target_items, k)
                ndcg_k = dcg_k / idcg_k

                scores['Prec'][k].update(prec_k)
                scores['Recall'][k].update(recall_k)
                scores['NDCG'][k].update(ndcg_k)

        return scores