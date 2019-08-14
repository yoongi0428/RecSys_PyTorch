import math
import time
import torch
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from utils.Tools import RunningAverage as AVG
from utils.Metrics import prec_recall_ndcg_at_k

class Evaluator:
    def __init__(self, dataset, top_k, split_type, num_threads):
        # self.train_matrix = dataset.train_matrix.tocsr()
        # self.test_matrix = dataset.test_matrix.tocsr()
        # self.num_users, self.num_items = self.train_matrix.shape

        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        self.split_type = split_type
        self.num_threads = num_threads

    def evaluate(self, model, dataset, test_batch_size):
        self.model = model
        self.dataset = dataset
        pred_matrix = model.predict(dataset, test_batch_size)
        self.pred_matrix = pred_matrix#.detach().cpu().numpy()
        # self.eval_model = model
        # predictions = []
        # test_user_start = time.time()
        # for u in range(self.dataset.num_users):
        #     predictions.append(model.predict(u, dataset.eval_items[u], dataset))
        #     print('test %d user: ' % u, time.time() - test_user_start)
        #
        # target_items = list(dataset.test_dict.values())

        if self.split_type == 'holdout':
            ret = self.eval_holdout()
        elif self.split_type == 'loo':
            ret = self.eval_loo()
        else:
            raise NotImplementedError

        return ret

    def eval_loo(self):
        num_users, num_items = self.pred_matrix.shape

        hrs = {k: AVG() for k in self.top_k}
        ndcgs = {k: AVG() for k in self.top_k}
        maps = AVG()
        aucs = AVG()

        if self.num_threads > 1:
            with ThreadPoolExecutor() as executor:
                ret = executor.map(self.eval_loo_per_user, range(num_users))
                for i, (hr_u, ndcg_u, ap_u, auc_u) in enumerate(ret):
                    for k in self.top_k:
                        hrs[k].update(hr_u[k])
                        ndcgs[k].update(ndcg_u[k])
                        maps.update(ap_u)
                        aucs.update(auc_u)
        else:
            for u in range(self.dataset.num_users):
                hr_u, ndcg_u, ap_u, auc_u = self.eval_loo_per_user(u)
                for k in self.top_k:
                    hrs[k].update(hr_u[k])
                    ndcgs[k].update(ndcg_u[k])
                    maps.update(ap_u)
                    aucs.update(auc_u)

        ret = OrderedDict()
        hr_k = dict(map(lambda x: ('hr@%d' % x, hrs[x].value), hrs))
        ndcg_k = dict(map(lambda x: ('ndcg@%d' % x, ndcgs[x].value), ndcgs))
        ap = {'map': maps.value}
        auc = {'auc': aucs.value}

        ret.update(hr_k)
        ret.update(ndcg_k)
        ret.update(ap)
        ret.update(auc)

        return ret

    def eval_loo_per_user(self, user_id):
        target_items = self.dataset.test_dict[user_id]
        predictions = self.pred_matrix[user_id]
        ranked_list = torch.argsort(predictions, -1, True)
        # ranked_list = sorted(predictions, key=lambda x: x[-1], reverse=True)
        # ranked_list = np.array([x[0] for x in ranked_list])

        # hit_pos = np.where(target_items == ranked_list)[0][0] + 1
        hit_pos = (ranked_list == target_items[0]).nonzero().item() + 1

        hr = {k: 1 if hit_pos <= k else 0 for k in self.top_k}
        ndcg = {k: 1 / math.log(hit_pos + 1, 2) if hit_pos <= k else 0 for k in self.top_k}
        ap = 1 / hit_pos
        auc = 1 - (1 / 2) * (1 - 1 / hit_pos)
        return hr, ndcg, ap, auc

    def eval_holdout(self):
        num_users, num_items = self.pred_matrix.shape

        prec = {k: AVG() for k in self.top_k}
        recall = {k: AVG() for k in self.top_k}
        ndcg = {k: AVG() for k in self.top_k}

        if self.num_threads > 1:
            with ThreadPoolExecutor() as executor:
                ret = executor.map(self.eval_holdout_per_user, range(num_users))
                for i, (prec_u, recall_u, ndcg_u) in enumerate(ret):
                    for k in self.top_k:
                        prec[k].update(prec_u[k])
                        recall[k].update(recall_u[k])
                        ndcg[k].update(ndcg_u[k])
        else:
            for u in range(num_users):
                prec_u, recall_u, ndcg_u = self.eval_holdout_per_user(u)
                for k in self.top_k:
                    prec[k].update(prec_u[k])
                    recall[k].update(recall_u[k])
                    ndcg[k].update(ndcg_u[k])

        ret = OrderedDict()
        prec_k = dict(map(lambda x: ('prec@%d' % x, prec[x].value), prec))
        recall_k = dict(map(lambda x: ('recall@%d' % x, recall[x].value), recall))
        ndcg_k = dict(map(lambda x: ('ndcg@%d' % x, ndcg[x].value), ndcg))

        ret.update(prec_k)
        ret.update(recall_k)
        ret.update(ndcg_k)

        return ret

    def eval_holdout_per_user(self, user_id):
        prec = {}
        recall = {}
        ndcg = {}

        target_items = self.dataset.test_dict[user_id]
        predictions = self.pred_matrix[user_id]
        ranked_list = torch.argsort(predictions, -1, True).numpy()

        # target_items = self.test_matrix[user_id].indices
        #
        # train_index = (self.train_matrix[user_id] > 0).tocoo().col
        # pred_u = self.pred_matrix[user_id]
        # pred_u[train_index] = float('-inf')
        #
        # ranked_list = np.argsort(pred_u, 0)[::-1]
        for k in self.top_k:
            _prec, _recall, _ndcg = prec_recall_ndcg_at_k(ranked_list, target_items, k)
            prec[k] = _prec
            recall[k] = _recall
            ndcg[k] = _ndcg
        return prec, recall, ndcg