import math
import numpy as np

# def evaluate(pred_dict, test_dict, top_k):
#     prec = {k: [] for k in top_k}
#     recall = {k: [] for k in top_k}
#     ndcg = {k: [] for k in top_k}
#
#     for u in test_dict:
#         ranked_list = sorted(pred_dict[u], key=lambda x: x[1], reverse=True)
#         ranked_list = [x[1] for x in ranked_list]
#
#         test_items = test_dict[u]
#
#         for k in top_k:
#             prec_at_k, recall_at_k, ndcg_at_k = prec_recall_ndcg_at_k(ranked_list, test_items, k)
#             prec[k].append(prec_at_k)
#             recall[k].append(recall_at_k)
#             ndcg[k].append(ndcg_at_k)
#
#     prec_dict = {k: np.mean(prec[k]) for k in prec}
#     recall_dict = {k: np.mean(recall[k]) for k in recall}
#     ndcg_dict = {k: np.mean(ndcg[k]) for k in ndcg}
#
#     score_dict = {'prec': prec_dict, 'recall': recall_dict, 'ndcg': ndcg_dict}
#
#     return score_dict

def prec_recall_ndcg_at_k(ranked_list, target_items, k):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(target_items) > k else len(target_items)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    ranked_list = ranked_list[:k]
    target_items_set = set(target_items)
    hits = [(idx, val) for idx, val in enumerate(ranked_list) if val in target_items_set]
    count = len(hits)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)
    prec = float(count / k)
    recall = float(count / len(target_items))
    ndcg = float(dcg_k / idcg_k)
    return prec, recall, ndcg
