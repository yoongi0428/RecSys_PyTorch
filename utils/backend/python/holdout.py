import math
from utils.Tools import RunningAverage as AVG

def compute_holdout(topk, target, Ks):
    prec = {k: AVG() for k in Ks}
    recall = {k: AVG() for k in Ks}
    ndcg = {k: AVG() for k in Ks}
    scores = {
        'Prec': prec,
        'Recall': recall,
        'NDCG': ndcg
    }
    for idx, u in enumerate(target):
        pred_u = topk[idx]
        target_u = target[u]
        num_target_items = len(target_u)
        for k in Ks:
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

            scores['Prec'][k].update(prec_k)
            scores['Recall'][k].update(recall_k)
            scores['NDCG'][k].update(ndcg_k)
    return scores