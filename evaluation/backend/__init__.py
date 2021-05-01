HOLDOUT_METRICS = ['Prec', 'Recall', 'NDCG']
LOO_METRICS = ['HR', 'NDCG']

try:
    from .cython.loo import compute_loo_metrics_cy
    from .cython.holdout import compute_holdout_metrics_cy
    from .cython.func import predict_topk_cy
    
    CYTHON_OK = True
except:
    print('evaluation with python backend...')
    from .python.loo import compute_loo_metrics_py
    from .python.holdout import compute_holdout_metrics_py
    from .python.func import predict_topk_py

    CYTHON_OK = False

if CYTHON_OK:
    eval_func_router = {
        'leave_one_out': compute_loo_metrics_cy,
        'holdout': compute_holdout_metrics_cy
    }
    predict_topk_func = predict_topk_cy
else:
    eval_func_router = {
        'leave_one_out': compute_loo_metrics_py,
        'holdout': compute_holdout_metrics_py
    }
    predict_topk_func = predict_topk_py