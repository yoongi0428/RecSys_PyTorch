try:
    from utils.backend.cython.tool import predict_topk
    from utils.backend.cython.holdout import compute_holdout
except:
    print('evaluation with python backend...')
    from utils.backend.python.tool import predict_topk