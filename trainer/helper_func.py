import copy
from time import time

def fit_model(model, dataset, exp_config, evaluator, early_stop, loggers, run_n=-1):
    # initialize experiment
    early_stop.initialize()

    # train model
    fit_start = time()
    best_valid_score = model.fit(dataset, exp_config, evaluator, early_stop, loggers)
    train_time = time() - fit_start
    return best_valid_score, train_time