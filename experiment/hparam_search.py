import os
import time
import copy

import optuna
from experiment import fit_model
from utils import ResultTable, set_random_seed
from logger import Logger

class GridSearch:
    def __init__(self, model_base, dataset, early_stop, config, device, seed=2020, num_parallel=1):
        self.model_base = model_base
        self.dataset = dataset
        self.early_stop = early_stop
        self.metric_to_optimize = early_stop.early_stop_measure
        self.config = config
        self.seed = seed
        self.num_parallel = num_parallel

        self.exp_logger = []

        self.search_params = config['GridSearch']
        self.search_space = self.generate_search_space()
        self.search_param_names = list(self.search_params.keys())

        # (score, [list of params])
        self.result = []
        self.valid_score = []
        self.exp_num = 0

        self.best_exp_num = 0
        self.best_score = -1
        self.best_params = None

        self.device = device

    def generate_search_space(self):
        search_params = list(self.search_params.keys())
        search_space = []
        for i, param_name in enumerate(search_params):
            new_space = []
            
            if not isinstance(self.search_params[param_name], list):
                self.config[param_name] = self.search_params[param_name]

            for param_value in self.search_params[param_name]:
                if i == 0:
                    new_space.append({param_name: param_value})
                else:
                    tmp_list = copy.deepcopy(search_space)
                    for param_setting in tmp_list:
                        param_setting[param_name] = param_value
                    new_space += tmp_list
            search_space = new_space

        return search_space

    def optimize(self, model, evaluator, early_stop, neptune, logger, config, fold=None):
        valid_score, train_time = fit_model(model, self.dataset, evaluator, early_stop, neptune, logger, config)

        self.valid_score.append(valid_score)

        score = valid_score[self.metric_to_optimize]

        return score

    def objective_function(self, cur_space):
        base_log_dir = self.base_logger.log_dir
        self.exp_num += 1

        # update config
        config_copy = copy.deepcopy(self.base_config)
        config_copy.update_params(cur_space)

        # create model
        set_random_seed(self.seed)
        model = self.model_base(self.dataset, config_copy['Model'], self.device)

        exp_logger = self.init_search_logger(base_log_dir, self.exp_num)

        config_path = os.path.join(exp_logger.log_dir, 'model_config.cfg')
        self.config.save_model_config(config_path)

        if self.neptune is not None:
            dest = os.path.join(*config_path.split('/')[3:])
            self.neptune.log_artifact(config_path, destination=dest)

        score = self.optimize(model, self.evaluator, self.early_stop, self.neptune, exp_logger, config_copy, self.fold)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = cur_space

        self.base_logger.info('Exp %d value=%.4f, current best value=%.4f with parameters %s\n' % (self.exp_num, score, self.best_score, str(self.best_params)))

        if self.neptune is not None:
            main_log_path = os.path.join(exp_logger.log_dir, 'experiments.log')
            dest = os.path.join(*main_log_path.split('/')[3:])
            self.neptune.log_artifact(main_log_path, destination=dest)

        exp_logger.close()
        del model

        return score

    def init_search_logger(self, base_dir, exp_num):
        exp_dir = os.path.join(base_dir, 'exp_%d' % exp_num)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        logger = Logger(exp_dir)
        self.exp_logger.append(logger)
        return logger

    def init(self):
        self.exp_logger = []
        
        self.result = []
        self.valid_score = []
        self.exp_num = 0

        self.best_exp_num = 0
        self.best_score = -1
        self.best_params = None

    def search(self, evaluator, neptune_manager, logger, config, fold=None):
        self.result = []
        self.neptune = neptune_manager
        self.evaluator=evaluator
        self.base_logger=logger
        self.base_config=config
        self.fold=fold
        
        start = time.time()
        scores = [self.objective_function(cur_space) for cur_space in self.search_space]
        search_time = time.time() - start
        
        results = [
            {'number': i, 'value': scores[i], 'params': cur_space}
            for i, cur_space in enumerate(self.search_space)]
        
        all_trials = sorted(results, key=lambda x: x['value'], reverse=True)
        best_trial = all_trials[0]

        self.best_exp_num = best_trial['number']
        self.best_score = best_trial['value']
        self.best_params = best_trial['params']

        search_result_table = ResultTable(table_name='Param Search Result', header=list(self.best_params.keys()) + [self.metric_to_optimize], float_formatter='%.6f')
        for trial in all_trials:
            row_dict = {}
            row_dict[self.metric_to_optimize] = trial['value']
            for k, v in trial['params'].items():
                row_dict[k] = v
            search_result_table.add_row('Exp %d' % trial['number'], row_dict)

        return search_result_table, search_time

    @ property
    def best_result(self):
        best_dir = self.exp_logger[self.best_exp_num].log_dir
        best_valid_score = self.valid_score[self.best_exp_num]

        best_config = copy.deepcopy(self.config)
        best_config.update_params(self.best_params)

        return best_dir, best_valid_score, best_config['Model']

class BayesianSearch:
    def __init__(self, model_base, dataset, early_stop, config, device, seed=2020, num_trials=10, num_parallel=1):
        self.model_base = model_base
        self.dataset = dataset
        self.early_stop = early_stop
        self.metric_to_optimize = early_stop.early_stop_measure
        self.config = config
        self.seed = seed
        
        self.num_trials = num_trials
        self.num_parallel = num_parallel

        self.exp_logger = []

        self.search_params = config['BayesSearch']

        # (score, [list of params])
        self.result = None
        self.valid_score = []
        self.exp_num = 0

        self.best_exp_num = 0
        self.best_score = -1
        self.best_param = None

        self.device = device

    def generate_search_space(self, trial):
        # For integer: ('int', [low, high])
        # For float: ('float', 'domai', [low, high])
        # For categorical: ('categorical', [list of choices])
        search_spaces = {}

        for param_name in self.search_params:
            space = self.search_params[param_name]
            space_type = space[0]

            if space_type == 'categorical':
                search_spaces[param_name] = trial.suggest_categorical(param_name, space[1])
            elif space_type == 'int':
                [low, high] = space[1]
                search_spaces[param_name] = trial.suggest_int(param_name, low, high)
            elif space_type == 'float':
                domain = space[1]
                [low, high] = space[2]
                if domain == 'uniform':
                    search_spaces[param_name] = trial.suggest_uniform(param_name, low, high)
                elif domain == 'loguniform':
                    search_spaces[param_name] = trial.suggest_loguniform(param_name, low, high)
                else:
                    raise ValueError('Unsupported float search domain: %s' % domain)
            else:
                raise ValueError('Search parameter type error: %s' % space_type)
        
        return search_spaces

    def optimize(self, model, early_stop, logger, config):
        valid_score, train_time = fit_model(model, self.dataset, self.evaluator, early_stop, None, logger, config)

        self.valid_score.append(valid_score)
        
        score = valid_score[self.metric_to_optimize]

        return score

    def objective_function(self, trial):
        # update config
        config_copy = copy.deepcopy(self.config)

        # config_to_update = dict(zip(self.search_param_names, cur_space))
        search_params = self.generate_search_space(trial)
        config_copy.update_params(search_params)
        
        # create model
        set_random_seed(self.seed)
        model = self.model_base(self.dataset, config_copy['Model'], self.device)

        exp_logger = self.init_search_logger()

        if self.neptune is not None:
            config_path = os.path.join(exp_logger.log_dir, 'model_config.cfg')
            self.config.save_model_config(config_path)
            dest = os.path.join(*config_path.split('/')[3:])
            self.neptune.log_artifact(config_path, destination=dest)

        score = self.optimize(model, self.early_stop, exp_logger, config_copy)

        if self.neptune is not None:
            main_log_path = os.path.join(exp_logger.log_dir, 'experiments.log')
            dest = os.path.join(*main_log_path.split('/')[3:])
            self.neptune.log_artifact(main_log_path, destination=dest)

        exp_logger.close()

        self.exp_num += 1

        del model

        return score

    def init_search_logger(self):
        exp_dir = os.path.join(self.base_dir, 'exp_%d' % self.exp_num)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        logger = Logger(exp_dir)
        self.exp_logger.append(logger)
        return logger

    def init(self):
        self.exp_logger = []
        
        self.result = []
        self.valid_score = []
        self.exp_num = 0

        self.best_exp_num = 0
        self.best_score = -1
        self.best_params = None

    def search(self, evaluator, neptune_manager, logger, config, fold=None):
        self.evaluator = evaluator
        self.base_logger = logger
        self.base_dir = logger.log_dir
        self.neptune = neptune_manager
        self.fold = fold

        start = time.time()
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective_function, n_trials=self.num_trials, n_jobs=self.num_parallel)

        search_time = time.time() - start

        all_trials = sorted(self.study.trials, key=lambda x: x.value, reverse=True)
        best_trial = all_trials[0]

        self.best_exp_num = best_trial.number
        self.best_score = best_trial.value
        self.best_params = best_trial.params

        search_result_table = ResultTable(table_name='Param Search Result', header=list(self.best_params.keys()) + [self.metric_to_optimize], float_formatter='%.6f')
        for trial in all_trials:
            row_dict = {}
            row_dict[self.metric_to_optimize] = trial.value
            for k, v in trial.params.items():
                row_dict[k] = v
            search_result_table.add_row('Exp %d' % trial.number, row_dict)
        
        if optuna.visualization.is_available():
            optuna.visualization.plot_optimization_history(self.study)

        return search_result_table, search_time

    @ property
    def best_result(self):
        best_dir = self.exp_logger[self.best_exp_num].log_dir
        best_valid_score = self.valid_score[self.best_exp_num]

        best_config = copy.deepcopy(self.config)
        best_config.update_params(self.best_params)

        return best_dir, best_valid_score, best_config['Model']