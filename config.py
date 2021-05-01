from typing import List, Tuple
from dataclasses import dataclass, field

from omegaconf import OmegaConf

@dataclass
class DatasetConfig:
    data_path:str='datasets/ml-100k/u.data'
    dataname:str='ml-1m'
    separator:str='\t'
    binarize_threshold:float=0.0
    implicit:bool=True
    min_item_per_user:int=10
    min_user_per_item:int=1

    protocol:str='holdout' # holdout, leave_one_out
    generalization:str='weak' # weak/strong
    holdout_users:int=600

    valid_ratio:float=0.1
    test_ratio:float=0.2
    leave_k:int=1
    split_random:bool=True

@dataclass
class EvaluatorConfig:
    ks:List[int] = field(default_factory=lambda: [5])

@dataclass
class EarlyStopConfig:
    early_stop:int=25
    early_stop_measure:str='NDCG@10'

@dataclass
class ExperimentConfig:
    debug:bool=False
    save_dir:str='saves'
    num_epochs:int=10
    batch_size:int=256
    verbose:int=0
    print_step:int=1
    test_step:int=1
    test_from:int=1
    model_name:str='EASE'
    num_exp:int=5
    seed:int=2020
    gpu:int=0

def load_config():
    dataset_config = OmegaConf.structured({'dataset' :DatasetConfig})
    evaluator_config = OmegaConf.structured({'evaluator': EvaluatorConfig})
    early_stop_config = OmegaConf.structured({'early_stop': EarlyStopConfig})
    experiment_config = OmegaConf.structured({'experiment': ExperimentConfig})
    
    model_name = experiment_config.experiment.model_name
    # model_config = OmegaConf.structured({'hparams': OmegaConf.load(f"conf/{model_name}.yaml")})
    model_config = OmegaConf.structured(OmegaConf.load(f"conf/{model_name}.yaml"))
    
    config = OmegaConf.merge(dataset_config, evaluator_config, early_stop_config, experiment_config, model_config)
    return config

if __name__ == '__main__':
    config = load_config()
    print(config)