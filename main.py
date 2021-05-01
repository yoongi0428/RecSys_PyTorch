# Import packages
import os
import torch

import models
from data.dataset import UIRTDataset
from evaluation.evaluator import Evaluator
from experiment.early_stop import EarlyStop

from loggers import FileLogger, CSVLogger
from utils.general import make_log_dir, set_random_seed
from config import load_config
""" 
    Configurations
"""
config = load_config()

exp_config = config.experiment
gpu_id = exp_config.gpu
seed = exp_config.seed

dataset_config = config.dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    set_random_seed(seed)

    """ 
        Dataset
    """
    dataset = UIRTDataset(**dataset_config)

    # """ 
    #     Early stop
    # """
    # early_stop = EarlyStop(**config['EarlyStop'])


    """ 
        Model base class
    """
    model_name = config.experiment.model_name
    model_base = getattr(models, model_name)
    hparams = config.hparams

    """ 
        Logger
    """

    log_dir = make_log_dir(os.path.join(exp_config.save_dir, model_name))
    logger = FileLogger(log_dir)
    csv_logger = CSVLogger(log_dir)

    # Save log & dataset config.
    logger.info(config)
    logger.info(dataset)

    valid_input, valid_target = dataset.valid_input, dataset.valid_target
    evaluator = Evaluator(valid_input, valid_target, protocol=dataset.protocol, ks=config.evaluator.ks)

    model = model_base(dataset, hparams, device)

    ret = model.fit(dataset, exp_config, evaluator=evaluator, loggers=[logger, csv_logger])
    print(ret['scores'])
    
    csv_logger.save()