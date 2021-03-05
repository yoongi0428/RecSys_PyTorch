import os
import numpy as np
import argparse
import torch

import models
from utils.Params import Params
from utils.Dataset import Dataset
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EASE')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./saves')
parser.add_argument('--conf_dir', type=str, default='./conf')
parser.add_argument('--seed', type=int, default=428)

conf = parser.parse_args()
model_conf = Params(os.path.join(conf.conf_dir, conf.model.lower() + '.json'))
model_conf.update_dict('exp_conf', conf.__dict__)

np.random.seed(conf.seed)
torch.random.manual_seed(conf.seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Dataset(
    data_dir=conf.data_dir,
    data_name=model_conf.data_name,
    train_ratio=model_conf.train_ratio,
    device=device
)

log_dir = os.path.join('saves', conf.model)
logger = Logger(log_dir)
model_conf.save(os.path.join(logger.log_dir, 'config.json'))

eval_pos, eval_target = dataset.eval_data()
item_popularity = dataset.item_popularity
evaluator = Evaluator(eval_pos, eval_target, item_popularity, model_conf.top_k)

model_base = getattr(models, conf.model)
model = model_base(model_conf, dataset.num_users, dataset.num_items, device)

logger.info(model_conf)
logger.info(dataset)

trainer = Trainer(
    dataset=dataset,
    model=model,
    evaluator=evaluator,
    logger=logger,
    conf=model_conf
)

trainer.train()