import os
import math
import time
import datetime

import random
import numpy as np
import torch

def make_log_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    existing_dirs = os.listdir(save_dir)
    if len(existing_dirs) == 0:
        idx = 0
    else:
        idx_list = sorted([int(d.split('_')[0]) for d in existing_dirs])
        idx = idx_list[-1] + 1

    cur_log_dir = '%d_%s' % (idx, time.strftime('%Y%m%d-%H%M'))
    full_log_dir = os.path.join(save_dir, cur_log_dir)

    if not os.path.exists(full_log_dir):
        os.mkdir(full_log_dir)
    else:
        full_log_dir = make_log_dir(save_dir)

    return full_log_dir

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getlocaltime():
    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())

def seconds_to_hms(second):
    return str(datetime.timedelta(seconds=second))