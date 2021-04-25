import io
import os
import csv
from time import strftime
from collections import OrderedDict

from loggers.base import Logger

class CSVLogger(Logger):
    
    LOG_FILE = 'results.csv'

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.hparams = {}
        self.metrics_history = None
        self.headers = None
    
    def log_metrics(self, metrics, epoch=None, prefix=None):
        if prefix:
            metrics = self.add_dict_prefix(metrics)
            
        if self.metrics_history is None:
            self.metrics_history = []
            self.headers = list(metrics.keys())
            if 'epoch' not in self.headers:
                self.headers = ['epoch'] + self.headers

        for key in self.headers:
            if key not in metrics:
                metrics[key] = ' - '
        
        for key in metrics:
            if key not in self.headers:
                self.headers.append(key)
        
        if epoch is None:
            epoch = len(self.metrics_history)

        metrics['epoch'] = epoch

        self.metrics_history.append(self.ensure_ordered_dict(metrics))
    
    def check_columns(self, d):
        pass
    
    def ensure_ordered_dict(self, d):
        if isinstance(d, OrderedDict):
            return d
        else:
            return OrderedDict(d)
    
    def save(self):
        if not self.metrics_history:
            return

        if 'train_loss' in self.headers:
            idx = self.headers.index('train_loss')
            self.headers = [self.headers.pop(idx)] + self.headers
        if 'elapsed' in self.headers:
            idx = self.headers.index('elapsed')
            self.headers = [self.headers.pop(idx)] + self.headers
        if 'epoch' in self.headers:
            idx = self.headers.index('epoch')
            self.headers = [self.headers.pop(idx)] + self.headers
        
        log_file = os.path.join(self.log_dir, self.LOG_FILE)
        with io.open(log_file, 'w', newline='') as f:
            self.writer = csv.DictWriter(f, fieldnames=self.headers)
            self.writer.writeheader()
            self.writer.writerows(self.metrics_history)