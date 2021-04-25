import os
import logging
from time import strftime, sleep
from loggers.base import Logger

class FileLogger(Logger):
    def __init__(self, log_dir):
        log_file_format = "[%(lineno)d]%(asctime)s: %(message)s"
        log_console_format = "%(message)s"

        # Main logger
        self.log_dir = log_dir

        self.logger = logging.getLogger(log_dir)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_console_format))
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(log_dir, 'experiments.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_file_format))
        self.logger.addHandler(file_handler)
    
    def log_metrics(self, metrics, epoch=None, prefix=None):
        log_str = ''
        if epoch is not None:
            log_str += '[epoch %3d]' % epoch
        
        metric_str_list = ['%s=%.4f' % (k, v) for k, v in metrics.items()]
        log_str += ', '.join(metric_str_list)

        self.info(log_str)
    
    def save(self):
        pass
    
    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        for handle in self.logger.handlers:
            handle.close()
            self.logger.removeHandler(handle)