from loggers.base import Logger

class ConsoleLogger(Logger):
    def __init__(self, log_dir):
        self.log_dir = log_dir
    
    def log_metrics(self, metrics, epoch=None, prefix=None):
        log_str = ''
        if epoch is not None:
            log_str += '[epoch %3d]' % epoch
        
        metric_str_list = ['%s=%.4f' % (k, v) for k, v in metrics.items()]
        log_str += ', '.join(metric_str_list)

        print(log_str)
    
    def save(self):
        pass
    
    def info(self, msg):
        print(msg)

    def close(self):
        for handle in self.logger.handlers:
            handle.close()
            self.logger.removeHandler(handle)