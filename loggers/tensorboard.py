import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as hparams_tb
from logger.base import Logger

class TensorboardLogger(Logger):
    def __init__(self,
                log_dir:str,
                experiment_name:str,
                hparams:dict,
                log_graph:bool=False):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_graph = log_graph
        self.hparams = hparams
        
        self.initialize()

    @property
    def system_property(self):
        return self.experiment.get_system_properties()

    def initialize(self):
        self.experiment = SummaryWriter(log_dir=self.log_dir)
    
    def add_dict_prefix(self, metrics, metric_prefix):
        return {f'{metric_prefix}/{k}':v for k, v in metrics.items()}

    def log_hparams(self, hparams, metrics=None):
        self.hparams.update(hparams)
        flattened = self._flatten_dict(self.hparams)
        for k, v in flattened.items():
            if isinstance(k, (int, float, str, bool, torch.tensor)):
                flattened[k] = str(v)
        if metrics:
            self.experiment.add_hparams(flattened, dict(metrics))

    def _log_metric(self, metric_name, value, epoch=None):
        # metric_name = self.add_metric_prefix(metric_name)
        if epoch is None:
            self.experiment.add_scalar(metric_name, value)
        else:
            self.experiment.add_scalar(metric_name, value, epoch)

    def log_metrics(self, metrics, epoch=None, prefix=None):
        if prefix is not None:
            metrics = self.add_dict_prefix(metrics, prefix)
            
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            try:
                self._log_metric(k, v, epoch)
            except Exception as e:
                m = f'\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor.'
                type(e)(e.message + m)            

    def log_image(self, image_name, image, epoch=None):
        if epoch is None:
            self.experiment.add_image(image_name, image)
        else:
            self.experiment.add_image(image_name, image, epoch)

    def log_artifact(self, artifact, destination=None):
        pass