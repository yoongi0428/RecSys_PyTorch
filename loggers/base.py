import abc
from typing import MutableMapping
from argparse import Namespace

import torch
import numpy as np

class Logger(abc.ABC):
    def __init__(self):
        super().__init__()

    def setup_logger(self):
        pass
    
    # @abc.abstractmethod
    # def log_hparams(self, hparams):
    #     raise NotImplementedError('setup_logger is not implemented.')
    
    # @abc.abstractmethod
    # def log_metrics(self, metrics, epoch=None):
    #     raise NotImplementedError('setup_logger is not implemented.')

    def log_image(self, image_name, image, epoch=None):
        pass

    def log_artifact(self, artifact, destination=None):
        pass

    def save(self):
        pass

    def add_dict_prefix(self, dictionary, prefix=None):
        if prefix:
            return {prefix + k: v for k, v in dictionary.items()}
        else:
            return dictionary
    
    # def _flatten_dict(params: Dict[str, Any], delimiter: str = '/') -> Dict[str, Any]:
    def _flatten_dict(self, params, delimiter= '/'):
        """
        Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
        Args:
            params: Dictionary containing the hyperparameters
            delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
        Returns:
            Flattened dict.
        Examples:
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
            {'a/b': 'c'}
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
            {'a/b': 123}
        """

        def _dict_generator(input_dict, prefixes=None):
            prefixes = prefixes[:] if prefixes else []
            if isinstance(input_dict, MutableMapping):
                for key, value in input_dict.items():
                    if isinstance(value, (MutableMapping, Namespace)):
                        value = vars(value) if isinstance(value, Namespace) else value
                        for d in _dict_generator(value, prefixes + [key]):
                            yield d
                    else:
                        yield prefixes + [key, value if value is not None else str(None)]
            else:
                yield prefixes + [input_dict if input_dict is None else str(input_dict)]

        return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}

    # def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    def _sanitize_params(self, params):
        """
        Returns params with non-primitvies converted to strings for logging.
        >>> params = {"float": 0.3,
        ...           "int": 1,
        ...           "string": "abc",
        ...           "bool": True,
        ...           "list": [1, 2, 3],
        ...           "namespace": Namespace(foo=3),
        ...           "layer": torch.nn.BatchNorm1d}
        >>> import pprint
        >>> pprint.pprint(LightningLoggerBase._sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
        {'bool': True,
         'float': 0.3,
         'int': 1,
         'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
         'list': '[1, 2, 3]',
         'namespace': 'Namespace(foo=3)',
         'string': 'abc'}
        """
        return {k: v if type(v) in [bool, int, float, str, torch.Tensor] else str(v) for k, v in params.items()}

if __name__ == '__main__':
    test_dict = {
        'a': 1,
        'b': [1,2,3],
        'c': '[1, 2, 3]',
        'd': {
            'd1': 123,
            'd2': 1
        }
    }
    logger = Logger()
    flattened = logger._flatten_dict(test_dict)
    sanitized = logger._sanitize_params(flattened)
    print(sanitized)