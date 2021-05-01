import numpy as np

class Statistics:
    def __init__(self, name='AVG'):
        self.name = name
        self.history = []
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        if isinstance(val, list):
            self.history += val
            self.sum += sum(val)
            self.cnt += len(val)
        elif isinstance(val, (int, float, np.int, np.float)):
            self.history.append(val)
            self.sum += val
            self.cnt += 1
        else:
            raise TypeError("\'val\' should be float, int or list of them.")

    @property
    def mean_std(self):
        # mean = self.sum / self.cnt
        mean = np.mean(self.history, dtype=np.float32)
        std = np.std(self.history, dtype=np.float32)
        return mean, std

    @property
    def mean(self):
        # return self.sum / self.cnt
        return np.mean(self.history, dtype=np.float32)

    @property
    def std(self):
        return np.std(self.history, dtype=np.float32)

    def __repr__(self):
        return '%s: mean=%.4f, std=%.4f' % (self.name, self.mean, self.std)