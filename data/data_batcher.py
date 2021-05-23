import torch
import numpy as np

class BatchSampler:
    def __init__(self, data_size, batch_size, drop_remain=False, shuffle=False):
        self.data_size = data_size
        self.batch_size = batch_size
        self.drop_remain = drop_remain
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.data_size)
        else:
            perm = range(self.data_size)

        batch_idx = []
        for idx in perm:
            batch_idx.append(idx)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []
        if len(batch_idx) > 0 and not self.drop_remain:
            yield batch_idx

    def __len__(self):
        if self.drop_remain:
            return self.data_size // self.batch_size
        else:
            return int(np.ceil(self.data_size / self.batch_size))

class DataBatcher:
    def __init__(self, *data_source, batch_size, drop_remain=False, shuffle=False):
        self.data_source = list(data_source)
        self.batch_size = batch_size
        self.drop_remain = drop_remain
        self.shuffle = shuffle

        for i, d in enumerate(self.data_source):
            if isinstance(d, list):
                self.data_source[i] = np.array(d)

        self.data_size = len(self.data_source[0])
        if len(self.data_source)> 1:
            flag = np.all([len(src) == self.data_size for src in self.data_source])
            if not flag:
                raise ValueError("All elements in data_source should have same lengths")

        self.sampler = BatchSampler(self.data_size, self.batch_size, self.drop_remain, self.shuffle)
        self.iterator = iter(self.sampler)

        self.n=0

    def __next__(self):
        batch_idx = next(self.iterator)
        batch_data = tuple([data[batch_idx] for data in self.data_source])

        if len(batch_data) == 1:
            batch_data = batch_data[0]
        return batch_data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.sampler)