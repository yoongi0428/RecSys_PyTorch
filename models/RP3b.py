"""
Bibek Paudel et al., Updatable, accurate, diverse, and scalablerecommendations for interactive applications. TiiS 2017.
https://www.zora.uzh.ch/id/eprint/131338/1/TiiS_2016.pdf

Main model codes from https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
"""
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm

from models.BaseModel import BaseModel

class RP3b(BaseModel):
    def __init__(self, dataset, hparams, device):
        super(RP3b, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.topk = hparams['topk']
        self.alpha = hparams['alpha']
        self.beta = hparams['beta']

    def fit_rp3b(self, train_matrix, block_dim=200):
        num_items = train_matrix.shape[1]

        Pui = normalize(train_matrix, norm='l1', axis=1)
        
        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = train_matrix.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(train_matrix.shape[1])
        nonZeroMask = X_bool_sum != 0.0
        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        if self.alpha != 1:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0
        
        item_blocks = range(0, num_items, block_dim)
        tqdm_iterator = tqdm(item_blocks, desc='# items blocks covered', total=len(item_blocks))
        for cur_items_start_idx in tqdm_iterator:

            if cur_items_start_idx + block_dim > num_items:
                block_dim = num_items - cur_items_start_idx

            # second * third transition matrix: # of ditinct path from item to item
            # block_dim x item
            Piui = Piu[cur_items_start_idx:cur_items_start_idx + block_dim, :] * Pui
            Piui = Piui.toarray()

            for row_in_block in range(block_dim):
                # Delete self connection
                row_data = np.multiply( Piui[row_in_block, :], degree)
                row_data[cur_items_start_idx + row_in_block] = 0

                # Top-k items
                best = row_data.argsort()[::-1][:self.topk]

                # add non-zero top-k path only (efficient)
                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = cur_items_start_idx + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))
    
    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        self.fit_rp3b(train_matrix.tocsc())

        output = train_matrix @ self.W_sparse

        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output.toarray()))

        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        
        return {'scores': scores, 'loss': loss}

    def predict(self, eval_users, eval_pos, test_batch_size):
        # eval_pos_matrix
        preds = (eval_pos * self.W_sparse).toarray()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds