"""
Xia Ning et al., SLIM: Sparse Linear Methods for Top-N Recommender Systems. ICDM 2011.
http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.linear_model import ElasticNet

from models.BaseModel import BaseModel

class SLIM(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(SLIM, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.l1_reg = model_conf.l1_reg
        self.l2_reg = model_conf.l2_reg
        self.topk = model_conf.topk

        self.device = device

        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha
        self.slim = ElasticNet(alpha=alpha,
                                l1_ratio=l1_ratio,
                                positive=True,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=300,
                                tol=1e-3)

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        train_matrix = dataset.train_matrix.tocsc()
        num_items = train_matrix.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0
        tqdm_iterator = tqdm(range(num_items), desc='# items covered', total=num_items)
        for item in tqdm_iterator:
            y = train_matrix[:, item].toarray()

            # set the j-th column of X to zero
            start_pos = train_matrix.indptr[item]
            end_pos = train_matrix.indptr[item + 1]

            current_item_data_backup = train_matrix.data[start_pos: end_pos].copy()
            train_matrix.data[start_pos: end_pos] = 0.0

            self.slim.fit(train_matrix, y)

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            nonzero_model_coef_index = self.slim.sparse_coef_.indices
            nonzero_model_coef_value = self.slim.sparse_coef_.data


            local_topK = min(len(nonzero_model_coef_value)-1, self.topk)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = item
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1
            
            train_matrix.data[start_pos:end_pos] = current_item_data_backup
        
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(num_items, num_items), dtype=np.float32)

        return 0.0

    def predict(self, eval_users, eval_pos, test_batch_size):
        # eval_pos_matrix
        preds = (eval_pos * self.W_sparse).toarray()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds