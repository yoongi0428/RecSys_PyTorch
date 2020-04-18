"""
Jun Wang et al., Unifying user-based and item-based collaborative filtering approaches by similarity fusion. SIGIR 2006.
http://web4.cs.ucl.ac.uk/staff/jun.wang/papers/2006-sigir06-unifycf.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm import tqdm

from models.BaseModel import BaseModel

class ItemKNN(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(ItemKNN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.topk = model_conf.topk
        self.shrink = model_conf.shrink
        self.feature_weighting = model_conf.feature_weighting
        assert self.feature_weighting in ['tf-idf', 'bm25', 'none']

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        train_matrix = dataset.train_matrix
        if self.feature_weighting == 'tf-idf':
            train_matrix = self.TF_IDF(train_matrix.T).T
        elif self.feature_weighting == 'bm25':
            train_matrix = self.okapi_BM25(train_matrix.T).T
        train_matrix = train_matrix.tocsc()
        num_items = train_matrix.shape[1]

        start_col_local = 0
        end_col_local = num_items

        start_col_block = start_col_local

        this_block_size = 0
        block_size = 500

        sumOfSquared = np.array(train_matrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(sumOfSquared)

        values = []
        rows = []
        cols = []
        while start_col_block < end_col_local:
            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block-start_col_block

            # All data points for a given item
            # item_data: user, item blocks
            item_data = train_matrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            this_block_weights = train_matrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):
                # this_block_size: (item,)
                # similarity between 'one block item' and whole items
                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:,col_index_in_block]

                # columnIndex = item index
                # zero out self similarity
                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # cosine similarity
                # denominator = sqrt(l2_norm(x)) * sqrt(l2_norm(y))+ shrinkage + eps
                denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                relevant_items_partition = (-this_column_weights).argpartition(self.topk-1)[0:self.topk]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)

            start_col_block += block_size
        
        self.W_sparse = sp.csr_matrix((values, (rows, cols)),
                            shape=(num_items, num_items),
                            dtype=np.float32)
        # sp.save_npz(os.path.join(log_dir, 'best_model'), self.W_sparse)

        return 0.0

    def predict(self, eval_users, eval_pos, test_batch_size):
        # eval_pos_matrix
        preds = (eval_pos * self.W_sparse).toarray()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds

    def okapi_BM25(self, dataMatrix, K1=1.2, B=0.75):
        assert B>0 and B<1, "okapi_BM_25: B must be in (0,1)"
        assert K1>0,        "okapi_BM_25: K1 must be > 0"


        # Weighs each row of a sparse matrix by OkapiBM25 weighting
        # calculate idf per term (user)

        dataMatrix = sp.coo_matrix(dataMatrix)

        N = float(dataMatrix.shape[0])
        idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

        # calculate length_norm per document
        row_sums = np.ravel(dataMatrix.sum(axis=1))

        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length

        # weight matrix rows by bm25
        dataMatrix.data = dataMatrix.data * (K1 + 1.0) / (K1 * length_norm[dataMatrix.row] + dataMatrix.data) * idf[dataMatrix.col]

        return dataMatrix.tocsr()

    def TF_IDF(self, matrix):
        """
        Items are assumed to be on rows
        :param dataMatrix:
        :return:
        """

        # TFIDF each row of a sparse amtrix
        dataMatrix = sp.coo_matrix(matrix)
        N = float(dataMatrix.shape[0])

        # calculate IDF
        idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

        # apply TF-IDF adjustment
        dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

        return dataMatrix.tocsr()