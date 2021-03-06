"""
Colin Cooper et al., Random Walks in Recommender Systems: Exact Computation and Simulations. WWW 2014.
http://wwwconference.org/proceedings/www2014/companion/p811.pdf

Main model codes from https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm

from models.BaseModel import BaseModel

class P3a(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(P3a, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.topk = model_conf.topk
        self.alpha = model_conf.alpha

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        train_matrix = dataset.train_matrix.tocsc()
        num_items = train_matrix.shape[1]

        # (user, item), 1 / # user
        Pui = normalize(train_matrix, norm='l1', axis=1)
        
        X_bool = train_matrix.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # (item, uesr), 1 / # item
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        if self.alpha != 1:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        block_dim = 200
        # d_t = Piu

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
                row_data = Piui[row_in_block, :]
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
        # sp.save_npz(os.path.join(log_dir, 'best_model'), self.W_sparse)

        return 0.0

    def predict(self, eval_users, eval_pos, test_batch_size):
        # eval_pos_matrix
        preds = (eval_pos * self.W_sparse).toarray()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds