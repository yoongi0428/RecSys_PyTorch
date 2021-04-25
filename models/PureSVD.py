import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn.functional as F
from models.BaseModel import BaseModel

class PureSVD(BaseModel):
    def __init__(self, dataset, hparams, device):
        super(PureSVD, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.num_factors = hparams['num_factors']
        self.device = device

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        # Solve EASE
        train_matrix = dataset.train_data.toarray()
        U, sigma, Vt = randomized_svd(train_matrix, n_components=self.num_factors, random_state=123)

        s_Vt = sp.diags(sigma) * Vt

        self.user_embedding = U
        self.item_embedding = s_Vt.T

        output = self.user_embedding @ self.item_embedding.T

        loss = F.binary_cross_entropy(torch.tensor(train_matrix), torch.tensor(output))
        
        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        
        return {'scores': scores, 'loss': loss}

    def predict_batch_users(self, user_ids):
        user_latent = self.user_embedding[user_ids]
        return user_latent @ self.item_embedding.T

    def predict(self, eval_users, eval_pos, test_batch_size):
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))

        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_eval_users:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
            
            batch_users = eval_users[batch_idx]
            pred_matrix[batch_users] = self.predict_batch_users(batch_users)

        pred_matrix[eval_pos.nonzero()] = float('-inf')

        return pred_matrix