"""
Harald Steck, Embarrassingly Shallow Autoencoders for Sparse Data. WWW 2019.
https://arxiv.org/pdf/1905.03375
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel

class EASE(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(EASE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.reg = model_conf.reg

        self.device = device
        self.to(self.device)

    def forward(self, rating_matrix):
        G = (rating_matrix.T @ rating_matrix).toarray()

        diag = np.diag_indices(G.shape[0])
        G[diag] += self.reg
        P = np.linalg.inv(G)
        self.enc_w = P / (-np.diag(P))
        self.enc_w[diag] = 0

        # Calculate the output matrix for prediction
        output = rating_matrix @ self.enc_w

        return output

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        self.train()
        
        # Solve EASE
        train_matrix = dataset.train_matrix
        output = self.forward(train_matrix)

        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output))
        
        return loss

    def generate_mask(self, mask_shape):
        return self.binomial.sample(mask_shape).to(self.device)

    def predict(self, eval_users, eval_pos, test_batch_size):
        input_matrix = eval_pos.toarray()
        preds = np.zeros_like(input_matrix)

        num_data = input_matrix.shape[0]
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_data:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
            test_batch_matrix = input_matrix[batch_idx]
            batch_pred_matrix = (test_batch_matrix @ self.enc_w)
            preds[batch_idx] = batch_pred_matrix
            
        preds[eval_pos.nonzero()] = float('-inf')

        return preds