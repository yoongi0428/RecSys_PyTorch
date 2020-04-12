# """
# Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.
# """
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial
from models.BaseModel import BaseModel
from utils.Tools import activation_function

class MultVAE(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(MultVAE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        if isinstance(model_conf.enc_dims, str):
            model_conf.enc_dims = eval(model_conf.enc_dims)
        self.enc_dims = [self.num_items] + model_conf.enc_dims
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.total_anneal_steps = model_conf.total_anneal_steps
        self.anneal_cap = model_conf.anneal_cap

        self.dropout = model_conf.dropout
        self.reg = model_conf.reg

        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0

        self.device = device

        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        self.to(self.device)

    def forward(self, rating_matrix, is_train=True):
        # encoder
        h = F.dropout(F.normalize(rating_matrix), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        # sample
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]  # log sigmod^2  batch x 200
        std_q = torch.exp(0.5 * logvar_q)  # sigmod batch x 200

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if is_train:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        # user, item, rating pairs
        train_matrix = dataset.generate_rating_matrix()

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))

        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_matrix = train_matrix[batch_idx]

            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            pred_matrix, kl_loss = self.forward(batch_matrix)

            # cross_entropy
            ce_loss = -(F.log_softmax(pred_matrix, 1) * batch_matrix).sum(1).mean()

            batch_loss = ce_loss + kl_loss * self.anneal

            batch_loss.backward()
            optimizer.step()

            self.update_count += 1

            loss += batch_loss
            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def predict(self, dataset, test_batch_size):
        with torch.no_grad():
            rating_matrix = dataset.generate_rating_matrix()
            preds = np.zeros_like(rating_matrix)

            num_data = rating_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                test_batch_matrix = rating_matrix[batch_idx]
                batch_pred_matrix = self.forward(test_batch_matrix, is_train=False)
                batch_pred_matrix.masked_fill(test_batch_matrix.bool(), float('-inf'))
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        return preds