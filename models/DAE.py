"""
Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial
from models.BaseModel import BaseModel
from utils.Tools import activation_function

class DAE(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(DAE, self).__init__()
        self.hidden_dim = model_conf.hidden_dims # list of dims
        self.act = model_conf.act
        self.corruption_ratio = model_conf.corruption_ratio
        self.num_users = num_users
        self.num_items = num_items
        self.binomial = Binomial(total_count=1, probs=(1 - self.corruption_ratio))
        self.device = device

        dims = [self.num_items] + self.hidden_dim
        encoder_dict = OrderedDict()
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            encoder_dict['encoder_%d' % (i + 1)] = nn.Linear(dim_in, dim_out)
            encoder_dict['encoder_act_%d' % (i + 1)] = activation_function(act_name=self.act)

        self.encoder = nn.Sequential(encoder_dict)

        dims = self.hidden_dim[::-1] + [self.num_items]
        decoder_dict = OrderedDict()
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            decoder_dict['decoder_%d' % (i + 1)] = nn.Linear(dim_in, dim_out)
            decoder_dict['decoder_act_%d' % (i + 1)] = \
                activation_function(act_name=self.act) if (i + 1) < len(dims) - 1 \
                    else activation_function(act_name='sigmoid')

        self.decoder = nn.Sequential(decoder_dict)

        self.to(self.device)

    def forward(self, rating_matrix, is_train=True):
        # normalize
        user_degree = torch.norm(rating_matrix, 2, 1).view(-1, 1)   # user, 1
        item_degree = torch.norm(rating_matrix, 2, 0).view(1, -1)   # 1, item
        normalize = torch.sqrt(user_degree @ item_degree)
        zero_mask = normalize == 0
        normalize = torch.masked_fill(normalize, zero_mask.bool(), 1e-10)

        normalized_rating_matrix = rating_matrix / normalize

        # corruption
        if is_train:
            mask = self.generate_mask(normalized_rating_matrix.shape)
            normalized_rating_matrix *= mask

        # AE
        enc = self.encoder(normalized_rating_matrix)

        dec = self.decoder(enc)

        return dec

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
            pred_matrix = self.forward(batch_matrix)

            # cross_entropy
            batch_loss = batch_matrix * (pred_matrix + 1e-10).log() + (1 - batch_matrix) * (1 - pred_matrix + 1e-10).log()
            batch_loss = -torch.sum(batch_loss)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def generate_mask(self, mask_shape):
        return self.binomial.sample(mask_shape).to(self.device)

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
                batch_pred_matrix = self.forward(test_batch_matrix, is_train=False).detach().cpu.numpy()
                batch_pred_matrix.masked_fill(test_batch_matrix.bool(), float('-inf'))
                preds[batch_idx] = batch_pred_matrix
        return preds