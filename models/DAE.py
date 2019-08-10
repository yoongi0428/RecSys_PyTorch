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
    def __init__(self, model_conf, num_user, num_item, device):
        super(DAE, self).__init__()
        self.hidden_dim = model_conf.hidden_dims # list of dims
        self.act = model_conf.act
        self.corruption_ratio = model_conf.corruption_ratio
        self.train_rkd = model_conf.train_rkd
        self.binomial = Binomial(total_count=1, probs=(1 - self.corruption_ratio))
        self.device = device

        dims = [num_item] + self.hidden_dim
        encoder_dict = OrderedDict()
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            encoder_dict['encoder_%d' % (i + 1)] = nn.Linear(dim_in, dim_out)
            encoder_dict['encoder_act_%d' % (i + 1)] = activation_function(act_name=self.act)
        self.encoder = nn.Sequential(encoder_dict)

        dims = self.hidden_dim[::-1] + [num_item]
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
        normalize = torch.masked_fill(normalize, zero_mask, 1e-10)

        normalized_rating_matrix = rating_matrix / normalize

        # corruption
        if is_train:
            mask = self.generate_mask(normalized_rating_matrix.shape)
            normalized_rating_matrix *= mask

        # AE
        enc = self.encoder(normalized_rating_matrix)

        dec = self.decoder(enc)

        if self.train_rkd and is_train:
            # rkd_loss = self.RKD_angle(normalized_rating_matrix, enc)
            rkd_loss = self.RKD_distance(normalized_rating_matrix, enc)
            return dec, rkd_loss
        else:
            return dec

    def RKD_angle(self, rating_matrix, enc):
        with torch.no_grad():
            td = (rating_matrix.unsqueeze(0) - rating_matrix.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (enc.unsqueeze(0) - enc.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        rkd_loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return rkd_loss

    def RKD_distance(self, rating_matrix, enc):
        def pdist(e, squared=False, eps=1e-12):
            e_square = e.pow(2).sum(dim=1)
            prod = e @ e.t()
            res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

            if not squared:
                res = res.sqrt()

            res = res.clone()
            res[range(len(e)), range(len(e))] = 0
            return res

        with torch.no_grad():
            t_d = pdist(rating_matrix, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(enc, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        rkd_loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return rkd_loss

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        # user, item, rating pairs
        # train_matrix = dataset.train_matrix
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
            if self.train_rkd:
                pred_matrix, rkd_loss = pred_matrix

            # cross_entropy
            batch_loss = batch_matrix * (pred_matrix + 1e-10).log() + (1 - batch_matrix) * (1 - pred_matrix + 1e-10).log()
            batch_loss = -torch.sum(batch_loss)
            if self.train_rkd:
                batch_loss += 0.5 * rkd_loss
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def generate_mask(self, mask_shape):
        return self.binomial.sample(mask_shape).to(self.device)

    def predict(self, dataset):
        # Temp
        train_matrix = dataset.generate_rating_matrix()
        pred_matrix = self.forward(train_matrix, is_train=False)
        return pred_matrix

