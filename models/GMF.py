"""
He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference
on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.BaseModel import BaseModel

class GMF(BaseModel):
    def __init__(self, model_conf, num_user, num_item, device):
        super(GMF, self).__init__()
        self.emb_dim = model_conf.emb_dim
        self.user_embedding = nn.Embedding(num_user, self.emb_dim)
        self.item_embedding = nn.Embedding(num_item, self.emb_dim)
        self.weight = torch.randn(self.emb_dim, 1, device=device, dtype=torch.float, requires_grad=True)

        self.device = device
        self.to(device)

    def forward(self, user_ids, item_ids):
        user_latent = self.user_embedding(user_ids)
        item_latent = self.item_embedding(item_ids)

        out = torch.mul(user_latent, item_latent) @ self.weight
        out = F.sigmoid(out.squeeze())
        return out

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        # user, item, rating pairs
        user_ids, item_ids, ratings = dataset.generate_pointwise_data()

        num_training = len(user_ids)
        num_batches = int(np.ceil(num_training / batch_size))

        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_users = user_ids[batch_idx]
            batch_items = item_ids[batch_idx]
            batch_ratings = ratings[batch_idx]

            pred_ratings = self.forward(batch_users, batch_items)

            batch_loss = F.binary_cross_entropy(pred_ratings, batch_ratings, reduction='sum')
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def predict(self, dataset, test_batch_size):
        eval_users = []
        eval_items = []
        eval_candidates = dataset.eval_items
        for u in eval_candidates:
            eval_users += [u] * len(eval_candidates[u])
            eval_items += eval_candidates[u]
        eval_users = torch.LongTensor(eval_users).to(self.device)
        eval_items = torch.LongTensor(eval_items).to(self.device)
        pred_matrix = torch.full((dataset.num_users, dataset.num_items), float('-inf')).to(self.device)

        num_data = len(eval_items)
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        with torch.no_grad():
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                batch_users, batch_items = eval_users[batch_idx], eval_items[batch_idx]
                pred_matrix[batch_users, batch_items] = self.forward(batch_users, batch_items)

        return pred_matrix
