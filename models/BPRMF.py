"""
Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009..
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel

class BPRMF(BaseModel):
    def __init__(self, model_conf, num_user, num_item, device):
        super(BPRMF, self).__init__()
        self.hidden_dim = model_conf.hidden_dim
        self.user_embedding = nn.Embedding(num_user, self.hidden_dim)
        self.item_embedding = nn.Embedding(num_item, self.hidden_dim)
        self.device = device

        self.to(device)

    def forward(self, user_ids, item_ids):
        user_latent = self.user_embedding(user_ids)
        item_latent = self.item_embedding(item_ids)

        pred_rating = torch.sum(torch.mul(user_latent, item_latent), 1)
        return pred_rating

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        # user, item, rating pairs
        # user_ids, item_ids, neg_ids = dataset.generate_pairwise_data()
        user_ids, item_ids, neg_ids = dataset.generate_pairwise_data_from_matrix(dataset.train_matrix, 1)

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
            batch_negs = neg_ids[batch_idx]

            pos_ratings = self.forward(batch_users, batch_items)
            neg_ratings = self.forward(batch_users, batch_negs)

            log_sigmoid_diff = F.sigmoid(pos_ratings - neg_ratings).log()
            batch_loss = -torch.sum(log_sigmoid_diff)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def predict(self, dataset, test_batch_size):
        # Temp
        # pred_matrix = F.sigmoid(self.user_embedding.weight @ torch.transpose(self.item_embedding.weight, 0, 1))
        eval_users = []
        eval_items = []
        eval_candidates = dataset.eval_items
        for u in eval_candidates:
            eval_users += [u] * len(eval_candidates[u])
            eval_items += eval_candidates[u]
        eval_users = torch.LongTensor(eval_users).to(self.device)
        eval_items = torch.LongTensor(eval_items).to(self.device)
        
        pred_matrix = np.full((dataset.num_users, dataset.num_items), float('-inf'))

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
                pred_matrix[batch_users, batch_items] = self.forward(batch_users, batch_items).detach().cpu().numpy()

        return pred_matrix

