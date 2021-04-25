"""
Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009. 
https://arxiv.org/pdf/1205.2618
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel
from data.generators import PointwiseGenerator, PairwiseGenerator

class MF(BaseModel):
    def __init__(self, dataset, hparams, device):
        super(MF, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.hidden_dim = hparams['hidden_dim']
        self.pointwise = hparams['pointwise']
        self.loss_func = F.mse_loss if hparams['loss_func'] == 'mse' else F.binary_cross_entropy_with_logits

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.hidden_dim)
        
        self.device = device

        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def embeddings(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        return user_emb, item_emb
    
    def forward(self, user_ids, item_ids):
        user_emb, item_emb = self.embeddings(user_ids, item_ids)

        pred_rating = torch.sum(torch.mul(user_emb, item_emb), 1)
        return pred_rating

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data

        # num_training = len(user_ids)
        # num_batches = int(np.ceil(num_training / batch_size))
        if self.pointwise:
            batch_generator = PointwiseGenerator(
                train_matrix, return_rating=True, num_negatives=1, 
                batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        else:
            batch_generator = PairwiseGenerator(
                train_matrix, num_negatives=1, num_positives_per_user=1,
                batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        
        num_batches = len(batch_generator)
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            # batch_ratings: rating if pointwise, negtive items if pairwise
            for b, (batch_users, batch_pos, batch_ratings) in enumerate(batch_generator):
                self.optimizer.zero_grad()

                batch_loss = self.process_one_batch(batch_users, batch_pos, batch_ratings)
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss

                if exp_config.verbose and b % 50 == 0:
                    print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
            
            epoch_summary = {'loss': epoch_loss}
            
            # Evaluate if necessary
            if evaluator is not None and epoch >= exp_config.test_from and epoch % exp_config.test_step == 0:
                scores = evaluator.evaluate(self)
                epoch_summary.update(scores)
                
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)

                ## Check early stop
                if early_stop is not None:
                    is_update, should_stop = early_stop.step(scores, epoch)
                    if should_stop:
                        break
            else:
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)

        best_score = early_stop.best_score if early_stop is not None else scores
        return {'scores': best_score}
    
    def process_one_batch(self, users, items, ratings):
        pos_ratings = self.forward(users, items)
        if self.pointwise:
            loss = self.loss_func(pos_ratings, ratings)
        else:
            neg_ratings = self.forward(users, ratings)
            loss = -F.sigmoid(pos_ratings - neg_ratings).log().mean()
        
        return loss
    
    def predict_batch_users(self, user_ids):
        user_latent = self.user_embedding(user_ids)
        all_item_latent = self.item_embedding.weight.data
        return user_latent @ all_item_latent.T

    def predict(self, eval_users, eval_pos, test_batch_size):
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        with torch.no_grad():
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_eval_users:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                
                batch_users = eval_users[batch_idx]
                batch_users_torch = torch.LongTensor(batch_users).to(self.device)
                pred_matrix[batch_users] = self.predict_batch_users(batch_users_torch).detach().cpu().numpy()

        pred_matrix[eval_pos.nonzero()] = float('-inf')

        return pred_matrix

