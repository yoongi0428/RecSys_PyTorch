"""
Dawen Liang et al., Variational Autoencoders for Collaborative Filtering. WWW 2018.
https://arxiv.org/pdf/1802.05814
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .BaseModel import BaseModel
from data.generators import MatrixGenerator

class MultVAE(BaseModel):
    def __init__(self, dataset, hparams, device):
        super(MultVAE, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        
        if isinstance(hparams['enc_dims'], str):
            hparams['enc_dims'] = eval(hparams['enc_dims'])
        self.enc_dims = [self.num_items] + list(hparams['enc_dims'])
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.total_anneal_steps = hparams['total_anneal_steps']
        self.anneal_cap = hparams['anneal_cap']

        self.dropout = hparams['dropout']

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, rating_matrix):
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

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        # user, item, rating pairs
        train_matrix = dataset.train_data

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / exp_config.batch_size))

        batch_generator = MatrixGenerator(train_matrix, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for b, batch_matrix in enumerate(batch_generator):
                self.optimizer.zero_grad()

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                pred_matrix, kl_loss = self.forward(batch_matrix)

                # cross_entropy
                ce_loss = F.binary_cross_entropy_with_logits(pred_matrix, batch_matrix, reduction='none').sum(1).mean()
                batch_loss = ce_loss + kl_loss * self.anneal
                batch_loss.backward()
                self.optimizer.step()

                self.update_count += 1

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

    def predict(self, eval_users, eval_pos, test_batch_size):
        with torch.no_grad():
            input_matrix = torch.FloatTensor(eval_pos.toarray()).to(self.device)
            preds = np.zeros(eval_pos.shape)

            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                test_batch_matrix = input_matrix[batch_idx]
                batch_pred_matrix = self.forward(test_batch_matrix)
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        
        preds[eval_pos.nonzero()] = float('-inf')

        return preds