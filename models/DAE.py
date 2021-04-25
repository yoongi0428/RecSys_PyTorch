"""
Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.
https://alicezheng.org/papers/wsdm16-cdae.pdf
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel
from data.generators import MatrixGenerator

class DAE(BaseModel):
    def __init__(self, dataset, hparams, device):
        super(DAE, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.hidden_dim = hparams['hidden_dim']
        self.act = hparams['act']
        self.corruption_ratio = hparams['corruption_ratio']
        
        self.device = device

        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)

        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, rating_matrix):
        # corruption
        rating_matrix = F.dropout(rating_matrix, self.corruption_ratio, training=self.training)

        # AE
        enc = torch.tanh(self.encoder(rating_matrix))

        dec = self.decoder(enc)

        return torch.sigmoid(dec)

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
                pred_matrix = self.forward(batch_matrix)

                # cross_entropy
                batch_loss = F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='none').sum(1).mean()
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
                preds[batch_idx] += batch_pred_matrix.detach().cpu().numpy()
        
        preds[eval_pos.nonzero()] = float('-inf')
        
        return preds