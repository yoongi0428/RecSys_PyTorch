import time
import torch
from utils.Table import Table

class Trainer:
    def __init__(self, dataset, model, evaluator, logger, conf):
        self.dataset = dataset
        self.model = model
        self.evaluator = evaluator
        self.logger = logger
        self.conf = conf

        self.num_epochs = conf.num_epochs
        self.lr = conf.learning_rate
        self.batch_size = conf.batch_size
        self.test_batch_size = conf.test_batch_size

        self.early_stop = conf.early_stop
        self.patience = conf.patience
        self.endure = 0

        self.best_epoch = -1
        self.best_score = None
        self.best_params = None

    def train(self):
        self.logger.info(self.conf)
        if len(list(self.model.parameters())) > 0:
            optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        else:
            optimizer = None
        score_table = Table(table_name='Scores')

        for epoch in range(1, self.num_epochs + 1):
            # train for an epoch
            epoch_start = time.time()
            loss = self.model.train_one_epoch(self.dataset, optimizer, self.batch_size, False)
            train_elapsed = time.time() - epoch_start

            # evaluate
            score = self.evaluate()
            epoch_elapsed = time.time() - epoch_start

            score_str = ' '.join(['%s=%.4f' % (m, score[m]) for m in score])

            self.logger.info('[Epoch %3d/%3d, epoch time: %.2f, train_time: %.2f] loss = %.4f, %s' % (
            epoch, self.num_epochs, epoch_elapsed, train_elapsed, loss, score_str))

            # update if ...
            standard = 'NDCG@100'
            if self.best_score is None or score[standard] >= self.best_score[standard]:
                self.best_epoch = epoch
                self.best_score = score
                self.best_params = self.model.parameters()
                self.endure = 0
            else:
                self.endure += 1
                if self.early_stop and self.endure >= self.patience:
                    print('Early Stop Triggered...')
                    break

        print('Training Finished.')
        score_table.add_row('Best at epoch %d' % self.best_epoch, self.best_score)
        self.logger.info(score_table.to_string())

    def evaluate(self):
        score = self.evaluator.evaluate(self.model, self.dataset, self.test_batch_size)
        return score