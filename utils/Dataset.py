import os
import numpy as np
import torch
from utils.Data_utils import read_ml_data, split_holdout, split_loo

class Dataset:
    def __init__(self, data_dir, data_name, split_type, train_ratio, split_random, device):
        self.split_type = split_type
        self.split_random = split_random
        self.train_ratio = train_ratio
        self.device = device

        if data_name == 'ml-100k':
            sep = '\t'
            filename = 'u.data'
            self.num_users, self.num_items = 943, 1682
        elif data_name == 'ml-1m':
            sep = '::'
            filename = 'ratings.dat'
            self.num_users, self.num_items = 6040, 3952
        else:
            raise NotImplementedError('Choose correct dataset: {ml-100k, ml-1m}')

        print('Read movielens data from %s' % filename)
        # load movie lens...
        self.data_dict, self.user_id_to_idx, self.item_id_to_idx = \
            read_ml_data(os.path.join(data_dir, data_name, filename), sep)

        print('Split data into train & test ... (%s)' % self.split_type)
        # split data into {holdout, loo}
        if self.split_type == 'holdout':
            self.train_matrix, self.train_dict, self.test_matrix, self.test_dict, self.time_matrix = \
                split_holdout(self.data_dict, self.num_users, self.num_items, self.train_ratio, self.split_random)
        elif self.split_type == 'loo':
            self.train_matrix, self.train_dict, self.test_matrix, self.test_dict, self.time_matrix = \
                split_loo(self.data_dict, self.num_users, self.num_items, self.split_random)
        else:
            raise NotImplementedError('Choose correct data split type: {holdout, loo}')

        print('Dataset prepared')

        self.neg_items = None

    # TODO: Code refatoring --> predefined neg items?
    def generate_pairwise_data(self):
        # generate item, pos, neg triplets
        users = []
        pos = []
        neg = []

        for u, pos_i in self.train_matrix.keys():
            neg_i = np.random.choice(self.num_items)
            while neg_i in self.train_dict[u]:
                neg_i = np.random.choice(self.num_items)

            users.append(u)
            pos.append(pos_i)
            neg.append(neg_i)
        users = torch.LongTensor(users).to(self.device)
        pos = torch.LongTensor(pos).to(self.device)
        neg = torch.LongTensor(neg).to(self.device)
        return users, pos, neg

    def generate_pointwise_data(self):
        pass

    def generate_rating_matrix(self):
        return torch.FloatTensor(self.train_matrix.toarray()).to(self.device)

    def __str__(self):
        # return string representation of 'Dataset' class
        # print(Dataset) or str(Dataset)
        ret = '======== [Dataset] ========\n'
        # ret += 'Train file: %s\n' % self.train_file
        # ret += 'Test file : %s\n' % self.test_file
        ret += 'Number of Users : %d\n' % self.num_users
        ret += 'Number of items : %d\n' % self.num_items
        ret += 'Split type: '
        ret += 'Leave-One-Out\n' if self.split_type == 'loo' else 'Holdout\n'
        if self.split_type == 'ratio':
            ret += 'Split ratio: %s\n' % str(self.train_ratio)
        ret += '\n'
        return ret