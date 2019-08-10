import os
import numpy as np
from scipy.sparse import dok_matrix

def read_ml_data(filepath, sep):
    with open(filepath, 'rt', encoding='utf-8') as f:
        lines = f.readlines()

    user_id_to_idx = {}
    item_id_to_idx = {}
    data_dict = {}
    for line in lines:
        u, i, r, t = line.strip().split(sep)


        if u not in user_id_to_idx:
            u_idx = len(user_id_to_idx)
            user_id_to_idx[u] = u_idx
        else:
            u_idx = user_id_to_idx[u]

        if i not in item_id_to_idx:
            i_idx = len(item_id_to_idx)
            item_id_to_idx[i] = i_idx
        else:
            i_idx = item_id_to_idx[i]

        if u_idx in data_dict:
            # data_dict[u_idx].append((i_idx, float(r), int(t)))
            data_dict[u_idx].append((i_idx, 1.0, int(t)))
        else:
            # data_dict[u_idx] = [(i_idx, float(r), int(t))]
            data_dict[u_idx] = [(i_idx, 1.0, int(t))]

    return data_dict, user_id_to_idx, item_id_to_idx

# train_matrix, train_dict, test_dict, time_matrix
def split_loo(data_dict, num_users, num_items, random_split):
    train_matrix = dok_matrix((num_users, num_items))
    test_matrix = dok_matrix((num_users, num_items))
    time_matrix = dok_matrix((num_users, num_items))
    train_dict = {}
    test_dict = {}

    for u in data_dict:
        IRTs = sorted(data_dict[u], key=lambda x: x[2])
        u_num_ratings = len(IRTs)

        if u_num_ratings > 5:
            if random_split:
                test_idx = np.random.choice(u_num_ratings)
            else:
                test_idx = -1

            test_i, test_r, test_t = IRTs[test_idx]
            test_matrix[u, test_i] = test_r
            time_matrix[u, test_i] = test_t
            test_dict[u] = [test_i]
            IRTs.remove(IRTs[test_idx])

        u_item = []
        for train_i, train_r, train_t in IRTs:
            train_matrix[u, train_i] = train_r
            time_matrix[u, train_i] = train_t
            u_item.append(train_i)
        train_dict[u] = u_item

    return train_matrix, train_dict, test_matrix, test_dict, time_matrix

# TODO: Code refactoring
def split_holdout(data_dict, num_users, num_items, split_ratio, random_split):
    train_matrix = dok_matrix((num_users, num_items))
    test_matrix = dok_matrix((num_users, num_items))
    time_matrix = dok_matrix((num_users, num_items))
    train_dict = {u: [] for u in range(num_users)}
    test_dict = {u: [] for u in range(num_users)}

    for u in data_dict:
        IRTs = sorted(data_dict[u], key=lambda x: x[2])
        u_num_ratings = len(IRTs)

        num_train = int(u_num_ratings * split_ratio)
        num_test = u_num_ratings - num_train

        if u_num_ratings > 10 and num_test > 0:
            if random_split:
                test_idx = np.random.choice(u_num_ratings, num_test)
            else:
                test_idx = list(range(u_num_ratings - num_test, u_num_ratings))
        else:
            test_idx = []

        for i in range(u_num_ratings):
            item, rating, time = IRTs[i]
            if i in test_idx:
                if (u, item) in test_matrix:
                    print('stop')
                test_matrix[u, item] = rating
                time_matrix[u, item] = time
                test_dict[u].append(item)
            else:
                if (u, item) in train_matrix:
                    print('stop')
                train_matrix[u, item] = rating
                time_matrix[u, item] = time
                train_dict[u].append(item)

    return train_matrix, train_dict, test_matrix, test_dict, time_matrix