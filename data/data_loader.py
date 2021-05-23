import math
import pickle

import numpy as np
import scipy.sparse as sp

def load_data_and_info(data_file, info_file, cv_flag, split_type):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    with open(info_file, 'rb') as f:
        info_dict = pickle.load(f)
        
    user_id_dict = info_dict['user_id_dict']
    user_to_num_items = info_dict['user_to_num_items']
    item_id_dict = info_dict['item_id_dict']
    item_to_num_users = info_dict['item_to_num_users']

    num_users = data_dict['num_users']
    num_items = data_dict['num_items']

    if cv_flag:
        k_fold_files = data_dict['k_fold_files']
        return k_fold_files, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users
    elif split_type == 'hold-user-out':
        train_dict = data_dict['train']
        valid_input = data_dict['valid_input']
        valid_target = data_dict['valid_target']
        test_input = data_dict['test_input']
        test_target = data_dict['test_target']
        return train_dict, valid_input, valid_target, test_input, test_target, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users
    else:
        train_sp_matrix = data_dict['train']
        valid_sp_matrix = data_dict['valid']
        test_sp_matrix = data_dict['test']

        return train_sp_matrix, valid_sp_matrix, test_sp_matrix, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users
    
        

    

# def load_data_and_info(data_file, info_file, implicit=True):
#     with open(data_file, 'rb') as f:
#         data_dict = pickle.load(f)

#     with open(info_file, 'rb') as f:
#         info_dict = pickle.load(f)

#     train, valid, test = data_dict['train'], data_dict['valid'], data_dict['test']
#     user_id_dict, user_to_num_items, item_id_dict, item_to_num_users = info_dict['user_id_dict'], info_dict['user_to_num_items'], info_dict['item_id_dict'], info_dict['item_to_num_users']

#     for train_u in train:
#         IRTs_user = train[train_u]
#         irts = []
#         for irt in IRTs_user:
#             if implicit:
#                 irts.append((irt[0], 1))
#             else:
#                 irts.append((irt[0], irt[1]))
#         train[train_u] = irts
    
#     for valid_u in valid:
#         IRTs_user = valid[valid_u]
#         irts = []
#         for irt in IRTs_user:
#             if implicit:
#                 irts.append((irt[0], 1))
#             else:
#                 irts.append((irt[0], irt[1]))
#         valid[valid_u] = irts
    
#     for test_u in test:
#         IRTs_user = test[test_u]
#         irts = []
#         for irt in IRTs_user:
#             if implicit:
#                 irts.append((irt[0], 1))
#             else:
#                 irts.append((irt[0], irt[1]))
#         test[test_u] = irts

#     return train, valid, test, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users