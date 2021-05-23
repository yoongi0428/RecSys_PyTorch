import os
import math
import pandas as pd
import numpy as np

from typing import List, Dict, Union
from pathlib import Path

def split_into_tr_val_te(data:pd.DataFrame, generalization:str, num_valid_items:Union[int, float], num_test_items:Union[int, float], 
                        holdout_users:int, split_random:bool, user2id:Dict, item2id:Dict) -> Dict:
    prepro_data_dict = {}
    if generalization == 'weak':
        # Split data into train, valid, test
        train_data, test_data = split_input_target_by_users(data, test_ratio=num_valid_items, split_random=split_random)
        train_data, valid_data = split_input_target_by_users(train_data, test_ratio=num_test_items, split_random=split_random)

        prepro_data_dict['train'] = train_data
        prepro_data_dict['valid'] = valid_data
        prepro_data_dict['test'] = test_data
    else:
        user_ids = np.array(list(user2id.values()))
        num_users = len(user_ids)
        num_valid_users = num_test_users = int(num_users * holdout_users)
        num_train_users = num_users - num_valid_users - num_test_users

        perm = np.random.permutation(num_users)
        
        train_user_idx = perm[:num_train_users]
        valid_user_idx = perm[num_train_users: num_train_users + num_valid_users]
        test_user_idx = perm[num_train_users + num_valid_users: num_train_users + num_valid_users + num_test_users]

        train_users = user_ids[train_user_idx]
        valid_users = user_ids[valid_user_idx]
        test_users = user_ids[test_user_idx]
        
        # possible refactor candidate: 3N -> N
        train_data = data.loc[data.user.isin(train_users)]
        valid_data = data.loc[data.user.isin(valid_users)]
        test_data = data.loc[data.user.isin(test_users)]

        valid_input, valid_target = split_input_target_by_users(valid_data, test_ratio=num_valid_items, split_random=split_random)
        test_input, test_target = split_input_target_by_users(test_data, test_ratio=num_test_items, split_random=split_random)

        prepro_data_dict['train'] = train_data
        prepro_data_dict['valid_input'] = valid_input
        prepro_data_dict['valid_target'] = valid_target
        prepro_data_dict['test_input'] = test_input
        prepro_data_dict['test_target'] = test_target
    
    return prepro_data_dict

def split_input_target_by_users(df:pd.DataFrame, test_ratio:float=0.2, split_random:bool=True):
    df_group = df.groupby('user')
    train_list, test_list = [], []

    num_zero_train, num_zero_test = 0, 0
    for _, group in df_group:
        user = pd.unique(group.user)[0]
        num_items_user = len(group)

        if isinstance(test_ratio, float):
            num_test_items = int(math.ceil(test_ratio * num_items_user))
        else:
            num_test_items = test_ratio
        group = group.sort_values(by='timestamp')
        
        idx = np.ones(num_items_user, dtype='bool')
        if split_random:
            test_idx = np.random.choice(num_items_user, num_test_items, replace=False)
            idx[test_idx] = False
        else:    
            idx[-num_test_items:] = False

        
        if len(group[idx]) == 0:
            num_zero_train += 1
        else:
            train_list.append(group[idx])

        if len(group[np.logical_not(idx)]) == 0:
            num_zero_test += 1
        else:
            test_list.append(group[np.logical_not(idx)])
    
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    # TODO: warn zero train, test users

    return train_df, test_df