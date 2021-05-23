import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from typing import List, Dict, Union, Optional
from pathlib import Path

from .preprocess import split_into_tr_val_te
from utils.types import df_to_sparse

class UIRTDataset(object):
    def __init__(self, data_path:str, dataname:Optional[str]=None, separator:str=',', binarize_threshold:Union[int, float]=0.0, implicit:bool=True, 
                        min_item_per_user:int=0, min_user_per_item:int=0, protocol:str='holdout', generalization:str='weak', 
                        holdout_users:Union[int, float]=0.1, valid_ratio:Union[int, float]=0.1, test_ratio:Union[int, float]=0.2, 
                        leave_k:int=1, split_random:bool=True, cache_dir:str='cache', seed:int=1234):
        self.data_path = Path(data_path)
        self.base_dir = self.data_path.parent
        self.dataname = dataname if dataname is not None else self.base_dir.name

        self.separator = separator
        self.binarize_threshold = binarize_threshold
        self.implicit = implicit
        self.min_item_per_user = min_item_per_user
        self.min_user_per_item = min_user_per_item

        self.protocol = protocol
        self.generalization = generalization
        self.holdout_users = holdout_users

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.leave_k = leave_k
        self.split_random = split_random

        self.seed = seed
        self.cache_dir = cache_dir

        self._prepro_cache_dir = None
        
        self._set_preprocessed_cache_dir()
        self._prepro_file_dict, self._user2id_file, self._item2id_file = self._ensure_preprocessed()
        self._load_preproessed_data()

    def _load_preproessed_data(self) -> None:
        def transform(df):
            if self.implicit:
                if self.binarize_threshold > 0:
                    df = df[df['rating'] >= self.threshold]
                df.rating = np.ones(len(df))
            return df
        
        print('Load preprocessed data...')

        self.user2id = self._load_id_map(self._user2id_file)
        self.item2id = self._load_id_map(self._item2id_file)
        self.num_users, self.num_items = len(self.user2id), len(self.item2id)

        names=['user', 'item', 'rating', 'timestamp']
        dtype={'user': int, 'item': int, 'rating': float, 'timestamp': float}
        
        if self.generalization == 'weak':
            train_df = transform(pd.read_csv(self._prepro_file_dict['train'], sep=',', names=names, dtype=dtype))
            valid_df = transform(pd.read_csv(self._prepro_file_dict['valid'], sep=',', names=names, dtype=dtype))
            test_df = transform(pd.read_csv(self._prepro_file_dict['test'], sep=',', names=names, dtype=dtype))

            self.train_users = self.valid_users = self.test_users = list(pd.unique(train_df.user))

            self.train_data = df_to_sparse(train_df, shape=(self.num_users, self.num_items))
            self.valid_target = df_to_sparse(valid_df, shape=(self.num_users, self.num_items))
            self.test_target = df_to_sparse(test_df, shape=(self.num_users, self.num_items))
        else:
            train_df = transform(pd.read_csv(self._prepro_file_dict['train'], sep=',', names=names, dtype=dtype))
            valid_input_df = transform(pd.read_csv(self._prepro_file_dict['valid_input'], sep=',', names=names, dtype=dtype))
            valid_target_df = transform(pd.read_csv(self._prepro_file_dict['valid_target'], sep=',', names=names, dtype=dtype))
            test_input_df = transform(pd.read_csv(self._prepro_file_dict['test_input'], sep=',', names=names, dtype=dtype))
            test_target_df = transform(pd.read_csv(self._prepro_file_dict['test_target'], sep=',', names=names, dtype=dtype))

            self.train_users = list(pd.unique(train_df.user))
            self.valid_users = list(pd.unique(valid_input_df.user))
            self.test_users = list(pd.unique(test_input_df.user))

            valid_user_ids = [self.user2id[u] for u in self.valid_users]
            test_user_ids = [self.user2id[u] for u in self.test_users]

            self.train_data = df_to_sparse(train_df, shape=(self.num_users, self.num_items))
            self.valid_input = df_to_sparse(valid_input_df, shape=(self.num_users, self.num_items))[valid_user_ids]
            self.valid_target = df_to_sparse(valid_target_df, shape=(self.num_users, self.num_items))[valid_user_ids]
            self.test_input = df_to_sparse(test_input_df, shape=(self.num_users, self.num_items))[test_user_ids]
            self.test_target = df_to_sparse(test_target_df, shape=(self.num_users, self.num_items))[test_user_ids]
    
    def _ensure_preprocessed(self) -> None:
        if self.generalization == 'weak':
            prepro_dict = {
                'train': self._prepro_cache_dir / 'train.csv',
                'valid': self._prepro_cache_dir / 'valid.csv',
                'test': self._prepro_cache_dir / 'test.csv'
            }
        else:
            prepro_dict = {
                'train': self._prepro_cache_dir / 'train.csv',
                'valid_input': self._prepro_cache_dir / 'valid_input.csv',
                'valid_target': self._prepro_cache_dir / 'valid_target.csv',
                'test_input': self._prepro_cache_dir / 'test_input.csv',
                'test_target': self._prepro_cache_dir / 'test_target.csv'
            }
        user2id_file = self._prepro_cache_dir / 'user_map'
        item2id_file = self._prepro_cache_dir / 'item_map'
        files_to_check = list(prepro_dict.values()) + [user2id_file, item2id_file]
        
        if self._check_preprocssed(files_to_check):
            print('Load from preprocssed')
        else:
            print('Preprocess raw data...')
            raw_data = pd.read_csv(self.data_path, sep=self.separator, 
                                names=['user', 'item', 'rating', 'timestamp'],
                                dtype={'user': int, 'item': int, 'rating': float, 'timestamp': float},
                                engine='python')
            
            # TODO: handle UI, UIR, UIT via NaN
            sample_row = raw_data.iloc[0,:]
            if pd.isna(sample_row.rating):
                raw_data.rating = np.ones(len(raw_data))
            if pd.isna(sample_row.timestamp):
                raw_data.timestamp = np.ones(len(raw_data))           
            
            # user item id map
            raw_num_users = len(pd.unique(raw_data.user))
            raw_num_items = len(pd.unique(raw_data.item))

            # Filter users
            num_items_by_user = raw_data.groupby('user', as_index=False).size()
            num_items_by_user = num_items_by_user.set_index('user')
            user_filter_idx = raw_data['user'].isin(num_items_by_user.index[num_items_by_user['size'] >= self.min_item_per_user])
            raw_data = raw_data[user_filter_idx]
            num_items_by_user = raw_data.groupby('user', as_index=False).size()
            num_items_by_user = num_items_by_user.set_index('user')
            
            num_users = len(pd.unique(raw_data.user))
            print('# user after filter (min %d items): %d' % (self.min_item_per_user, num_users))

            # Filter items
            num_users_by_item = raw_data.groupby('item', as_index=False).size()
            num_users_by_item = num_users_by_item.set_index('item')
            item_filter_idx = raw_data['item'].isin(num_users_by_item.index[num_users_by_item['size'] >= self.min_user_per_item])
            raw_data = raw_data[item_filter_idx]
            num_users_by_item = raw_data.groupby('item', as_index=False).size()
            num_users_by_item = num_users_by_item.set_index('item')

            num_items = len(pd.unique(raw_data.item))
            print('# item after filter (min %d users): %d' % (self.min_user_per_item, num_items))

            # Build user old2new id map
            # user_frame = num_items_by_user.to_frame()
            num_items_by_user.columns = ['item_cnt']
            raw_users = list(num_items_by_user.index)
            user2id = {u: uid for uid, u in enumerate(raw_users)}

            # Build item old2new id map
            # item_frame = num_users_by_item.to_frame()
            num_users_by_item.columns = ['user_cnt']
            raw_items = list(num_users_by_item.index)
            item2id = {i: iid for iid, i in enumerate(raw_items)}
            
            # Convert to new id
            raw_data.user = [user2id[u] for u in  raw_data.user.tolist()]
            raw_data.item = [item2id[i] for i in  raw_data.item.tolist()]

            # preprocess and save
            if self.protocol == 'leave_one_out':
                prepro_data_dict = split_into_tr_val_te(
                    data=raw_data, 
                    generalization=self.generalization,
                    num_valid_items=self.leave_k,
                    num_test_items=self.leave_k,
                    holdout_users=self.holdout_users,
                    split_random=self.split_random,
                    user2id=user2id,
                    item2id=item2id)
            elif self.protocol == 'holdout':
                prepro_data_dict = split_into_tr_val_te(
                    data=raw_data, 
                    generalization=self.generalization,
                    num_valid_items=self.valid_ratio,
                    num_test_items=self.test_ratio,
                    holdout_users=self.holdout_users,
                    split_random=self.split_random,
                    user2id=user2id,
                    item2id=item2id)
            else:
                raise ValueError(f'{self.protocol} is not a valid protocol.')
            
            for filename, filepath in prepro_dict.items():
                prepro_data_dict[filename].to_csv(filepath, index=False, header=False)
            
            self._save_id_map(user2id, user2id_file)
            self._save_id_map(item2id, item2id_file)

        return prepro_dict, user2id_file, item2id_file

    def _check_preprocssed(self, files_to_check: List) -> bool:
        if not self._prepro_cache_dir.exists():
            return False
        
        for filepath in files_to_check:
            if not filepath.exists():
                return False
        
        return True
    
    def _set_preprocessed_cache_dir(self) -> None:
        if self._prepro_cache_dir is None:
            random_or_not = 'random' if self.split_random else 'time'
            if self.protocol == 'leave_one_out':
                protocol_name = f'loo_{self.leave_k}_{self.generalization}_{random_or_not}_minUI_{self.min_item_per_user}_{self.min_user_per_item}_seed{self.seed}/'
            elif self.protocol == 'holdout':
                random_or_not = 'random' if self.split_random else 'time'
                valid_ratio_str = '%.2f' % self.valid_ratio
                test_ratio_str = '%.2f' % self.test_ratio
                protocol_name = f'holdout_{valid_ratio_str}_{test_ratio_str}_{self.generalization}_{random_or_not}_minUI_{self.min_item_per_user}_{self.min_user_per_item}_seed{self.seed}/'
            else:
                raise ValueError(f'Incorrect protocol passed ({self.protocol}). Choose between leave_one_out, holdout, hold_user_out')
            
            self._prepro_cache_dir = Path(os.path.join(self.base_dir, self.cache_dir, protocol_name))
        
        if not self._prepro_cache_dir.exists():
            os.makedirs(self._prepro_cache_dir)
    
    def _load_id_map(self, id_map_file: Path) -> Dict:
        old2new = {}
        with open(id_map_file, 'rt') as f:
            for line in f.readlines():
                u, uid = line.strip().split(', ')
                old2new[int(u)] = int(uid)
        return old2new
    
    def _save_id_map(self, id_map: Dict, id_map_file: Path) -> None:
        # Write user/item id map into files
        with open(id_map_file, 'wt') as f:
            for u, uid in id_map.items():
                f.write('%d, %d\n' % (u, uid))

    @property
    def valid_input(self) -> sp.csr_matrix:
        if self.generalization == 'weak':
            return self.train_data
        else:
            return self.valid_input

    @property
    def test_input(self) -> sp.csr_matrix:
        if self.generalization == 'weak':
            return self.train_data + self.valid_target
        else:
            return self.test_input

    @property
    def num_train_users(self) -> int:
        return len(self.train_users)
    
    @property
    def num_valid_users(self) -> int:
        return len(self.valid_users)

    @property
    def num_test_users(self) -> int:
        return len(self.test_users)