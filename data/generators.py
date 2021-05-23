import torch
import numpy as np

class MatrixGenerator:
    def __init__(self, input_matrix, return_index=False, batch_size=32, shuffle=True, 
                        matrix_as_numpy=False, index_as_numpy=False, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self.return_index = return_index
        self._num_data = self.input_matrix.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.matrix_as_numpy = matrix_as_numpy
        self.index_as_numpy = index_as_numpy
        self.device = device
    
    def __len__(self):
        return int(np.ceil(self._num_data / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self._num_data)
        else:
            perm = np.arange(self._num_data, dtype=np.int32)

        for b, st in enumerate(range(0, self._num_data, self.batch_size)):
            ed = min(st + self.batch_size, self._num_data)
            batch_idx = perm[st:ed]
            
            if self.matrix_as_numpy:
                batch_input = self.input_matrix[batch_idx].toarray()
            else:
                batch_input = torch.tensor(self.input_matrix[batch_idx].toarray(),
                                            dtype=torch.float32, device=self.device)
            
            if self.return_index:
                if not self.index_as_numpy:
                    batch_idx = torch.tensor(batch_idx, dtype=torch.int64, device=self.device)
                yield batch_input, batch_idx
            else:
                yield batch_input

class PointwiseGenerator:
    def __init__(self, input_matrix, return_rating=True, as_numpy=False, negative_sample=True, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self.return_rating = return_rating
        self.negative_sample = negative_sample
        self.num_negatives = num_negatives
        self.as_numpy = as_numpy
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self._construct()

    def _construct(self):
        num_users, num_items = self.input_matrix.shape

        self.users = []
        self.items = []
        self.ratings = []
        for u in range(num_users):
            u_items = self.input_matrix[u].indices
            u_ratings = self.input_matrix[u].data

            self.users += [u] * len(u_items)
            self.items += u_items.tolist()
            if self.return_rating:
                self.ratings += u_ratings.tolist()
        
        self.users = np.array(self.users)
        self.items = np.array(self.items)
        self.ratings = np.array(self.ratings)

        self._num_data = len(self.users)
    
    def sample_negatives(self, users):
        num_users, num_items = self.input_matrix.shape
        users = []
        negatives = []

        for u in range(num_users):
            u_pos_items = self.input_matrix[u].indices
            
            prob = np.ones(num_items)
            prob[u_pos_items] = 0.0
            prob = prob / sum(prob)

            neg_samples = np.random.choice(num_items, size=self.num_negatives, replace=False, p=prob)

            users += [u] * len(neg_samples)
            negatives += neg_samples.tolist()
        
        users = np.array(users)
        negatives = np.array(negatives)
        ratings = np.zeros_like(users)

        return users, negatives, ratings

    def __len__(self):  
        return int(np.ceil(self._num_data / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self._num_data)
        else:
            perm = np.arange(self._num_data)

        for b, st in enumerate(range(0, self._num_data, self.batch_size)):
            ed = min(st + self.batch_size, self._num_data)
            batch_idx = perm[st:ed]

            batch_users = self.users[batch_idx]
            batch_items = self.items[batch_idx]
            if self.return_rating:
                batch_ratings = self.ratings[batch_idx]

                if self.negative_sample and self.num_negatives > 0:
                    neg_users, neg_items, neg_ratings = self.sample_negatives(batch_users)
                    
                    batch_users = np.concatenate((batch_users, neg_users))
                    batch_items = np.concatenate((batch_items, neg_items))
                    batch_ratings = np.concatenate((batch_ratings, neg_ratings))
            
                if not self.as_numpy:
                    batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                    batch_items = torch.tensor(batch_items, dtype=torch.long, device=self.device)
                    batch_ratings = torch.tensor(batch_ratings, dtype=torch.float32, device=self.device)
                yield batch_users, batch_items, batch_ratings
            else:
                if not self.as_numpy:
                    batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                    batch_items = torch.tensor(batch_items, dtype=torch.long, device=self.device)
                yield batch_users, batch_items

class PairwiseGenerator:
    def __init__(self, input_matrix, as_numpy=False, num_positives_per_user=-1, num_negatives=1, batch_size=32, shuffle=True, device=None):
        self.input_matrix = input_matrix
        self.num_positives_per_user = num_positives_per_user
        self.num_negatives = num_negatives
        self.as_numpy = as_numpy
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self._construct()

    def _construct(self):
        num_users, num_items = self.input_matrix.shape

        # self.users = []
        # self.items = []
        # for u in range(num_users):
        #     u_items = self.input_matrix[u].indices

        #     self.users += [u] * len(u_items)
        #     self.items += u_items.tolist()
        
        # self.users = np.array(self.users)
        # self.items = np.array(self.items)

        self._data = self.sample_negatives()
        self._num_data = len(self._data[0])
    
    def sample_negatives(self):
        num_users, num_items = self.input_matrix.shape
        users = []
        positives = []
        negatives = []

        for u in range(num_users):
            u_pos_items = self.input_matrix[u].indices
            num_pos_user = len(u_pos_items)

            prob = np.ones(num_items)
            prob[u_pos_items] = 0.0
            prob = prob / sum(prob)

            if self.num_positives_per_user > 0 and self.num_positives_per_user < num_pos_user:
                # subsample
                pos_sampled = np.random.choice(num_items, size=self.num_positives_per_user, replace=False)
                neg_sampled = np.random.choice(num_items, size=self.num_positives_per_user, replace=False, p=prob)
            else:
                # sample all
                pos_sampled = u_pos_items
                neg_sampled = np.random.choice(num_items, size=num_pos_user, replace=False, p=prob)
            
            assert len(pos_sampled) == len(neg_sampled)

            users += [u] * len(neg_sampled)
            positives += pos_sampled.tolist()
            negatives += neg_sampled.tolist()
        
        users = np.array(users)
        positives = np.array(positives)
        negatives = np.array(negatives)

        return users, positives, negatives

    def __len__(self):  
        return int(np.ceil(self._num_data / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self._num_data)
        else:
            perm = np.arange(self._num_data)

        for b, st in enumerate(range(0, self._num_data, self.batch_size)):
            ed = min(st + self.batch_size, self._num_data)
            batch_idx = perm[st:ed]

            batch_users = self._data[0][batch_idx]
            batch_pos = self._data[1][batch_idx]
            batch_neg = self._data[2][batch_idx]

            if not self.as_numpy:
                batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                batch_pos = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
                batch_neg = torch.tensor(batch_neg, dtype=torch.long, device=self.device)
            yield batch_users, batch_pos, batch_neg