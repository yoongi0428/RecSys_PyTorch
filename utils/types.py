import pandas as pd
import scipy.sparse as sp
from typing import Tuple

def df_to_sparse(df: pd.DataFrame, shape: Tuple[int, int]) -> sp.csr_matrix:
    users = df.user
    items = df.item
    ratings = df.rating

    sp_matrix = sp.csr_matrix((ratings, (users, items)), shape=shape)
    return sp_matrix

def sparse_to_dict(sparse: sp.csr_matrix) -> dict:
    if isinstance(sparse, dict):
        return sparse
    
    ret_dict = {}
    dim1 = sparse.shape[0]
    for i in range(dim1):
        ret_dict[i] = sparse.indices[sparse.indptr[i]: sparse.indptr[i+1]]
    return ret_dict