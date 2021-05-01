# distutils: language = c++
from time import time

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "include/func.h":
    void c_top_k_array_index(float *scores_pt, int columns_num, int rows_num,
                             int max_k, int *rankings_pt)

def predict_topk_cy(scores, max_k):
    users_num, rank_len = np.shape(scores)

    # get the pointer of ranking scores
    cdef float * scores_pt = <float *>np.PyArray_DATA(scores)

    # store ranks results
    topk = np.zeros([users_num, max_k], dtype=np.int32)
    cdef int * rankings_pt = <int *>np.PyArray_DATA(topk)

    # get top k rating index
    c_top_k_array_index(scores_pt, rank_len, users_num, max_k, rankings_pt)

    return topk

