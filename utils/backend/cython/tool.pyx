# distutils: language = c++
from time import time

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "include/tool.h":
    void c_top_k_array_index(float *scores_pt, int columns_num, int rows_num,
                             int max_k, int *rankings_pt)

def predict_topk(scores, max_k):
    users_num, rank_len = np.shape(scores)

    # get the pointer of ranking scores
    s = time()
    cdef float * scores_pt = <float *>np.PyArray_DATA(scores)
    # print('convert scores to pointer:', time() - s)

    # store ranks results
    s = time()
    top_rankings = np.zeros([users_num, max_k], dtype=np.int32)
    cdef int * rankings_pt = <int *>np.PyArray_DATA(top_rankings)
    # print('convert ranking to pointer:', time() - s)

    # get top k rating index
    s = time()
    c_top_k_array_index(scores_pt, rank_len, users_num, max_k, rankings_pt)
    # print('topk:', time() - s)

    return top_rankings

