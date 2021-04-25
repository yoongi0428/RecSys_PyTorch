# distutils: language = c++
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "include/loo.h":
    void evaluate_loo(int users_num,
                        int *rankings, int max_k, int * Ks, int K_len,
                        int **ground_truths, float *results)

def compute_loo(topk, target, metrics_num, Ks):
    # topk: (user, max_k), numpy array, top k item index
    # ground truth: dictionary, user id: truth item indices
    # num_eval_users
    int_type = np.int32
    num_eval_users = len(target)

    cdef int *rankings_pt = <int *>np.PyArray_DATA(topk)
    
    # the pointer of ground truth, the pointer of the length array of ground truth
    target_pt = <int **> PyMem_Malloc(num_eval_users * sizeof(int *))
    for idx, u in enumerate(target):
        target[u] = np.array(target[u], dtype=int_type)
        target_pt[idx] = <int *> np.PyArray_DATA (target[u])

    cdef int max_k = max(Ks)
    cdef int num_k = len(Ks)
    cdef int * Ks_pt = <int *>np.PyArray_DATA(Ks)

    # | M1@10 | M1@100 | M2@10 | M2@100| ...
    results = np.zeros([num_eval_users, metrics_num * num_k], dtype=np.float32)
    results_pt = <float *>np.PyArray_DATA(results)

    #evaluate
    evaluate_loo(num_eval_users, 
                    rankings_pt, max_k, Ks_pt, num_k,
                    target_pt, results_pt)

    #release the allocated space
    PyMem_Free(target_pt)

    return results