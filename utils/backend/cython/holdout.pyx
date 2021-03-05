# distutils: language = c++
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "include/holdout.h":
    void evaluate_holdout(int users_num,
                        int *rankings, int max_k, int * Ks, int K_len,
                        int **ground_truths, int *ground_truths_num,
                        float *results)

def compute_holdout(topk, target, metrics_num, Ks):
    # topk: (user, max_k), numpy array, top k item index
    # ground truth: dictionary, user id: truth item indices
    # num_eval_users
    int_type = np.int32
    num_eval_users = len(target)

    cdef int *rankings_pt = <int *>np.PyArray_DATA(topk)
    
    # the pointer of ground truth, the pointer of the length array of ground truth
    target_pt = <int **> PyMem_Malloc(num_eval_users * sizeof(int *))
    target_num = np.zeros([num_eval_users], dtype=int_type)
    target_num_pt = <int *>np.PyArray_DATA(target_num)
    for idx, u in enumerate(target):
        # if not is_ndarray(ground_truth[u], int_type):
        #     ground_truth[u] = np.array(ground_truth[u], dtype=int_type, copy=True)
        target[u] = np.array(target[u], dtype=int_type)
        target_pt[idx] = <int *> np.PyArray_DATA (target[u])
        target_num[idx] = len(target[u])

    cdef int max_k = max(Ks)
    cdef int num_k = len(Ks)
    cdef int * Ks_pt = <int *>np.PyArray_DATA(Ks)

    # | M1@10 | M1@100 | M2@10 | M2@100| ...
    results = np.zeros([num_eval_users, metrics_num * num_k], dtype=np.float32)
    results_pt = <float *>np.PyArray_DATA(results)

    #evaluate
    evaluate_holdout(num_eval_users, 
                    rankings_pt, max_k, Ks_pt, num_k,
                    target_pt, target_num_pt,
                    results_pt)

    #release the allocated space
    PyMem_Free(target_pt)

    return results