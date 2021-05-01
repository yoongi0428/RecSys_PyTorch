from time import time
import numpy as np

def predict_topk_py(scores, max_k):
    # top_k item index (not sorted)
    s = time()
    relevant_items_partition = (-scores).argpartition(max_k, 1)[:, 0:max_k]
    
    # top_k item score (not sorted)
    relevant_items_partition_original_value = np.take_along_axis(scores, relevant_items_partition, 1)
    
    # top_k item sorted index for partition
    relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, 1)
    
    # sort top_k index
    topk = np.take_along_axis(relevant_items_partition, relevant_items_partition_sorting, 1)

    return topk