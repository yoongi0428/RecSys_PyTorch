// distutils: language = c++

/*
@author: Zhongchuan Sun
*/
#ifndef EVALUATE_LOO
#define EVALUATE_LOO

#include <vector>
#include <set>
#include <cmath>
#include <future>
// #include "thread_pool.h"

using std::vector;
using std::set;
using std::future;
using std::min;

void evaluate_loo(int users_num,
                      int *rankings, int max_k, int * Ks, int K_len,
                      int **ground_truths, float *results)
{
    vector< vector<float> > hr_ks_results;
    vector< vector<float> > ndcg_ks_results;
    
    for(int uid=0; uid<users_num; uid++)
    {
        int * cur_rankings = rankings + uid * max_k;
        int * truth = ground_truths[uid];
        int truth_len = 1;
        
        vector<float> hr_results(K_len);
        vector<float> ndcg_result(K_len);

        int hit_at_k = max_k + 1;
        set <int> truth_set(truth, truth + truth_len);
        for(int i=0; i<max_k; i++)
        {
            if(truth_set.count(cur_rankings[i]))
            {
                hit_at_k = i + 1;
                break;
            }
        }
        for(int j=0; j<K_len; j++)
        {
            if(Ks[j] >= hit_at_k)
            {
                hr_results[j] = 1.0;
                ndcg_result[j] = 1 / log2(hit_at_k + 1);
            }
            else
            {
                hr_results[j] = 0.0;
                ndcg_result[j] = 0.0;
            }
        }

        hr_ks_results.emplace_back(hr_results);
        ndcg_ks_results.emplace_back(ndcg_result);
    }
    
    float *hr_offset = results + 0 * K_len;  // the offset address of precision in the first user result
    float *ndcg_offset = results + 1 * K_len;  // the offset address of ndcg in the first user result
    
    int metric_num = 2;
    for(vector<float> result: hr_ks_results)
    {
        for(int k=0; k<K_len; k++)
        {
            hr_offset[k] = result[k];
        }
        hr_offset += K_len * metric_num;  // move to the next user's result address
    }

    for(vector<float> result: ndcg_ks_results)
    {
        for(int k=0; k<K_len; k++)
        {
            ndcg_offset[k] = result[k];
        }
        ndcg_offset += K_len * metric_num;  // move to the next user's result address
    }
}

#endif