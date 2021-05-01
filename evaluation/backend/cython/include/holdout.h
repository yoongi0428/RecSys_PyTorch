// distutils: language = c++

/*
@author: Zhongchuan Sun
*/
#ifndef EVALUATE_HOLDOUT
#define EVALUATE_HOLDOUT

#include <vector>
#include <set>
#include <cmath>
#include <future>
// #include "thread_pool.h"

using std::vector;
using std::set;
using std::future;
using std::min;

void evaluate_holdout(int users_num,
                      int *rankings, int max_k, int * Ks, int K_len,
                      int **ground_truths, int *ground_truths_num,
                      float *results)
{
    vector< vector<float> > prec_ks_results;
    vector< vector<float> > recall_ks_results;
    vector< vector<float> > ndcg_ks_results;
    
    for(int uid=0; uid<users_num; uid++)
    {
        int * cur_rankings = rankings + uid * max_k;
        int * truth = ground_truths[uid];
        int truth_len = ground_truths_num[uid];
        
        vector<float> prec_result(K_len);
        vector<float> recall_result(K_len);
        vector<float> ndcg_result(K_len);

        set <int> truth_set(truth, truth + truth_len);

        float hits = 0;
        float iDCG = 0;
        float DCG = 0;
        for(int i=0; i<max_k; i++)
        {
            if(truth_set.count(cur_rankings[i]))
            {
                hits += 1;
                DCG += 1.0 / log2(i + 2);
            }
            if(i < truth_len)
            {
                iDCG += 1.0 / log2(i + 2);
            }
            for(int j=0; j<K_len; j++)
            {
                if(Ks[j] == (i + 1))
                {
                    prec_result[j] = hits / (float) Ks[j];
                    // recall_result[j] = hits / min(truth_len, Ks[j]);
                    recall_result[j] = hits / truth_len;
                    ndcg_result[j] = DCG / iDCG;
                }
            }
        }

        prec_ks_results.emplace_back(prec_result);
        recall_ks_results.emplace_back(recall_result);
        ndcg_ks_results.emplace_back(ndcg_result);
    }
    
    float *prec_offset = results + 0 * K_len;  // the offset address of precision in the first user result
    float *recall_offset = results + 1 * K_len;  // the offset address of recall in the first user result
    float *ndcg_offset = results + 2 * K_len;  // the offset address of ndcg in the first user result
    
    int metric_num = 3;
    for(vector<float> result: prec_ks_results)
    {
        for(int k=0; k < K_len; k++)
        {
            prec_offset[k] = result[k];
        }
        prec_offset += K_len * metric_num;  // move to the next user's result address
    }

    for(vector<float> result: recall_ks_results)
    {
        for(int k=0; k<K_len; k++)
        {
            recall_offset[k] = result[k];
        }
        recall_offset += K_len * metric_num;  // move to the next user's result address
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
