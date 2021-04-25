// distutils: language = c++

/*
@author: Zhongchuan Sun
*/
#ifndef EVALUATE_NOVELTY
#define EVALUATE_NOVELTY

#include <vector>
#include <set>
#include <cmath>
#include <future>
// #include "thread_pool.h"

using std::vector;
using std::set;
using std::future;
using std::min;

void evaluate_novelty(int users_num,
                        int *rankings, int max_k, int * Ks, int K_len,
                        float *item_self_information, float *results)
{
    vector< vector<float> > novelty_ks_results;

    for(int uid=0; uid<users_num; uid++){
        int * cur_rankings = rankings + uid * max_k;
        vector<float> novelty_result(K_len);

        float total = 0.0;
        int top_item = -1;
        int K_idx = 0;
        float cur_self_info;
        for(int i=0; i<max_k; i++){ //i: 0~99
            top_item = cur_rankings[i];
            cur_self_info = item_self_information[top_item];
            total += cur_self_info;

            if((i+1) == Ks[K_idx]){
                novelty_result[K_idx] = total / Ks[K_idx];
                // novelty_result[K_idx] = total;
                K_idx += 1;
            }
        }
        novelty_ks_results.emplace_back(novelty_result);
    }

    float * novelty_offset = results;  // the offset address of precision in the first user result
    for(vector<float> result: novelty_ks_results)
    {
        for(int k=0; k < K_len; k++)
        {
            novelty_offset[k] = result[k];
        }
        novelty_offset += K_len;  // move to the next user's result address
    }
}
#endif