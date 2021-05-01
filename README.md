# Recommender System in PyTorch

Implementations of various top-N recommender systems in [PyTorch](pytorch.org) for practice.
 
[Movielens](https://grouplens.org/datasets/movielens/) 100k & 1M are used as datasets.
 
## Available models
| Model    | Paper                                                                                                                                          |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| BPRMF            | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009. [Link](https://arxiv.org/pdf/1205.2618) |
| ItemKNN          | Jun Wang et al., Unifying user-based and item-based collaborative filtering approaches by similarity fusion. SIGIR 2006. [Link](http://web4.cs.ucl.ac.uk/staff/jun.wang/papers/2006-sigir06-unifycf.pdf) |
| PureSVD          | Paolo Cremonesi et al., Performance of Recommender Algorithms on Top-N Recommendation Tasks. RecSys 2010. [Link](https://dl.acm.org/doi/pdf/10.1145/1864708.1864721) |
| SLIM             | Xia Ning et al., SLIM: Sparse Linear Methods for Top-N Recommender Systems. ICDM 2011. [Link](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf) |
| P3a              | Colin Cooper et al., Random Walks in Recommender Systems: Exact Computation and Simulations. WWW 2014. [Link](http://wwwconference.org/proceedings/www2014/companion/p811.pdf) |
| RP3b             | Bibek Paudel et al., Updatable, accurate, diverse, and scalablerecommendations for interactive applications. TiiS 2017. [Link](https://www.zora.uzh.ch/id/eprint/131338/1/TiiS_2016.pdf) |
| DAE, CDAE        | Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.[Link](https://alicezheng.org/papers/wsdm16-cdae.pdf) |
| MultVAE          | Dawen Liang et al., Variational Autoencoders for Collaborative Filtering. WWW 2018. [Link](https://arxiv.org/pdf/1802.05814) |
| EASE             | Harald Steck, Embarrassingly Shallow Autoencoders for Sparse Data. WWW 2019. [Link](https://arxiv.org/pdf/1905.03375) |
| NGCF             | Xiang Wang, et al., Neural Graph Collaborative Filtering. SIGIR 2019. [Link](https://arxiv.org/pdf/1905.08108.pdf) |
| LightGCN         | Xiangnan He, et al., LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020. [Link](https://arxiv.org/abs/2002.02126) |

<!-- ## To be implemented
| Model | Paper                                                                                                                                          |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| GMF, MLP, NeuMF  | Xiangnan He et al., Neural Collaborative Filtering. WWW 2017. [Link](https://arxiv.org/pdf/1708.05031.pdf) |

| RecVAE           | Ilya Shenbin et al., RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback. WSDM 2020. [Link](https://arxiv.org/abs/1912.11160) | -->

## Enable C++ evaluation
To evaluate with C++ backend, you have to compile C++ and cython with the following script:
```
python setup.py build_ext --inplace
```

If compiled NOT successfully, ```"evaluation with python backend.."``` will be printed in the beginning.

## How to run
1. Edit experiment configurations in ```config.py```
2. Edit model hyperparameters you choose in ```conf/[MODEL_NAME]```
3. run ```main.py```

## Implement your own model
You can add your own model into the framework if:

1. Your model inherits ```BaseModel``` class in ```models/BaseModel.py```
2. Implement necessary methods and add additional methods if you want.
3. Make ```YourModel.conf``` file in ```conf```
4. Add your model in ```models.__init__```

# Reference
Some model implementations and util functions refers to these nice repositories.
- NeuRec: An open source neural recommender library. [Repository](https://github.com/wubinzzu/NeuRec)
- RecSys 2019 - DeepLearning RS Evaluation. [Paper](https://arxiv.org/pdf/1907.06902) [Repository](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation)