# Recommender System in PyTorch

Implementations of deep learning based top-N recommender systems in [PyTorch](pytorch.org)
 for practice.
 
[Movielens](https://grouplens.org/datasets/movielens/) 100k & 1M are used as datasets.
 

## List of Models
- DAE
- BPRMF

[To be implented]
- GMF
- MLP
- NeuMF
- CML
- NGCF
- And more

## How to run
1. Choose RecSys model and edit configurations in main.py
2. Edit configurations of the model you choose in 'conf'
3. run 'main.py'

## Implement your own model
You can add your own model into the framework if:

1. Your model inherits 'BaseModel' class in 'models/BaseModel.py'
2. Implement necessary methods and add additional methods if you want.
3. Make 'YourModel.conf' file in 'conf'
4. Add your model in 'utils.ModelBuilder.py'

## References


## Update history
- 2019/08/10: First Commit. Base structure codes with Readme 

 