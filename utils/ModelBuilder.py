from models.ItemKNN import ItemKNN
from models.SLIMElastic import SLIMElastic
from models.DAE import DAE
from models.CDAE import CDAE
from models.BPRMF import BPRMF
# from models.CML import CML
# from models.GMF import GMF
# from models.MLP import MLP
# from models.NeuMF import NeuMF
from models.MultVAE import MultVAE
from models.EASE import EASE

def build_model(model_name, model_conf, num_users, num_items, device):
    if model_name == 'DAE':
        model = DAE(model_conf, num_users, num_items, device)
    elif model_name == 'CDAE':
        model = CDAE(model_conf, num_users, num_items, device)
    elif model_name =='BPRMF':
        model = BPRMF(model_conf, num_users, num_items, device)
    # elif model_name == 'gmf':
    #     model = GMF(model_conf, num_users, num_items, device)
    # elif model_name == 'mlp':
    #     model = MLP(model_conf, num_users, num_items, device)
    # elif model_name == 'neumf':
    #     model = NeuMF(model_conf, num_users, num_items, device)
    elif model_name == 'MultVAE':
        model = MultVAE(model_conf, num_users, num_items, device)
    elif model_name == 'EASE':
        model = EASE(model_conf, num_users, num_items, device)
    elif model_name == 'SLIM':
        model = SLIMElastic(model_conf, num_users, num_items, device)
    elif model_name == 'ItemKNN':
        model = ItemKNN(model_conf, num_users, num_items, device)
    else:
        raise NotImplementedError('Choose correct model name.')

    return model