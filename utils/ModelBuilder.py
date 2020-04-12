from models.DAE import DAE
from models.BPRMF import BPRMF
# from models.CML import CML
# from models.GMF import GMF
# from models.MLP import MLP
# from models.NeuMF import NeuMF
from models.MultVAE import MultVAE
from models.EASE import EASE

def build_model(model_name, model_conf, num_users, num_items, device):
    model_name = model_name.lower()

    if model_name == 'dae':
        model = DAE(model_conf, num_users, num_items, device)
    elif model_name =='bprmf':
        model = BPRMF(model_conf, num_users, num_items, device)
    # elif model_name == 'gmf':
    #     model = GMF(model_conf, num_users, num_items, device)
    # elif model_name == 'mlp':
    #     model = MLP(model_conf, num_users, num_items, device)
    # elif model_name == 'neumf':
    #     model = NeuMF(model_conf, num_users, num_items, device)
    elif model_name == 'multvae':
        model = MultVAE(model_conf, num_users, num_items, device)
    elif model_name == 'ease':
        model = EASE(model_conf, num_users, num_items, device)
    else:
        raise NotImplementedError('Choose correct model name.')

    return model