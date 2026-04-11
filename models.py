""" Componets of the model
"""
import torch.nn as nn
from contrastive_learning import ContrastiveTrainer
from gnn_modules import GCN_E
from omics_fusion_model import MultiOmicsFusionModel as multi_model
from transformer_fusion import MultiOmicsFusionModel as all
from attention_modules import CrossModalAttention as cross_atten
import torch

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)



class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x
class MultiModel(nn.Module):
    def __init__(self, input_dims, d_model, output,  rank, num_classes):
        super(MultiModel, self).__init__()
        self.multi_model = multi_model(input_dims, d_model, output,  rank, num_classes)

    def forward(self, in_list,W=False):
        output = self.multi_model(in_list,W)
        return output
def init_model_dict(num_view, dim_list, dim_he_list,nhead,d_ff,num_layer, cross_num_heads,d_model,
                    rank,gcn_dopout=0.3,comp_dopout=0.1,h_dopout=0.1,cross_dopout=0.1,output=1):
    model_dict = {}
    for i in range(num_view):
        #这里有改动2被我改成了1
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier(dim_he_list[-1], output)
        model_dict["P{:}".format(i+1)] = ContrastiveTrainer(dim_he_list[-1],comp_dopout)
        model_dict["H{:}".format(i+1)] = all(dim_he_list[-1], nhead, d_ff, num_layer, h_dopout)

    if num_view >= 2:
        model_dict["D"] = cross_atten(dim_he_list[-1], dim_he_list[-1],dim_he_list[-1],dim_he_list[-1], cross_num_heads,cross_dopout)
        model_dict["C"] = MultiModel([1,1,1],  d_model, output, rank, output)
    return model_dict
def init_optim(num_view, model_dict, lr_e=1e-4,lr_e_cl_tramsformer=1e-5,lr_cross_atten=1e-5, lr_c=1e-4,lr_a=1e-6):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()),
                lr=lr_e)
        optim_dict["V{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["P{:}".format(i + 1)].parameters())+list(model_dict["E{:}".format(i + 1)].parameters())+list(model_dict["H{:}".format(i + 1)].parameters()) +list(model_dict["C{:}".format(i+1)].parameters()),
            lr=lr_e_cl_tramsformer)

    all_params = []
    for i in range(num_view):
        e_params = list(model_dict["E{:}".format(i + 1)].parameters())
        p_params = list(model_dict["P{:}".format(i + 1)].parameters())
        h_params = list(model_dict["H{:}".format(i + 1)].parameters())
        c_params = list(model_dict["C{:}".format(i + 1)].parameters())
        all_params.extend(e_params + p_params + h_params + c_params)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(
            model_dict["C"].parameters(), lr=lr_c)
        optim_dict["R"] = torch.optim.Adam(
            list(model_dict["E{:}".format(3)].parameters())+list(model_dict["H{:}".format(3)].parameters())+list(model_dict["D"].parameters()) +list(model_dict["C{:}".format(3)].parameters())+list(model_dict["P{:}".format(3)].parameters()),
            lr=lr_cross_atten)
        d_params = list(model_dict["D"].parameters())
        all_params.extend(d_params)
        c_global_params = list(model_dict["C"].parameters())
        all_params.extend(c_global_params)
        optim_dict["A"] = torch.optim.Adam(all_params, lr_a)
    return optim_dict