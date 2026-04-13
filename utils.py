import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from lifelines.utils import concordance_index as c_index
import numpy as np
from sklearn.metrics import precision_recall_curve

cuda = True if torch.cuda.is_available() else False
def prepare_trte_data(data_folder, view_list):

    labels = np.loadtxt(os.path.join(data_folder, "labels_data.csv"), delimiter=',')
    labels = labels.astype(int)
    data_list = []
    for i in view_list:
        data_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_data.csv"), delimiter=','))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_data = []
    for fold, (train_index, test_index) in enumerate(kf.split(labels)):
        fold_data.append({
            'train_idx': train_index,
            'test_idx': test_index
        })
    data_tensor_list = []
    for i in range(len(data_list)):
        data_tensor_list.append(torch.FloatTensor(data_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()


    return data_tensor_list, labels, fold_data



def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"
    adj_train_list = []
    adj_test_list = []
    # print(f"1.device: {data_tr_list.device}")
    # print(f"1.device: {data_trte_list.device}")
    # print(f"1.device: {trte_idx.device}")
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    return adj_train_list, adj_test_list
def gen_trte_adj_mat_test(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"
    adj_train_test_list = []
    for i in range(len(data_trte_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_test_list.append(gen_adj_mat_tensor(data_trte_list[i], adj_parameter_adaptive, adj_metric))
    return adj_train_test_list

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return parameter.detach().cpu().item()
def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    # device = adj.device  # 获取 adj 的设备
    # I = torch.eye(adj.size(0), device=device)
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):

    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr,num_tr:] = 1-dist_tr2te
    else:
        raise NotImplementedError

    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr # retain selected edges
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    return adj


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict


class DynamicLossBalancer(nn.Module):
    def __init__(self, num_losses, init_weights=None, momentum=0.9, lr=0.01):

        super().__init__()
        # 将权重注册为可学习参数
        self.log_weights = nn.Parameter(
            torch.log(torch.ones(num_losses)) if init_weights is None
            else torch.log(torch.tensor(init_weights, dtype=torch.float)))
        self.momentum = momentum
        self.lr = lr

        self.register_buffer('loss_history', torch.zeros(num_losses))
        self.register_buffer('step', torch.zeros(1))
        self.register_buffer('first_pass', torch.ones(1, dtype=torch.bool))

    def get_weights(self):
        """获取归一化权重"""
        return F.softmax(self.log_weights, dim=0)

    def forward(self, losses):

        loss_tensors = [l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in losses]

        current_losses = torch.tensor([l.item() for l in loss_tensors], device=self.loss_history.device)

        if self.first_pass:
            self.loss_history = current_losses
            self.first_pass = torch.zeros(1, dtype=torch.bool)
        else:
            self.loss_history = self.momentum * self.loss_history + (1 - self.momentum) * current_losses

        relative_losses = current_losses / self.loss_history.clamp(min=1e-8)

        weights = self.get_weights()

        total_loss = sum(w * l for w, l in zip(weights, loss_tensors))

        with torch.no_grad():
            grad_direction = relative_losses - relative_losses.mean()

            self.log_weights += self.lr * grad_direction

        self.step += 1

        return total_loss, weights




def calculate_classification_metrics(y_true, y_probs,test_time):
    prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    # print(f1_scores)
    best_idx = np.argmax(f1_scores[:-1])
    best_th = thresholds[best_idx]
    pred_labels = (y_probs >= best_th).astype(int)
    accuracy = accuracy_score(y_true, pred_labels)
    f1 = f1_score(y_true, pred_labels)
    auc_score = roc_auc_score(y_true, y_probs)
    C_index = c_index(test_time, y_probs, y_true)

    print("Test F1: {:.3f}".format(f1))
    print("Test ACC1: {:.3f}".format(accuracy))
    print("Test AUC: {:.3f}".format(auc_score))
    print("Test C-INDEX: {:.3f}".format(C_index))

    return accuracy, f1, auc_score, c_index

