import torch.nn as nn
import torch
from gnn_modules import GraphConvolution
from GCL.models import DualBranchContrast
import GCL.losses as L
from itertools import combinations, islice
import random
cuda = True if torch.cuda.is_available() else False

class ContrastiveTrainer(nn.Module):
    def __init__(self, g_dim, dropout):
        super(ContrastiveTrainer, self).__init__()
        self.gc = GraphConvolution(g_dim, g_dim)
        self.dropout = dropout

    def forward(self, node_features, adj, label=None, trainW=None):

        if trainW:
            drop_percent1 = 0.4
            drop_percent2 = 0.4
            pert_percent1 = 0.1
            pert_percent2 = 0.1

            num_nodes = node_features.shape[0]
            aug1_embedding = random_feature_mask(node_features, drop_percent1)
            aug1_edge_index = random_edge_pert_adj(adj, label, pert_percent1, None, 3)

            aug2_embedding = random_feature_mask(node_features, drop_percent2)
            aug2_edge_index = random_edge_pert_adj(adj, label, pert_percent2, None, 3)

            h0 = self.gc(node_features, adj)
            h1 = self.gc(aug1_embedding, aug1_edge_index)
            h2 = self.gc(aug2_embedding, aug2_edge_index)

            loss = contrastive_loss_wo_cross_network(h1, h2, h0)

            return loss


        else:
            h0 = self.gc(node_features, adj)
            return h0


def sim(h1, h2):
    z1 = nn.functional.normalize(h1, dim=-1, p=2)
    z2 = nn.functional.normalize(h2, dim=-1, p=2)

    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).cuda()
    loss = contrast_model(z1, z2)
    return loss


def contrastive_loss_wo_cross_network(h1, h2, ho):
    intra1 = sim(ho, h1)
    intra2 = sim(ho, h2)
    return intra1 + intra2


# 随机特征掩码
def random_feature_mask(input_feature, drop_percent, device=torch.device('cuda')):
    p = torch.ones(input_feature.shape, dtype=torch.float).bernoulli_(1 - drop_percent).to(device)
    aug_feature = input_feature * p
    return aug_feature


def random_edge_pert_adj(adj: torch.Tensor,
                         labels: torch.Tensor,
                         pert_percent: float = 0.1,
                         minority_class: int = None,
                         protection_factor: float = 5.0,
                         max_candidates: int = 1000000):  # 新增最大候选边限制

    # 确保张量是合并的
    adj = adj.coalesce()
    idx, val = adj.indices(), adj.values()
    N = adj.shape[0]
    device = adj.device

    if minority_class is None:
        unique_labels, counts = torch.unique(labels, return_counts=True)
        if len(counts) > 0:
            minority_class = unique_labels[torch.argmin(counts)].item()
        else:
            minority_class = -1

    if minority_class != -1:
        minority_mask = (labels == minority_class)
        minority_nodes = torch.where(minority_mask)[0].tolist()
        n_minority = len(minority_nodes)
    else:
        minority_mask = torch.zeros(N, dtype=torch.bool, device=device)
        minority_nodes = []
        n_minority = 0

    mask = idx[0] < idx[1]
    u_idx, u_val = idx[:, mask], val[mask]
    K = u_idx.size(1)

    edge_bitmap = torch.zeros((N, N), dtype=torch.bool, device=device)
    edge_bitmap[u_idx[0], u_idx[1]] = True
    edge_bitmap[u_idx[1], u_idx[0]] = True

    n_drop = max(0, int(K * pert_percent // 2))
    n_add = max(0, int(K * pert_percent // 2))

    if n_drop > 0:
        node1 = u_idx[0, :]
        node2 = u_idx[1, :]
        is_minority_edge = minority_mask[node1] | minority_mask[node2]
        edge_weights = torch.where(is_minority_edge, protection_factor, 1.0)

        probs = 1.0 / edge_weights
        probs = probs / probs.sum()
        drop_indices = torch.multinomial(probs, n_drop, replacement=False)

        mask_keep = torch.ones(K, dtype=torch.bool, device=device)
        mask_keep[drop_indices] = False

        dropped_edges = u_idx[:, drop_indices]
        edge_bitmap[dropped_edges[0], dropped_edges[1]] = False
        edge_bitmap[dropped_edges[1], dropped_edges[0]] = False

        u_idx = u_idx[:, mask_keep]
        u_val = u_val[mask_keep]
        K = u_idx.size(1)
    existing_set = set()
    for i in range(u_idx.size(1)):
        n1, n2 = u_idx[0, i].item(), u_idx[1, i].item()
        existing_set.add((min(n1, n2), max(n1, n2)))
    added_edges = []

    if n_minority >= 2 and n_add > 0:
        minority_candidates = []
        for i, j in combinations(minority_nodes, 2):
            edge = (min(i, j), max(i, j))
            if edge not in existing_set:
                minority_candidates.append(edge)
                if len(minority_candidates) >= max_candidates:
                    break

        num_to_add = min(n_add, len(minority_candidates))
        if num_to_add > 0:
            selected = random.sample(minority_candidates, num_to_add)
            added_edges.extend(selected)
            n_add -= num_to_add
            existing_set.update(selected)

    if n_add > 0:
        candidate_count = 0
        while n_add > 0 and candidate_count < max_candidates:
            candidate_count += 1
            i, j = random.randint(0, N - 1), random.randint(0, N - 1)
            if i == j:
                continue
            edge = (min(i, j), max(i, j))

            if edge not in existing_set:
                if not edge_bitmap[i, j]:
                    added_edges.append(edge)
                    existing_set.add(edge)
                    edge_bitmap[i, j] = True
                    edge_bitmap[j, i] = True
                    n_add -= 1

    if added_edges:
        new_edges = torch.tensor(added_edges, dtype=torch.long, device=device).t()
        new_vals = torch.ones(new_edges.size(1), dtype=torch.float, device=device)
        u_idx = torch.cat([u_idx, new_edges], dim=1)
        u_val = torch.cat([u_val, new_vals])

    full_idx = torch.cat([u_idx, u_idx.flip(0)], dim=1)
    full_val = torch.cat([u_val, u_val])

    return torch.sparse_coo_tensor(full_idx, full_val,
                                   size=(N, N),
                                   device=adj.device).coalesce()


