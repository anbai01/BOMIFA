# 1. Load trained model
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from Saliency import TrainedModelSaliencyAnalyzer
from models import GCN_E, Classifier, ContrastiveTrainer, all
from contrastive_learning import ContrastiveTrainer


class CombinedModel(nn.Module):
    def __init__(self, gcn_e, classifier, gcn_pare_model, trans_f_model):
        super(CombinedModel, self).__init__()
        self.gcn_e = gcn_e
        self.classifier = classifier
        self.gcn_pare_model = gcn_pare_model
        self.trans_f_model = trans_f_model

    def forward(self, x, adj):
        x1, adj1 = self.gcn_e(x, adj, True)
        x2 = self.trans_f_model(x1)
        x2 = self.gcn_pare_model(x2, adj)
        x = self.classifier(x2)
        return x


def load_trained_model(in_dim, hgcn_dim, i, num, data_folder, dropout=0.4):
    gcn_e_model = GCN_E(in_dim, hgcn_dim, dropout)
    gcn_e_model_path = f'./{data_folder}/models/{num}/E{i}.pth'
    gcn_e_model.load_state_dict(torch.load(gcn_e_model_path))
    gcn_e_model.eval()

    trans_f_model = all(hgcn_dim[-1], 5, 1600, 10, 0.3)
    trans_f_model_path = f'./{data_folder}/models/{num}/H{i}.pth'
    trans_f_model.load_state_dict(torch.load(trans_f_model_path))
    trans_f_model.eval()

    gcn_pare_model = ContrastiveTrainer(hgcn_dim[-1], 0.1)
    gcn_pare_model_path = f'./{data_folder}/models/{num}/P{i}.pth'
    gcn_pare_model.load_state_dict(torch.load(gcn_pare_model_path))
    gcn_pare_model.eval()

    classifier_model = Classifier(hgcn_dim[-1], 1)
    classifier_model_path = f'./{data_folder}/models/{num}/C{i}.pth'
    classifier_model.load_state_dict(torch.load(classifier_model_path))
    classifier_model.eval()

    combined_model = CombinedModel(gcn_e_model, classifier_model, gcn_pare_model, trans_f_model)
    return combined_model


def run_saliency(h, adj_T, dim_he_list, num, pred_label, data_folder='preprocessed', dropout=0.1):
    for i in range(3):
        x = h[i]
        adj = adj_T[i]
        a = i + 1

        in_dim = x.size(1)  # input feature dimension

        trained_model = load_trained_model(in_dim, dim_he_list, a, num, data_folder, dropout)
        saliency_analyzer = TrainedModelSaliencyAnalyzer(trained_model)
        global_importance = saliency_analyzer.analyze_global_importance(x, adj, pred_label, top_k=500)
        path1 = f'./{data_folder}/{a}_featname.csv'
        df = pd.read_csv(path1, header=None)
        featname_list = []
        featname_list.append(df.values.flatten())
        c = df.values.flatten()
        important_feature_names = c[global_importance['top_indices']]

        print("Top 100 globally important features:")
        print(f"Feature indices: {global_importance['top_indices']}")
        print(f"Feature names: {important_feature_names}")
        print(f"Importance scores: {global_importance['top_scores']}")

        save_dir = os.path.join(data_folder, "marker")
        os.makedirs(save_dir, exist_ok=True)  # create directory if it does not exist
        pd.DataFrame({
            "feature_name": c[global_importance['top_indices']],
            "importance": global_importance['top_scores']
        }).to_csv(f"./{data_folder}/marker/{i}_features.csv",
                  index=False, header=True)