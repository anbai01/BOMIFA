import numpy as np
import torch
import torch.nn.functional as F

class TrainedModelSaliencyAnalyzer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    def compute_saliency(self, x, adj, true,target_node_idx=None, target_class=None):

        x.requires_grad_(True)
        output = self.model(x, adj)
        target = output.sum()
        self.model.zero_grad()
        target.backward()
        saliency = torch.abs(x.grad)
        return saliency.detach(), output.detach()

    def compute_feature_importance(self, x, adj,true, aggregation='mean'):
        num_classes = 1
        all_saliencies = []

        for class_idx in range(num_classes):
            saliency, _ = self.compute_saliency(x, adj, true,target_class=class_idx)
            all_saliencies.append(saliency)

        stacked_saliency = torch.stack(all_saliencies, dim=0)

        if aggregation == 'mean':
            feature_importance = stacked_saliency.mean(dim=0).mean(dim=0)
        elif aggregation == 'max':
            feature_importance = stacked_saliency.max(dim=0)[0].max(dim=0)[0]
        else:
            raise ValueError("不支持的聚合方法")

        return feature_importance.cpu().numpy()

    def analyze_node(self, x, adj, node_idx):
        saliency, output = self.compute_saliency(x, adj, target_node_idx=node_idx)

        pred_class = output[node_idx].argmax().item()
        pred_prob = F.softmax(output[node_idx], dim=0)[pred_class].item()

        node_feature_importance = saliency[node_idx].cpu().numpy()

        return {
            'saliency_map': saliency,
            'predicted_class': pred_class,
            'prediction_confidence': pred_prob,
            'feature_importance': node_feature_importance,
            'top_features': np.argsort(node_feature_importance)[::-1]  
        }

    def analyze_global_importance(self, x, adj, true_label,top_k=10):

        feature_importance = self.compute_feature_importance(x, adj,true_label)


        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_scores = feature_importance[top_indices]

        return {
            'feature_importance': feature_importance,
            'top_indices': top_indices,
            'top_scores': top_scores
        }