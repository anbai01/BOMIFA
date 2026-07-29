import numpy as np
import torch
import torch.nn.functional as F

class TrainedModelSaliencyAnalyzer:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # 确保模型在评估模式

    def compute_saliency(self, x, adj, true,target_node_idx=None, target_class=None):
        # 确保输入需要梯度
        x.requires_grad_(True)

        # 前向传播
        output = self.model(x, adj)
        target = output.sum()
        # 反向传播计算梯度
        self.model.zero_grad()
        target.backward()

        # 获取输入梯度并取绝对值作为Saliency
        saliency = torch.abs(x.grad)

        return saliency.detach(), output.detach()

    def compute_feature_importance(self, x, adj,true, aggregation='mean'):
        """
        计算整体特征重要性（跨所有节点和类别）
        """
        # 计算每个类别的Saliency
        # num_classes = 2#这里要调整一下
        num_classes = 1
        all_saliencies = []

        for class_idx in range(num_classes):
            saliency, _ = self.compute_saliency(x, adj, true,target_class=class_idx)
            all_saliencies.append(saliency)

        # 合并所有类别的Saliency
        stacked_saliency = torch.stack(all_saliencies, dim=0)

        # 根据聚合方法计算最终重要性
        if aggregation == 'mean':
            feature_importance = stacked_saliency.mean(dim=0).mean(dim=0)  # 跨类别和节点平均
        elif aggregation == 'max':
            feature_importance = stacked_saliency.max(dim=0)[0].max(dim=0)[0]  # 取最大值
        else:
            raise ValueError("不支持的聚合方法")

        return feature_importance.cpu().numpy()

    def analyze_node(self, x, adj, node_idx):
        """
        分析特定节点的特征重要性
        """
        saliency, output = self.compute_saliency(x, adj, target_node_idx=node_idx)

        # 获取预测类别
        pred_class = output[node_idx].argmax().item()
        pred_prob = F.softmax(output[node_idx], dim=0)[pred_class].item()

        # 计算特征重要性
        node_feature_importance = saliency[node_idx].cpu().numpy()

        return {
            'saliency_map': saliency,
            'predicted_class': pred_class,
            'prediction_confidence': pred_prob,
            'feature_importance': node_feature_importance,
            'top_features': np.argsort(node_feature_importance)[::-1]  # 降序排列
        }

    def analyze_global_importance(self, x, adj, true_label,top_k=10):
        """
        分析全局特征重要性
        """
        feature_importance = self.compute_feature_importance(x, adj,true_label)

        # 获取Top-K重要特征
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_scores = feature_importance[top_indices]

        return {
            'feature_importance': feature_importance,
            'top_indices': top_indices,
            'top_scores': top_scores
        }