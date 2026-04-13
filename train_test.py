from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils import DynamicLossBalancer as DB, DynamicLossBalancer
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from cox_loss import loss as surv_loss
cuda = True if torch.cuda.is_available() else False
import matplotlib.pyplot as plt
from utils import calculate_classification_metrics as metrics
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

CUDA_AVAILABLE = torch.cuda.is_available()

def train_epoch_gnn(data_list, adj_list, label, model_dict, optim_dict):
    """训练GNN epoch"""
    for model in model_dict.values():
        model.train()

    num_view = len(data_list)
    losses = []
    for i in range(num_view):
        optim_dict[f"C{i + 1}"].zero_grad()
        features = model_dict[f"E{i + 1}"](data_list[i], adj_list[i])
        predictions = model_dict[f"C{i + 1}"](features)
        ci_loss = surv_loss(predictions, label)

        ci_loss.backward()
        optim_dict[f"C{i + 1}"].step()
        losses.append(ci_loss)
    return losses

def train_epoch_transform(data_list, adj_list, label, model_dict, optim_dict, loss_balancer):
    """训练Transformer epoch"""
    for model in model_dict.values():
        model.train()

    num_view = len(data_list)
    losses = []
    for i in range(num_view):
        optim_dict[f"V{i + 1}"].zero_grad()
        x_out, adj_out = model_dict[f"E{i + 1}"](data_list[i], adj_list[i], True)
        features = model_dict[f"H{i + 1}"](x_out)
        pred1 = model_dict[f"C{i + 1}"](features)
        pred2 = model_dict[f"P{i + 1}"](features, adj_out, label, True)

        loss1 = surv_loss(pred1, label)
        combined_loss, weights = loss_balancer([loss1, pred2])
        print(f"Loss weights: {weights}")

        combined_loss.backward()
        optim_dict[f"V{i + 1}"].step()
        losses.append(combined_loss)
    return losses

def train_cross_attention(data_list, adj_list, label, model_dict, optim_dict):
    """训练交叉注意力"""
    for model in model_dict.values():
        model.train()

    num_view = len(data_list)
    optim_dict["R"].zero_grad()

    feature_list = []
    for j in range(num_view):
        features, adj = model_dict[f"E{j + 1}"](data_list[j], adj_list[j], True)
        processed_features = model_dict[f"H{j + 1}"](features)
        projected_features = model_dict[f"P{j + 1}"](processed_features, adj)
        feature_list.append(projected_features)

    cross_features = model_dict["D"](feature_list[0], feature_list[1], feature_list[2])
    predictions = model_dict["C3"](cross_features)
    loss = surv_loss(predictions, label)

    loss.backward()
    torch.nn.utils.clip_grad_value_(model_dict["D"].parameters(), clip_value=5.0)
    optim_dict["R"].step()

    return loss


def train_epoch_final(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict):
    """最终训练epoch"""
    for model in model_dict.values():
        model.train()

    num_view = len(data_list)
    optim_dict["C"].zero_grad()

    feature_list = []
    for j in range(num_view):
        features, adj = model_dict[f"E{j + 1}"](data_list[j], adj_list[j], True)
        processed_features = model_dict[f"H{j + 1}"](features)
        projected_features = model_dict[f"P{j + 1}"](processed_features, adj)
        feature_list.append(projected_features)

    predictions_list = [
        model_dict["C1"](feature_list[0]),
        model_dict["C2"](feature_list[1]),
        model_dict["C3"](model_dict["D"](feature_list[0], feature_list[1], feature_list[2]))
    ]

    final_predictions = model_dict["C"](predictions_list)
    loss = surv_loss(final_predictions, label)

    loss.backward()
    torch.nn.utils.clip_grad_value_(model_dict["C"].parameters(), clip_value=3.0)
    optim_dict["C"].step()

    return loss

def train_epoch_all(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict):
    """完整训练epoch"""
    for model in model_dict.values():
        model.train()

    num_view = len(data_list)
    optim_dict["A"].zero_grad()

    feature_list = []
    for j in range(num_view):
        features, adj = model_dict[f"E{j + 1}"](data_list[j], adj_list[j], True)
        processed_features = model_dict[f"H{j + 1}"](features)
        projected_features = model_dict[f"P{j + 1}"](processed_features, adj)
        feature_list.append(projected_features)

    predictions_list = [
        model_dict["C1"](feature_list[0]),
        model_dict["C2"](feature_list[1]),
        model_dict["C3"](model_dict["D"](feature_list[0], feature_list[1], feature_list[2]))
    ]

    final_predictions = model_dict["C"](predictions_list)
    loss = surv_loss(final_predictions, label)

    loss.backward()
    torch.nn.utils.clip_grad_value_(model_dict["C"].parameters(), clip_value=3.0)
    optim_dict["A"].step()

    return loss

def test_epoch(test_label, data_list, adj_list, test_indices, model_dict, threshold=None):
    """测试epoch"""
    for model in model_dict.values():
        model.eval()

    num_view = len(data_list)
    feature_list = []

    for j in range(num_view):
        features, adj = model_dict[f"E{j + 1}"](data_list[j], adj_list[j], True)
        processed_features = model_dict[f"H{j + 1}"](features)
        projected_features = model_dict[f"P{j + 1}"](processed_features, adj)
        feature_list.append(projected_features)

    if num_view >= 2:
        predictions_list = [
            model_dict["C1"](feature_list[0]),
            model_dict["C2"](feature_list[1]),
            model_dict["C3"](model_dict["D"](feature_list[0], feature_list[1], feature_list[2]))
        ]
        predictions = model_dict["C"](predictions_list)
    else:
        predictions = feature_list[0]

    test_predictions = predictions[test_indices, :]
    scores = test_predictions.data.cpu().numpy()
    probabilities = F.sigmoid(test_predictions).data.cpu().numpy()
    return scores

def train_test(view_list, num_class, dim_he_list,
               lr_e_gcn, lr_e_cl_transformer, nhead, d_ff, num_layers,
               cross_num_heads, d_model, rank, lr_cross_attention, lr_c,
               all_lr, num_epoch_pretrain, transformer_epochs, adj_tr_list, adj_te_list,
               dim_list, onehot_labels_tr_tensor, labels_tr_tensor,
               sample_weight_tr, fold_data_train, fold_data_trte, labels_trte,
               trte_idx, iteration_folder,common_train,common_test):
    """主训练测试函数"""

    num_view = len(view_list)
    model_dict = init_model_dict(
        num_view, dim_list, dim_he_list, nhead, d_ff, num_layers,
        cross_num_heads, d_model, rank
    )

    if CUDA_AVAILABLE:
        for model in model_dict.values():
            model.cuda()

    print("\nPretraining GCNs...")

    # 训练历史记录
    gnn_loss_history = [[] for _ in range(num_view)]
    transformer_loss_history = [[] for _ in range(num_view)]
    cross_loss_history = []
    final_loss_history = []

    best_accuracy = 0.0

    optim_dict = init_optim(num_view, model_dict, lr_e_gcn, lr_e_cl_transformer,lr_cross_attention,lr_c, all_lr)
    schedulers = {
        key: StepLR(optimizer, step_size=100, gamma=0.99)
        for key, optimizer in optim_dict.items()
    }

    for epoch in range(num_epoch_pretrain):
        losses = train_epoch_gnn(fold_data_train, adj_tr_list, labels_tr_tensor, model_dict, optim_dict)

        for i, loss in enumerate(losses):
            gnn_loss_history[i].append(loss.item())

        for scheduler in schedulers.values():
            scheduler.step()

    plt.figure()
    plt.title('GNN Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    for i in range(num_view):
        plt.plot(range(num_epoch_pretrain), gnn_loss_history[i], marker='o', label=f'View {i + 1}')
    plt.legend()
    plt.savefig('./loss_gnn_pretraining.png')
    plt.close()

    print("\nTraining Transformers...")

    df_surv = pd.read_csv('BOMIFA\\UCEC\\surv_time.csv')
    df_surv.columns = ['Death', 'PatientID', 'Survival']  # 按列顺序重命名
    print(df_surv)
    # 准备样本ID列表（common_train, common_test）
    # 直接合并生存信息
    train_surv = pd.DataFrame({'PatientID': common_train}).merge(df_surv, on='PatientID', how='left')
    test_surv = pd.DataFrame({'PatientID': common_test}).merge(df_surv, on='PatientID', how='left')
    test_times = test_surv['Survival'].values
    train_times = train_surv['Survival'].values

    print(test_times)
    optim_dict = init_optim(num_view, model_dict, lr_e_gcn, lr_e_cl_transformer, lr_cross_attention, lr_c)
    schedulers = {
        key: StepLR(optimizer, step_size=100, gamma=0.99)
        for key, optimizer in optim_dict.items()
    }

    loss_balancer = DynamicLossBalancer(num_losses=2, init_weights=[0.7, 0.3])

    for epoch in range(transformer_epochs + 1):
        print(f"Epoch: {epoch}")
        losses = train_epoch_transform(fold_data_train, adj_tr_list, labels_tr_tensor, model_dict, optim_dict,
                                       loss_balancer)

        for scheduler in schedulers.values():
            scheduler.step()

        for i, loss in enumerate(losses):
            transformer_loss_history[i].append(loss.item())

    plt.figure()
    plt.title('Transformer Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    for i in range(num_view):
        plt.plot(range(transformer_epochs + 1), transformer_loss_history[i], marker='o', label=f'View {i + 1}')
    plt.legend()
    plt.savefig('./loss_transformer_training.png')
    plt.close()

    print("\nTraining Cross Attention...")

    cross_attention_epochs = 300
    for epoch in range(cross_attention_epochs + 1):
        print(f"Epoch: {epoch}")
        loss = train_cross_attention(fold_data_train, adj_tr_list, labels_tr_tensor, model_dict, optim_dict)
        cross_loss_history.append(loss.item())

        # for scheduler in schedulers.values():
        #     scheduler.step(loss.item())

    plt.figure()
    plt.plot(range(cross_attention_epochs + 1), cross_loss_history, marker='o')
    plt.title('Cross Attention Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('./loss_cross_attention.png')
    plt.close()

    print("\nFinal Training...")

    # 第四阶段：最终训练
    final_epochs = 200
    for epoch in range(final_epochs + 1):
        loss1 = train_epoch_final(fold_data_train, adj_tr_list, labels_tr_tensor,
                                  onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        train_epoch_all(fold_data_train, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

        final_loss_history.append(loss1.item())

        if epoch % 1 == 0:
            scores = test_epoch(
                labels_trte[trte_idx["te"]], fold_data_trte, adj_te_list, trte_idx["te"], model_dict, None
            )
            print(f"\nTest: Epoch {epoch}")
            if num_class == 2:
                accuracy, f1, auc_score, c_index = metrics(labels_trte[trte_idx["te"]], scores, test_times)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    df_results = pd.DataFrame([{
                        'accuracy': best_accuracy,
                        'f1': f1,
                        'auc': auc_score,
                        'c_index': c_index
                    }])
                    df_results.to_csv('best_results.csv', index=False)

                    # 保存模型
                    for model_key, model in model_dict.items():
                        if CUDA_AVAILABLE:
                            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                        else:
                            state_dict = model.state_dict()
                        torch.save(state_dict, f'{iteration_folder}/{model_key}.pth')
                        # print(f'Saved {model_key} model parameters')

    # 绘制最终损失
    plt.figure()
    plt.title('Final Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.plot(range(final_epochs + 1), final_loss_history, marker='o')
    plt.savefig('./loss_final_training.png')
    plt.close()