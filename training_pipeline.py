import numpy as np
import pandas as pd
import torch
import os
from utils import (
    one_hot_tensor,
    cal_sample_weight,
    gen_trte_adj_mat,
)
from train_test import train_test

def model_prepare(
        data_folder, view_list, num_class, lr_e_gcn, lr_e_cl_transformer, nhead, d_ff,
        num_layers, cross_num_heads, d_model, rank, lr_cross_attention, lr_c,
        all_lr, num_epoch_pretrain, transformer_epochs):

    # Configuration for different datasets
    dataset_config = {
        'OV': {'adj_parameter': 2, 'dim_he_list': [300, 300, 200]},
        'CESC': {'adj_parameter': 2, 'dim_he_list': [400, 400, 300]},
        'UCEC': {'adj_parameter': 2, 'dim_he_list': [400, 400, 300]},
        'LGG': {'adj_parameter': 2, 'dim_he_list': [400, 400, 200]},
        'STAD': {'adj_parameter': 2, 'dim_he_list': [400, 400, 300]},
        'HNSC': {'adj_parameter': 2, 'dim_he_list': [400, 400, 200]},
        'BRCA': {'adj_parameter': 2, 'dim_he_list': [400, 400, 300]},
        'LUAD': {'adj_parameter': 2, 'dim_he_list': [300, 300, 200]}
    }

    if data_folder not in dataset_config:
        raise ValueError(f"Unsupported dataset: {data_folder}")
    config = dataset_config[data_folder]
    adj_parameter = config['adj_parameter']
    dim_he_list = config['dim_he_list']

    # 创建模型保存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_folder)
    model_folder = os.path.join(data_path, "models")
    os.makedirs(model_folder, exist_ok=True)

    # ================== 加载预处理好的 CSV 文件 ==================
    # 预处理脚本生成的 CSV 文件默认保存在 ./preprocessed 目录
    output_folder = "./preprocessed"
    X_train_mrna = pd.read_csv(os.path.join(output_folder, "X_train_mrna.csv"), index_col=0)
    X_test_mrna = pd.read_csv(os.path.join(output_folder, "X_test_mrna.csv"), index_col=0)
    X_train_methyl = pd.read_csv(os.path.join(output_folder, "X_train_methyl.csv"), index_col=0)
    X_test_methyl = pd.read_csv(os.path.join(output_folder, "X_test_methyl.csv"), index_col=0)
    X_train_mirna = pd.read_csv(os.path.join(output_folder, "X_train_mirna.csv"), index_col=0)
    X_test_mirna = pd.read_csv(os.path.join(output_folder, "X_test_mirna.csv"), index_col=0)

    train_label_path = os.path.join(data_folder, f"fold1_train_labels.csv")
    test_label_path = os.path.join(data_folder, f"fold1_test_labels.csv")
    train_labels = pd.read_csv(train_label_path)
    test_labels = pd.read_csv(test_label_path)

    y_train = train_labels['label'].values
    y_test = test_labels['label'].values
    common_train = train_labels['sample_id'].tolist()
    common_test = test_labels['sample_id'].tolist()

    # 使用数据的实际类别数覆盖传入的 num_class（如果传入的不一致）
    actual_num_class = len(np.unique(y_train))
    if num_class != actual_num_class:
        print(f"警告：传入的 num_class={num_class} 与数据实际类别数 {actual_num_class} 不符，将使用 {actual_num_class}")
        num_class = actual_num_class

    # ========== 转换为模型所需格式 ==========
    def to_tensor(df):
        return torch.FloatTensor(df.T.values)  # (样本, 特征)

    data_train = [
        to_tensor(X_train_mrna),
        to_tensor(X_train_methyl),
        to_tensor(X_train_mirna)
    ]

    X_all_mrna = pd.concat([X_train_mrna, X_test_mrna], axis=1)
    X_all_methyl = pd.concat([X_train_methyl, X_test_methyl], axis=1)
    X_all_mirna = pd.concat([X_train_mirna, X_test_mirna], axis=1)

    data_all = [
        to_tensor(X_all_mrna),
        to_tensor(X_all_methyl),
        to_tensor(X_all_mirna)
    ]

    labels = np.concatenate([y_train, y_test]).astype(int)
    n_train = len(y_train)
    n_test = len(y_test)
    trte_idx = {"tr": list(range(n_train)), "te": list(range(n_train, n_train + n_test))}

    # 生成邻接矩阵

    # 准备训练标签张量和样本权重
    labels_tr_tensor = torch.LongTensor(labels[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(y_train, num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    dim_list = [x.shape[1] for x in data_train]

    # 如果 GPU 可用，将数据转移到 GPU
    if torch.cuda.is_available():
        data_train = [t.cuda() for t in data_train]
        data_all = [t.cuda() for t in data_all]
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_train, data_all, trte_idx, adj_parameter)

    # 定义迭代文件夹（这里仅使用 fold 0）
    iteration_folder = os.path.join(model_folder, "0")
    os.makedirs(iteration_folder, exist_ok=True)

    # 调用训练函数
    train_test(
        view_list=view_list,
        num_class=num_class,
        dim_he_list=dim_he_list,
        lr_e_gcn=lr_e_gcn,
        lr_e_cl_transformer=lr_e_cl_transformer,
        nhead=nhead,
        d_ff=d_ff,
        num_layers=num_layers,
        cross_num_heads=cross_num_heads,
        d_model=d_model,
        rank=rank,
        lr_cross_attention=lr_cross_attention,
        lr_c=lr_c,
        all_lr=all_lr,
        num_epoch_pretrain=num_epoch_pretrain,
        transformer_epochs=transformer_epochs,
        adj_tr_list=adj_tr_list,
        adj_te_list=adj_te_list,
        dim_list=dim_list,
        onehot_labels_tr_tensor=onehot_labels_tr_tensor,
        labels_tr_tensor=labels_tr_tensor,
        sample_weight_tr=sample_weight_tr,
        fold_data_train=data_train,
        fold_data_trte=data_all,
        labels_trte=labels,
        trte_idx=trte_idx,
        iteration_folder=iteration_folder,
        common_train=common_train,
        common_test=common_test
    )