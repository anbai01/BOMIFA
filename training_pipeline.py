import numpy as np
import pandas as pd
import torch
import os
cuda = True if torch.cuda.is_available() else False
from sklearn.feature_selection import f_classif
import os
import torch
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from utils import (
    one_hot_tensor,
    cal_sample_weight,
    # gen_adj_mat_tensor,
    # gen_test_adj_mat_tensor,
    # cal_adj_mat_parameter,
    gen_trte_adj_mat,
    # gen_trte_adj_mat_test,
    # prepare_trte_data
)
from train_test import train_test
def model_prepare(
        data_folder,view_list,num_class,lr_e_gcn,lr_e_cl_transformer,nhead,d_ff,
        num_layers,cross_num_heads,d_model,rank,lr_cross_attention,lr_c,
        all_lr,num_epoch_pretrain,transformer_epochs):

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

    if data_folder in dataset_config:
        config = dataset_config[data_folder]
        adj_parameter = config['adj_parameter']
        dim_he_list = config['dim_he_list']
    else:
        raise ValueError(f"Unsupported dataset: {data_folder}")

        # 创建模型保存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, data_folder)
    model_folder = os.path.join(data_folder, "models")
    os.makedirs(model_folder, exist_ok=True)
    # ================== 2. 预处理：读取数据并筛选特征 ==================
    # 假设数据文件路径如下（根据实际情况修改）
    path_mrna = os.path.join(data_folder, "mrna.csv")
    path_methyl = os.path.join(data_folder, "methylation.csv")
    path_mirna = os.path.join(data_folder, "micrna.csv")
    train_label_path = os.path.join(data_folder, "fold1_train_labels.csv")
    test_label_path = os.path.join(data_folder, "fold1_test_labels.csv")

    # 读取组学数据（行=特征，列=样本）
    df_mrna = pd.read_csv(path_mrna, index_col=0)
    df_methyl = pd.read_csv(path_methyl, index_col=0)
    df_mirna = pd.read_csv(path_mirna, index_col=0)

    # 读取标签文件，获取样本ID
    train_labels = pd.read_csv(train_label_path)
    test_labels = pd.read_csv(test_label_path)


    train_samples = train_labels['sample_id'].tolist()
    test_samples = test_labels['sample_id'].tolist()

    # 样本匹配
    common_train = [s for s in train_samples if s in df_mrna.columns]
    common_test = [s for s in test_samples if s in df_mrna.columns]



    # 提取训练/测试数据
    X_train_mrna = df_mrna[common_train]
    X_test_mrna = df_mrna[common_test]
    X_train_methyl = df_methyl[common_train]
    X_test_methyl = df_methyl[common_test]
    X_train_mirna = df_mirna[common_train]
    X_test_mirna = df_mirna[common_test]

    # 准备训练标签
    y_train = train_labels.set_index('sample_id').loc[common_train]['label'].values
    print(f"训练集标签分布: {np.bincount(y_train)}")

    # ---------- 方差过滤（按组学设置不同阈值）----------
    def variance_filter(X_train, X_test, threshold):
        var_train = X_train.var(axis=1)
        keep = var_train > threshold
        if keep.sum() == 0:
            print("警告：没有特征通过方差阈值，使用所有特征")
            keep = np.ones(len(var_train), dtype=bool)
        return X_train.loc[keep], X_test.loc[keep]

    X_train_mrna, X_test_mrna = variance_filter(X_train_mrna, X_test_mrna, threshold=0.00001)
    X_train_methyl, X_test_methyl = variance_filter(X_train_methyl, X_test_methyl, threshold=0.001)
    X_train_mirna, X_test_mirna = variance_filter(X_train_mirna, X_test_mirna, threshold=0)
    if X_train_mirna.shape[0] > 1000:
        var_train = X_train_mirna.var(axis=1)
        top800_idx = var_train.sort_values(ascending=False).head(1000).index
        X_train_mirna = X_train_mirna.loc[top800_idx]
        X_test_mirna = X_test_mirna.loc[top800_idx]
        print(f"miRNA 保留前800个方差最大特征，实际保留 {X_train_mirna.shape[0]} 个")
    else:
        print(f"miRNA 特征数不足800，保留全部 {X_train_mirna.shape[0]} 个")
    print(f"方差过滤后 mRNA 特征数: {X_train_mrna.shape[0]}")
    print(f"方差过滤后 Methylation 特征数: {X_train_methyl.shape[0]}")
    print(f"方差过滤后 miRNA 特征数: {X_train_mirna.shape[0]}")

    # ---------- FDR+PCA 特征选择 ----------
    def fdr_pca_filter(X_train, X_test, y_train, pca_var_thresh=0.5, alpha=0.5, top_k=1000):
        # 计算 F 值和 p 值
        f_scores, p_values = f_classif(X_train.T, y_train)

        # FDR 校正
        reject, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        selected_idx = np.where(reject)[0]

        if len(selected_idx) == 0:
            print("警告：没有特征通过 FDR 校正，保留 F 值最大的 1000 个特征")
            selected_idx = np.argsort(f_scores)[-1000:]

        # 按 F 值降序排序所有通过 FDR 的特征
        fdr_passed_idx = set(selected_idx)
        candidates = [idx for idx in np.argsort(f_scores)[::-1] if idx in fdr_passed_idx]

        # PCA 自适应削减
        for k in range(len(candidates), 0, -1):
            sub_idx = candidates[:k]
            sub_features = X_train.index[sub_idx]
            X_train_sub = X_train.loc[sub_features]
            pca_sub = PCA(n_components=2)
            pca_sub.fit(X_train_sub.T)
            if pca_sub.explained_variance_ratio_[0] <= pca_var_thresh:
                final_idx = sub_idx
                print(
                    f"第一主成分解释方差比例 {pca_sub.explained_variance_ratio_[0]:.4f} ≤ {pca_var_thresh}，候选特征数 {k}")
                break
        else:
            print("警告：无法满足 PCA 条件，保留 F 值最大的 1 个特征")
            final_idx = [candidates[0]]

        # 可选 top_k 截断
        if top_k is not None and top_k > 0:
            if len(final_idx) > top_k:
                print(f"应用 top_k={top_k}，从 {len(final_idx)} 个特征中选取前 {top_k} 个")
                final_idx = final_idx[:top_k]

        final_features = X_train.index[final_idx]
        return X_train.loc[final_features], X_test.loc[final_features]

    X_train_mrna, X_test_mrna = fdr_pca_filter(X_train_mrna, X_test_mrna, y_train, top_k=1000)
    X_train_methyl, X_test_methyl = fdr_pca_filter(X_train_methyl, X_test_methyl, y_train, top_k=1000)
    # X_train_mirna, X_test_mirna = fdr_pca_filter(X_train_mirna, X_test_mirna, y_train, top_k=1000)

    print(f"FDR+PCA 后 mRNA 特征数: {X_train_mrna.shape[0]}")
    print(f"FDR+PCA 后 Methylation 特征数: {X_train_methyl.shape[0]}")
    print(f"FDR+PCA 后 miRNA 特征数: {X_train_mirna.shape[0]}")

    # ---------- MinMax 归一化 ----------
    def minmax_by_train(X_train, X_test):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.T)  # (样本, 特征)
        X_test_scaled = scaler.transform(X_test.T)
        X_train_out = pd.DataFrame(X_train_scaled.T, index=X_train.index, columns=X_train.columns)
        X_test_out = pd.DataFrame(X_test_scaled.T, index=X_test.index, columns=X_test.columns)
        return X_train_out, X_test_out

    X_train_mrna, X_test_mrna = minmax_by_train(X_train_mrna, X_test_mrna)
    X_train_methyl, X_test_methyl = minmax_by_train(X_train_methyl, X_test_methyl)
    X_train_mirna, X_test_mirna = minmax_by_train(X_train_mirna, X_test_mirna)

    # ================== 3. 转换为模型所需格式 ==================
    # 将数据转为 (样本, 特征) 的张量，并统一数据类型
    def to_tensor(df):
        return torch.FloatTensor(df.T.values)  # 注意转置为 (样本, 特征)

    # 训练数据张量列表（三个视图）
    data_train = [
        to_tensor(X_train_mrna),
        to_tensor(X_train_methyl),
        to_tensor(X_train_mirna)
    ]

    # 全部数据（训练+测试）用于构建邻接矩阵和测试
    X_all_mrna = pd.concat([X_train_mrna, X_test_mrna], axis=1)
    X_all_methyl = pd.concat([X_train_methyl, X_test_methyl], axis=1)
    X_all_mirna = pd.concat([X_train_mirna, X_test_mirna], axis=1)

    data_all = [
        to_tensor(X_all_mrna),
        to_tensor(X_all_methyl),
        to_tensor(X_all_mirna)
    ]
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        data_train = [t.cuda() for t in data_train]
        data_all = [t.cuda() for t in data_all]


    # 标签（训练和测试）
    y_test = test_labels.set_index('sample_id').loc[common_test]['label'].values
    labels = np.concatenate([y_train, y_test]).astype(int)

    # 索引
    n_train = len(y_train)
    n_test = len(y_test)
    trte_idx = {"tr": list(range(n_train)), "te": list(range(n_train, n_train + n_test))}

    # ================== 4. 生成邻接矩阵 ==================
    adj_tr_list, adj_te_list = gen_trte_adj_mat(
        data_train, data_all, trte_idx, adj_parameter
    )

    # ================== 5. 准备训练标签和权重 ==================
    cuda_available = torch.cuda.is_available()
    labels_tr_tensor = torch.LongTensor(labels[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(y_train, num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    if cuda_available:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
        # Get dimensions and adjacency matrices
        dim_list = [x.shape[1] for x in data_train]

        # 定义迭代文件夹（这里仅使用 fold 0）
        iteration_folder = os.path.join(model_folder, "0")
        os.makedirs(iteration_folder, exist_ok=True)

        # Train and test the model
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
            fold_data_train=data_train,  # 训练数据列表
            fold_data_trte=data_all,  # 全部数据列表
            labels_trte=labels,  # 全部标签
            trte_idx=trte_idx,
            iteration_folder=iteration_folder,
            common_train=common_train,
        common_test=common_test
        )
