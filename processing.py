import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests


def variance_filter(X_train, X_test, threshold):
    var_train = X_train.var(axis=1)
    keep = var_train > threshold
    if keep.sum() == 0:
        print("警告：没有特征通过方差阈值，使用所有特征")
        keep = np.ones(len(var_train), dtype=bool)
    return X_train.loc[keep], X_test.loc[keep]

def fdr_pca_filter(X_train, X_test, y_train, pca_var_thresh=0.5, alpha=0.5, top_k=1000):
    f_scores, p_values = f_classif(X_train.T, y_train)
    reject, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    selected_idx = np.where(reject)[0]
    if len(selected_idx) == 0:
        print("警告：没有特征通过 FDR 校正，保留 F 值最大的 1000 个特征")
        selected_idx = np.argsort(f_scores)[-1000:]
    fdr_passed_idx = set(selected_idx)
    candidates = [idx for idx in np.argsort(f_scores)[::-1] if idx in fdr_passed_idx]
    for k in range(len(candidates), 0, -1):
        sub_idx = candidates[:k]
        sub_features = X_train.index[sub_idx]
        X_train_sub = X_train.loc[sub_features]
        pca_sub = PCA(n_components=2)
        pca_sub.fit(X_train_sub.T)
        if pca_sub.explained_variance_ratio_[0] <= pca_var_thresh:
            final_idx = sub_idx
            print(f"第一主成分解释方差比例 {pca_sub.explained_variance_ratio_[0]:.4f} ≤ {pca_var_thresh}，候选特征数 {k}")
            break
    else:
        print("警告：无法满足 PCA 条件，保留 F 值最大的 1 个特征")
        final_idx = [candidates[0]]
    if top_k is not None and top_k > 0 and len(final_idx) > top_k:
        print(f"应用 top_k={top_k}，从 {len(final_idx)} 个特征中选取前 {top_k} 个")
        final_idx = final_idx[:top_k]
    final_features = X_train.index[final_idx]
    return X_train.loc[final_features], X_test.loc[final_features]

def minmax_by_train(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.T)  # (样本, 特征)
    X_test_scaled = scaler.transform(X_test.T)
    X_train_out = pd.DataFrame(X_train_scaled.T, index=X_train.index, columns=X_train.columns)
    X_test_out = pd.DataFrame(X_test_scaled.T, index=X_test.index, columns=X_test.columns)
    return X_train_out, X_test_out

# ================== 主流程 ==================
def main():
    # ========== 请根据实际情况修改以下参数 ==========
    data_folder = "UCEC"          # 原始数据所在文件夹
    fold_num = 1                         # 使用第几折的标签
    output_folder = "./preprocessed"     # 保存预处理结果的文件夹（会自动创建）
    # =============================================

    os.makedirs(output_folder, exist_ok=True)

    # 1. 读取原始数据
    path_mrna = os.path.join(data_folder, "mrna.csv")
    path_methyl = os.path.join(data_folder, "methylation.csv")
    path_mirna = os.path.join(data_folder, "micrna.csv")
    train_label_path = os.path.join(data_folder, f"fold{fold_num}_train_labels.csv")
    test_label_path = os.path.join(data_folder, f"fold{fold_num}_test_labels.csv")

    df_mrna = pd.read_csv(path_mrna, index_col=0)
    df_methyl = pd.read_csv(path_methyl, index_col=0)
    df_mirna = pd.read_csv(path_mirna, index_col=0)
    train_labels = pd.read_csv(train_label_path)
    test_labels = pd.read_csv(test_label_path)

    # 标签转换（1 - label，与原代码一致）
    # train_labels['label'] = 1 - train_labels['label']
    # test_labels['label'] = 1 - test_labels['label']

    train_samples = train_labels['sample_id'].tolist()
    test_samples = test_labels['sample_id'].tolist()
    common_train = [s for s in train_samples if s in df_mrna.columns]
    common_test = [s for s in test_samples if s in df_mrna.columns]

    X_train_mrna = df_mrna[common_train]
    X_test_mrna = df_mrna[common_test]
    X_train_methyl = df_methyl[common_train]
    X_test_methyl = df_methyl[common_test]
    X_train_mirna = df_mirna[common_train]
    X_test_mirna = df_mirna[common_test]

    y_train = train_labels.set_index('sample_id').loc[common_train]['label'].values
    y_test = test_labels.set_index('sample_id').loc[common_test]['label'].values
    print(f"训练集标签分布: {np.bincount(y_train)}")
    print(f"测试集标签分布: {np.bincount(y_test)}")

    # 2. 方差过滤
    X_train_mrna, X_test_mrna = variance_filter(X_train_mrna, X_test_mrna, threshold=0.00001)
    X_train_methyl, X_test_methyl = variance_filter(X_train_methyl, X_test_methyl, threshold=0.001)
    X_train_mirna, X_test_mirna = variance_filter(X_train_mirna, X_test_mirna, threshold=0)

    if X_train_mirna.shape[0] > 1000:
        var_train = X_train_mirna.var(axis=1)
        top_idx = var_train.sort_values(ascending=False).head(1000).index
        X_train_mirna = X_train_mirna.loc[top_idx]
        X_test_mirna = X_test_mirna.loc[top_idx]
        print(f"miRNA 保留前1000个方差最大特征，实际保留 {X_train_mirna.shape[0]} 个")

    print(f"方差过滤后 mRNA 特征数: {X_train_mrna.shape[0]}")
    print(f"方差过滤后 Methylation 特征数: {X_train_methyl.shape[0]}")
    print(f"方差过滤后 miRNA 特征数: {X_train_mirna.shape[0]}")

    # 3. FDR+PCA 特征选择
    X_train_mrna, X_test_mrna = fdr_pca_filter(X_train_mrna, X_test_mrna, y_train, top_k=1000)
    X_train_methyl, X_test_methyl = fdr_pca_filter(X_train_methyl, X_test_methyl, y_train, top_k=1000)
    # 不对 miRNA 做 FDR+PCA（与原代码一致）

    print(f"FDR+PCA 后 mRNA 特征数: {X_train_mrna.shape[0]}")
    print(f"FDR+PCA 后 Methylation 特征数: {X_train_methyl.shape[0]}")
    print(f"FDR+PCA 后 miRNA 特征数: {X_train_mirna.shape[0]}")

    # 4. MinMax 归一化
    X_train_mrna, X_test_mrna = minmax_by_train(X_train_mrna, X_test_mrna)
    X_train_methyl, X_test_methyl = minmax_by_train(X_train_methyl, X_test_methyl)
    X_train_mirna, X_test_mirna = minmax_by_train(X_train_mirna, X_test_mirna)

    # 5. 分别保存每个组学的训练/测试数据
    X_train_mrna.to_csv(os.path.join(output_folder, "X_train_mrna.csv"))
    print(os.path.join(output_folder, "X_train_mrna.csv"))
    X_test_mrna.to_csv(os.path.join(output_folder, "X_test_mrna.csv"))
    X_train_methyl.to_csv(os.path.join(output_folder, "X_train_methyl.csv"))
    X_test_methyl.to_csv(os.path.join(output_folder, "X_test_methyl.csv"))
    X_train_mirna.to_csv(os.path.join(output_folder, "X_train_mirna.csv"))
    X_test_mirna.to_csv(os.path.join(output_folder, "X_test_mirna.csv"))
if __name__ == "__main__":
    main()