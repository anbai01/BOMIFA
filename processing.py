import os
import argparse
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
        print("Warning: No features passed variance threshold, using all features")
        keep = np.ones(len(var_train), dtype=bool)
    return X_train.loc[keep], X_test.loc[keep]


def fdr_pca_filter(X_train, X_test, y_train, pca_var_thresh=0.5, alpha=0.5, top_k=1000):
    f_scores, p_values = f_classif(X_train.T, y_train)
    reject, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    selected_idx = np.where(reject)[0]
    if len(selected_idx) == 0:
        print("Warning: No features passed FDR correction, keeping top 1000 features by F-score")
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
            print(f"First PC explained variance ratio {pca_sub.explained_variance_ratio_[0]:.4f} <= {pca_var_thresh}, candidate features {k}")
            break
    else:
        print("Warning: Could not satisfy PCA condition, keeping 1 feature with highest F-score")
        final_idx = [candidates[0]]
    if top_k is not None and top_k > 0 and len(final_idx) > top_k:
        print(f"Applying top_k={top_k}, selecting first {top_k} from {len(final_idx)} features")
        final_idx = final_idx[:top_k]
    final_features = X_train.index[final_idx]
    return X_train.loc[final_features], X_test.loc[final_features]


def minmax_by_train(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.T)  # (samples, features)
    X_test_scaled = scaler.transform(X_test.T)
    X_train_out = pd.DataFrame(X_train_scaled.T, index=X_train.index, columns=X_train.columns)
    X_test_out = pd.DataFrame(X_test_scaled.T, index=X_test.index, columns=X_test.columns)
    return X_train_out, X_test_out


def main():
    parser = argparse.ArgumentParser(description="Multi-omics data preprocessing (variance filter + FDR/PCA feature selection + MinMax scaling)")
    parser.add_argument("--data_folder", type=str, default="UCEC",
                        help="Folder containing raw data (mrna.csv, methylation.csv, micrna.csv and label files)")
    parser.add_argument("--train_labels", type=str, default="fold1_train_labels.csv",
                        help="Training label file name (under data_folder)")
    parser.add_argument("--test_labels", type=str, default="fold1_test_labels.csv",
                        help="Test label file name (under data_folder)")
    parser.add_argument("--output_folder", type=str, default="./preprocessed",
                        help="Output folder for preprocessed data (will be created)")
    parser.add_argument("--top_k", type=int, default=1000,
                        help="Number of top features to keep after FDR+PCA (for mRNA and methylation) and also for miRNA variance-based selection")
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder
    top_k = args.top_k

    os.makedirs(output_folder, exist_ok=True)

    # 1. Load raw data
    path_mrna = os.path.join(data_folder, "mrna.csv")
    path_methyl = os.path.join(data_folder, "methylation.csv")
    path_mirna = os.path.join(data_folder, "micrna.csv")
    train_label_path = os.path.join(data_folder, args.train_labels)
    test_label_path = os.path.join(data_folder, args.test_labels)

    df_mrna = pd.read_csv(path_mrna, index_col=0)
    df_methyl = pd.read_csv(path_methyl, index_col=0)
    df_mirna = pd.read_csv(path_mirna, index_col=0)
    train_labels = pd.read_csv(train_label_path)
    test_labels = pd.read_csv(test_label_path)

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

    # Correction: y_test uses test_labels, not train_labels
    y_train = train_labels.set_index('sample_id').loc[common_train]['label'].values
    y_test = test_labels.set_index('sample_id').loc[common_test]['label'].values
    print(f"Training label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")

    # 2. Variance filtering
    X_train_mrna, X_test_mrna = variance_filter(X_train_mrna, X_test_mrna, threshold=0.00001)
    X_train_methyl, X_test_methyl = variance_filter(X_train_methyl, X_test_methyl, threshold=0.001)
    X_train_mirna, X_test_mirna = variance_filter(X_train_mirna, X_test_mirna, threshold=0)

    # miRNA truncation uses top_k (if top_k > 0 and number of features exceeds top_k)
    if top_k is not None and top_k > 0 and X_train_mirna.shape[0] > top_k:
        var_train = X_train_mirna.var(axis=1)
        top_idx = var_train.sort_values(ascending=False).head(top_k).index
        X_train_mirna = X_train_mirna.loc[top_idx]
        X_test_mirna = X_test_mirna.loc[top_idx]
        print(f"miRNA kept top {top_k} by variance, actual retained: {X_train_mirna.shape[0]}")
    elif X_train_mirna.shape[0] > 0:
        print(f"miRNA retained all {X_train_mirna.shape[0]} features (top_k={top_k} not applied)")

    print(f"After variance filter - mRNA features: {X_train_mrna.shape[0]}")
    print(f"After variance filter - Methylation features: {X_train_methyl.shape[0]}")
    print(f"After variance filter - miRNA features: {X_train_mirna.shape[0]}")

    # 3. FDR + PCA feature selection (only for mRNA and methylation)
    X_train_mrna, X_test_mrna = fdr_pca_filter(X_train_mrna, X_test_mrna, y_train, top_k=top_k)
    X_train_methyl, X_test_methyl = fdr_pca_filter(X_train_methyl, X_test_methyl, y_train, top_k=top_k)
    # miRNA does not go through FDR+PCA

    print(f"After FDR+PCA - mRNA features: {X_train_mrna.shape[0]}")
    print(f"After FDR+PCA - Methylation features: {X_train_methyl.shape[0]}")
    print(f"After FDR+PCA - miRNA features: {X_train_mirna.shape[0]}")

    # 4. MinMax scaling (fit on train, transform both)
    X_train_mrna, X_test_mrna = minmax_by_train(X_train_mrna, X_test_mrna)
    X_train_methyl, X_test_methyl = minmax_by_train(X_train_methyl, X_test_methyl)
    X_train_mirna, X_test_mirna = minmax_by_train(X_train_mirna, X_test_mirna)

    # 5. Save each omic's train/test data
    X_train_mrna.to_csv(os.path.join(output_folder, "X_train_mrna.csv"))
    X_test_mrna.to_csv(os.path.join(output_folder, "X_test_mrna.csv"))
    X_train_methyl.to_csv(os.path.join(output_folder, "X_train_methyl.csv"))
    X_test_methyl.to_csv(os.path.join(output_folder, "X_test_methyl.csv"))
    X_train_mirna.to_csv(os.path.join(output_folder, "X_train_mirna.csv"))
    X_test_mirna.to_csv(os.path.join(output_folder, "X_test_mirna.csv"))

    # 6. Save feature names (one column, no header, no index)
    feat_mrna = pd.DataFrame(X_train_mrna.index)
    feat_mrna.to_csv(os.path.join(output_folder, "1_featname.csv"), header=False, index=False)

    feat_methyl = pd.DataFrame(X_train_methyl.index)
    feat_methyl.to_csv(os.path.join(output_folder, "2_featname.csv"), header=False, index=False)

    feat_mirna = pd.DataFrame(X_train_mirna.index)
    feat_mirna.to_csv(os.path.join(output_folder, "3_featname.csv"), header=False, index=False)

    print(f"Feature name files saved: 1_featname.csv (mRNA), 2_featname.csv (methylation), 3_featname.csv (miRNA)")
    print(f"Preprocessing finished. All files saved to: {output_folder}")


if __name__ == "__main__":
    # If you want to run directly in IDE, use the following mock args; if using command line, comment them out
    import sys
    # sys.argv = [
    #     "processing.py",
    #     "--data_folder", "UCEC1",
    #     "--train_labels", "fold1_train_labels.csv",
    #     "--test_labels", "fold1_test_labels.csv",
    #     "--output_folder", "./preprocessed1",
    #     "--top_k", "500"      # Controls the feature count limit for all omics
    # ]
    main()