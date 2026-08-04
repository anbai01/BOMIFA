# BOMIFA
Biologically Informed Multi-Omics Integration with Graph Contrastive Learning for Women's Cancer Prognosis
![4e14c86ec332cde9fb7b1b8e1e91688.png](4e14c86ec332cde9fb7b1b8e1e91688.png)

## Overview
BOMIFA is a biologically informed deep learning framework that integrates mRNA expression, miRNA expression, and DNA methylation data for cancer prognosis prediction in women.



## Framework
<img width="1943" height="1013" alt="58beeb3dd3e9d274919dc9ac489b8ba" src="https://github.com/user-attachments/assets/6ecfd731-f77b-4afe-b03b-8c3768b9f5ea" />

BOMIFA takes mRNA expression, miRNA expression, and DNA methylation data as input. Each modality is first processed by an Omics-Enhancement Encoder. A contrastive Transformer module then learns robust modality-specific representations from augmented views. Cross-attention is used to model interactions among omics modalities, and the resulting representations are integrated through low-rank multimodal fusion to generate the final prognostic prediction.


## Repository Structure

```plaintext
BOMIFA/
├── README.md                         # Project documentation
├── requirements.txt                  # List of dependencies
│
├── UCEC/                             # Example dataset (Uterine Corpus Endometrial Carcinoma)
│   ├── fold1_test_labels.csv         # Test set labels for fold 1
│   ├── fold1_train_labels.csv        # Training set labels for fold 1
│   ├── 0_featname.csv           # Full feature names for mRNA (used for saliency mapping)
│   ├── 1_featname.csv           # Full feature names for methylation
│   ├── 2_featname.csv           # Full feature names for miRNA
│   ├── fold1_train_labels.csv   # Training set labels for fold 1
│   ├── fold1_test_labels.csv    # Test set labels for fold 1
│   └── surv_time.csv                 # Survival time and event information
│
├── preprocessed/                     # Preprocessed multi-omics feature data
│   ├── X_test_methyl.csv             # Test set: DNA methylation
│   ├── X_test_mirna.csv              # Test set: miRNA expression
│   ├── X_test_mrna.csv               # Test set: mRNA expression
│   ├── X_train_methyl.csv            # Training set: DNA methylation
│   ├── X_train_mirna.csv             # Training set: miRNA expression
│   └── X_train_mrna.csv              # Training set: mRNA expression
│
├── main_bomifa.py                    # Main entry script (set hyperparameters and launch training)
├── main_fine_marker.py               # Biomarker extraction and saliency analysis script
├── Saliency.py                       # Saliency analysis module for model interpretability
├── attention_modules.py              # Attention mechanism layers (GNN/Transformer)
├── contrastive_learning.py           # Contrastive learning losses and modules
├── cox_loss.py                       # Cox loss function for survival analysis
├── gnn_modules.py                    # Graph neural network (GNN) layer definitions
├── lmf_fusion.py                     # Low-rank multi-modal fusion (LMF)
├── transformer_fusion.py             # Single-modal Transformer encoder
├── omics_fusion_model.py             # Detailed LMF implementation (fusion model)
├── training_pipeline.py              # BOMIFA end-to-end core pipeline
├── train_test.py                     # Training and testing loop functions
├── models.py                         # Complete BOMIFA model definition
├── processing.py                     # Data preprocessing pipeline
└── utils.py                          # Utility functions (metrics, weights, adjacency matrices, etc.)

```

# Quick Start
The following workflow uses the preprocessed UCEC example data included in this repository.
## 1. Clone the repository
git clone https://github.com/anbai01/BOMIFA.git
cd BOMIFA
## 2. Create the environment
conda create -n bomifa python=3.9
conda activate bomifa
## 3. Install dependencies
pip install -r requirements.txt
## 4. Run the example
python main_bomifa.py \
  --data_folder ./UCEC \

## Requirements
- Python 3.9+  We do not recommend using Python versions higher than 3.11, as they often lead to dependency conflicts between deep learning frameworks including TensorFlow, PyTorch, and DGL.##
## Main dependencies
TensorFlow ==2.18.0
PyTorch ==2.2.0
DGL ==1.1.2
NumPy ==1.26.4
pandas ==2.2.2
scikit-learn ==1.5.1
lifelines ==0.29.0
- Other dependencies listed in `requirements.txt`

## Recommended reproducible installation
conda env create -f environment.yml
conda activate bomifa

## Repository Structure
### Input Data Specifications
BOMIFA uses three omics matrices and corresponding outcome labels.
 ```bash
UCEC/
├── mrna.csv                 # mRNA expression matrix (genes × samples)
├── methylation.csv          # DNA methylation matrix (probes × samples)
├── micrna.csv               # miRNA expression matrix (miRNAs × samples)
├── fold1_train_labels.csv   # Training set labels for fold 1
├── fold1_test_labels.csv    # Test set labels for fold 1
├── 0_featname.csv           # Full feature names for mRNA (used for saliency mapping)
├── 1_featname.csv           # Full feature names for methylation
├── 2_featname.csv           # Full feature names for miRNA
└──surv_time.csv  
```
Recommended omic-data format**Example (miRNA expression matrix)**：
 ```bash
Ensembl_ID	TCGA-AJ-A3NH	TCGA-SL-A6J9	TCGA-AJ-A8CW	TCGA-AX-A3GI
hsa-let-7a-1	12.62750927	12.32609183	13.84191944	12.35855423
hsa-let-7a-2	12.62163327	12.33025323	13.82986345	12.34579788
hsa-let-7a-3	12.62603131	12.33025323	13.86777909	12.37162985

```

## Minimal Runnable Example
This minimal example allows users to verify the installation and complete the full workflow without downloading the complete UCEC dataset.


### Use preprocessed data (recommended)
#### Due to the large size of the raw data, we provide ready‑to‑use preprocessed multi‑omics files in the PREPROCESSED/ directory (you can place this folder in the project root). The files are:
 ```bash

PREPROCESSED/
├── X_train_mrna.csv
├── X_train_methyl.csv
├── X_train_mirna.csv
├── X_test_mrna.csv
├── X_test_methyl.csv
└── X_test_mirna.csv
```
### Run the model：
 ```bash
python main_bomifa.py

```
### main_bomifa.py accepts the following command‑line arguments. To see the full help, run:
 ```bash
 python main_bomifa.py -h
usage: main_bomifa.py [-h] [--data_folder DATA_FOLDER]
                      [--view_list VIEW_LIST [VIEW_LIST ...]]
                      [--num_epoch_pretrain NUM_EPOCH_PRETRAIN]
                      [--transformer_epochs TRANSFORMER_EPOCHS] [--lr_e_gcn LR_E_GCN]   
                      [--lr_e_cl_transformer LR_E_CL_TRANSFORMER] [--n_head N_HEAD]     
                      [--d_ff D_FF] [--num_layers NUM_LAYERS]
                      [--cross_num_heads CROSS_NUM_HEADS] [--d_model D_MODEL]
                      [--rank RANK] [--lr_cross_attention LR_CROSS_ATTENTION]
                      [--lr_c LR_C] [--all_lr ALL_LR] [--num_classes NUM_CLASSES]    
 ```
#### Model Evaluation Metrics
During model training, test set metrics are printed at each epoch, including F1 score, Accuracy (ACC), AUC and Concordance Index (C-index) for survival prediction.
Example console output:
 ```bash
Test: Epoch 
Test F1: 
Test ACC1:
Test AUC:
Test C-INDEX:
 ```
#### After training, the script automatically invokes Saliency.py to perform saliency analysis and extract top‑ranked biomarkers. The results (feature names and importance scores) are saved in the UCEC/marker/.




### Marker Genes Output
Final filtered marker gene lists are saved under the directory:
`./UCEC/marker/`
0_features.csv – important markers identified from mRNA expression data
1_features.csv – important markers identified from DNA methylation data
2_features.csv – important markers identified from miRNA expression data
### Example content of 0_features.csv (mRNA, top 10 features shown, ranked by importance)
```bash
feature_name	importance
ENSG00000236054.1	1.3456509
ENSG00000278317.1	1.2892737
ENSG00000252258.1	1.2818351
ENSG00000248826.2	1.1951966
ENSG00000237525.7	1.1387562
ENSG00000259734.1	1.1228423
ENSG00000286006.2	1.1180277
ENSG00000243674.1	1.0469915
ENSG00000267591.1	1.0240933
ENSG00000213091.2	1.0034016
```
### Example content of 1_features.csv (methylation, top 10 features shown, ranked by importance)

```bash
feature_name	importance
cg05462236	1.1118071
cg24207616	1.076387
cg11753771	0.9999563
cg19523029	0.94767946
cg13728003	0.9268874
cg13651986	0.9215557
cg23694187	0.9067134
cg06317803	0.8951645
cg14993167	0.8761975
cg24864831	0.8727188
```
### Example content of 2_features.csv (miRNA, top 10 features shown, ranked by importance)


```bash

feature_name	importance
hsa-mir-599	1.2169006
hsa-mir-3941	1.1117061
hsa-mir-4684	1.0730591
hsa-mir-6774	0.99189335
hsa-mir-4434	0.9726206
hsa-mir-4423	0.944213
hsa-mir-6733	0.89460856
hsa-mir-3660	0.89249796
hsa-mir-3146	0.8750569
hsa-mir-1181	0.8715525
```
#### feature_name: A unique identifier for the feature 
#### importance: The saliency score – higher values mean greater contribution to the prediction.



## Usage


### Data Preprocessing

Run the preprocessing script to generate the `preprocessed/` folder:

python processing.py

### processing.py  
The multi-omics data preprocessing pipeline consists of five key steps to ensure robust and biologically meaningful feature representation:

1. **Data Loading & Sample Alignment**  
   Raw mRNA, DNA methylation, and miRNA expression data are loaded and aligned with the corresponding training and test sample labels from the UCEC dataset.

2. **Variance Filtering**  
   Low-variance features are removed to reduce noise. Separate thresholds are applied for mRNA, methylation, and miRNA data. For miRNA, only the top 1000 features with the highest variance are retained.

3. **FDR + PCA Feature Selection**  
   Statistical feature selection is performed using ANOVA F-test followed by FDR correction. PCA is then applied to ensure the first principal component explains less than 50% variance, avoiding over-dominant features. Up to 1000 significant features are retained for mRNA and methylation.

4. **Min-Max Normalization**  
   All features are normalized to the range [0, 1] using statistics computed solely from the training set to prevent data leakage.

5. **Output Saving**  
   Preprocessed training and test sets for all three modalities are saved into the `preprocessed/` directory for model training.

   
python main_bomifa.py

### Parameters(main_bomifa.py)
| Parameter                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| data_folder              | Path to the dataset directory                                               |
| view_list                | List of multi-omics modalities (mRNA, miRNA, methylation)                  |
| num_epoch_pretrain       | Number of epochs for GNN pre-training                                      |
| transformer_epochs       | Number of epochs for contrastive learning Transformer training              |
| lr_e_gcn                 | Learning rate for the GNN encoder                                          |
| lr_e_cl_transformer      | Learning rate for the contrastive learning Transformer encoder             |
| n_head                   | Number of attention heads in the single-modal Transformer                  |
| d_ff                     | Hidden dimension of the Transformer feed-forward network (FFN)              |
| num_layers               | Number of stacked layers in the Transformer module                          |
| cross_num_heads          | Number of attention heads in the cross-attention module                      |
| d_model                  | Dimension of the model feature embedding                                    |
| rank                     | Rank parameter for the low-rank multi-modal fusion (LMF) module             |
| lr_cross_attention       | Learning rate for the cross-attention module                                |
| lr_c                     | Learning rate for the classifier layer                                      |
| all_lr                   | Learning rate for the joint training of the entire model                    |
| num_classes              | Number of classes for the classification task (binary classification)       |


## Applying BOMIFA to a New Dataset
Step 1. Prepare the input files
Create a new dataset directory:
data/MY_DATASET/
├── mrna.csv
├── methylation.csv
├── mirna.csv
├── train_labels.csv
├── test_labels.csv
├── surv_time.csv
├── 0_featname.csv
├── 1_featname.csv
└── 2_featname.csv
Step 2. Run preprocessing
python processing.py \
  --data_folder ./data/MY_DATASET \
  --output_folder ./preprocessed/MY_DATASET
Step 3. Train and evaluate the model
python main_bomifa.py \
  --dataset MY_DATASET \
  --data_folder ./preprocessed/MY_DATASET \
  --view_list mrna methylation mirna \
  --seed 42



## GPU/CPU Support
The code automatically detects GPU availability:
GPU available: Uses CUDA for training (recommended)
CPU only: Falls back to CPU (slower but functional)



