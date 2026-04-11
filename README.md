
# BOMIFA
BOMIFA is an advanced computing framework specifically designed for predicting the prognosis of female patients with cancer. By integrating multiple types of omics data with biological pathway knowledge, BOMIFA is capable of providing accurate and understandable prediction results for cancer outcomes, as well as identifying significant clinical value biomarkers for female cancers.
## File
main_bomifa.py: Examples of BOMIFA for classification tasks
training_pipeline.py:'BOMIFA's core end-to-end multi-omics pipeline
models.py: BOMIFA model
train_test.py: Training and testing functions
attention_modules.py: Graph neural network (GNN) layers
contrastive_learning.py: Contrastive learning losses and modules for enhanced representation learning.
cox_loss.py: Cox loss function for survival analysis tasks.
gnn_modules.py: Graph neural network (GNN) layers
lmf_fusion.py: Low‑rank Multi‑modal Fusion (LMF) or similar fusion modules.
omics_fusion_model.py: Detailed code of LMF
transformer_fusion.py: Detailed code of Single-modal Transformer 
utils.py: Supporting functions
## Requirements
- Python 3.9+
- Tensorflow==2.18+
- Other dependencies listed in `requirements.txt`
```bash
git clone https://github.com/anbai01/BOMIFA.git
cd BOMIFA
pip install -r requirements.txt
```
## Usage
python main_bomifa.py
