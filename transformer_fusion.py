import torch.nn as nn
class MultiOmicsFusionModel(nn.Module):
    def __init__(self,d_model, nhead, d_ff, num_layers,dropout):
        super(MultiOmicsFusionModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(d_model, 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    def forward(self, in_list):
        x1 = self.transformer_encoder(in_list)
        return x1
