import torch
import torch.nn as nn
import math
from lmf_fusion import LMF_fusion

class MultiOmicsFusionModel(nn.Module):
    def __init__(self, input_dims, d_model, output_dim, rank, num_classes):
        super(MultiOmicsFusionModel, self).__init__()
        self.input_dims = input_dims
        self.d_model = d_model
        self.lmf = LMF_fusion(input_dims, [d_model, d_model, d_model], [0.1, 0.1, 0.1, 0.1], output_dim, rank, use_softmax=False)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, in_list, W):
        output = self.lmf(in_list[0], in_list[1], in_list[2], W)
        return output
