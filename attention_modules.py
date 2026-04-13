import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossModalAttention(nn.Module):
    def __init__(self,query_dim, key_value_dim, key_dim, value_dim, num_heads, dropout):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_size = key_dim * num_heads

        # Projection layers
        self.query_proj = nn.Linear(query_dim, key_dim * num_heads)
        self.key_proj = nn.Linear(key_value_dim, key_dim * num_heads)
        self.value_proj = nn.Linear(key_value_dim, value_dim * num_heads)
        self.output_proj = nn.Linear(key_dim * num_heads, query_dim)

        # Initialize weights
        self._initialize_weights()

        # Regularization and normalization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(key_dim)

        # Attention counter for debugging/monitoring
        self.attention_count = 0

    def _initialize_weights(self):
        """Initialize linear layer weights using Xavier uniform initialization."""
        for layer in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query, key, value):

        sequence_length, query_dim = query.size()
        self.attention_count += 1

        # Linear projections with activation
        projected_query = self.activation(self.query_proj(query))
        projected_key = self.activation(self.key_proj(key))
        projected_value = self.activation(self.value_proj(value))

        # Reshape for multi-head attention [batch_size, num_heads, seq_len, dim]
        reshaped_query = projected_query.view(
            1, sequence_length, self.num_heads, self.key_dim
        ).transpose(1, 2)
        reshaped_key = projected_key.view(
            1, sequence_length, self.num_heads, self.key_dim
        ).transpose(1, 2)
        reshaped_value = projected_value.view(
            1, sequence_length, self.num_heads, self.value_dim
        ).transpose(1, 2)

        # Compute attention scores
        scale_factor = math.sqrt(self.key_dim)
        attention_scores = torch.matmul(reshaped_query, reshaped_key.transpose(-1, -2)) / scale_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, reshaped_value)

        # Reshape back to original format
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(1, sequence_length, self.hidden_size)
        attended_values = attended_values.squeeze(0)  # Remove batch dimension

        # Final output projection
        output = self.output_proj(attended_values)

        return output


