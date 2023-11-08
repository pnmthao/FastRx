import torch
import math
import torch.nn as nn

from torch.nn.parameter import Parameter
import torch.nn.functional as F

import einops
from einops import rearrange

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        # x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=1000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.attention = MultiheadAttention(hidden_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x

        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(residual + x)

        residual = x

        x = self.fc(x)
        x = self.dropout2(x)
        x = self.norm2(residual + x)

        return x

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        x = torch.matmul(attention_weights, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        x = self.fc(x)

        return x

class Fastformer(nn.Module):
    def __init__(self, dim = 3, decode_dim = 16):
        super(Fastformer, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias = False)
        self.weight_q = nn.Linear(dim, decode_dim, bias = False)
        self.weight_k = nn.Linear(dim, decode_dim, bias = False)
        self.weight_v = nn.Linear(dim, decode_dim, bias = False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias = False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask = None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # Caculate the global query
        alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim = -1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy = n)
        p = repeat_global_query * key
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim = -1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)

        # key-value
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result