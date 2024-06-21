import torch
import torch.nn as nn

from causal_self_attention import CausalSelfAttention
from mlp import MLP

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # incl residual pathway
        x = x + self.mlp(self.ln_2(x))
        return x