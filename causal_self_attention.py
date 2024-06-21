import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Defines a class for causal self-attention head. See readme for details."""

    def __init__(self, config):
        """Init layers and defines properties."""
        super().__init__()
        # n_embd must be multiple of head num
        assert config.n_embd % config.n_head == 0
        
        # linear initialization of all tunable params
        # since we use this layer for all query, key, and value params
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # autoregressive mask for tokens (also adds two surrounding dim - equivalent to torch.unsqueeze twice)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)
                                                .view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        """Forward pass through the model."""

        B, T, C = x.size() # batch, time (position),  channel (embedding dimensionality)
        # gets a combined tensor with all parameters for query, key, and value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # split tensor into dims
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # dimensions (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # dimensions (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # dimensions (B, nh, T, hs)

        # flash attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # concat all head outputs

        y = self.c_proj(y)

        return y