import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataloader import DataLoaderLite


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

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # attn w normalization
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # replace 0s in mask with -inf as prep for softmax
        # att = F.softmax(att, dim=-1)
        # y = att @ v # combine attention with values: (B, nh, T, T) -> (B, nh, T, hs)

        # flash attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # concat all head outputs

        y = self.c_proj(y)

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

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

@dataclass
class GPTConfig:
    # params from GPT-2 paper
    block_size: int = 1024
    n_layer: int = 12
    # number of tokens: 50k BPE merges + 256 byte tokens + 1 <|endoftext|> special char = 50257
    # padded to 50304 bc multiple of 64
    vocab_size: int = 50304 
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # implements analagous, decoder-only transformer from OpenAI GPT2 paper
        # (minor differences discussed later)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        # final layer to project back to dimension of vocab_size
        # no bias needed bc after layernorm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

max_length = 30
num_return_sequences = 5
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
# model.eval()
model.to(device)
print('starting')


B, T = 4, 32
train_loader = DataLoaderLite(B, T, filename='input.txt')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys
sys.exit(0)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # clamps anything below top 50 to 0 so it will never be sampled
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

