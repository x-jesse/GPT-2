from dataclasses import dataclass

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