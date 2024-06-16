import tiktoken
import torch

class DataLoaderLite:
    def __init__(self, B, T, filename):
        self.B = B
        self.T = T

        with open(filename) as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens // (B * T))} batches")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += 1

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y