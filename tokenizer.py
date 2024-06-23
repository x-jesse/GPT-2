import tiktoken
import re

class Tokenizer():
    """BPE tokenizer"""

    def __init__(self, vocab_size) -> None:
        """"""
        self.vocab_size = vocab_size
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merge_keys = {}

        # This regex is copied directly from GPT-4 - should split text the same way
        self.regex = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    
    def _get_freqs(self, ids):
        """"""
        freq = {}
        for pair in zip(ids, ids[1:]):
            freq[pair] = freq.get(pair, 0) + 1
        return freq
    
    def _merge(self, ids, merge_pair, merged_token):
        """"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == merge_pair:
                new_ids.append(merged_token)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def decode(self, ids):
        """"""
        tokens = b"".join([self.vocab[idx] for idx in ids])
        text = tokens.decode('utf-8', errors='replace')
        return text 
    
    def encode(self, text):
        """"""
        tokens = list(text.encode('utf-8'))
        # looks through all pairs in tokens
        # gets the pair that corresponds to the lowest idx in merge_keys
        # if the pair is not in the list of available merges, every value in freq will return inf
        # therefore no merges are left to be made
        while len(tokens) >= 2:
            freq = self.get_freqs(tokens)
            pair = min(freq, key=lambda p : self.merge_keys.get(p, float('inf')))
            if pair not in self.merge_keys:
                break
            idx = self.merge_keys[pair]
            tokens = self.merge(tokens, pair, idx) # list of tokens, pair to merge, idx to merge to
        return tokens

    def train(self, text):
        ids = text.encode('utf-8')
        ids = list(map(int, ids))

        vocab_size = 276
        merge_keys = {}
        for idx in range(256, vocab_size):
            freq = self.get_freqs(ids)
            pair = max(freq, key=freq.get)
            ids = self.merge(ids, pair, idx)
            print(f"merging {pair} into a new token {idx}")
            merge_keys[pair] = idx







