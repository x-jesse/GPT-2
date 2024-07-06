import argparse
import tiktoken
import numpy as np
import os
from datasets import load_dataset
import multiprocessing as mp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--shard-size', default=int(1e8))
parser.add_argument('-t', '--is-test', action='store_true')

args = parser.parse_args()

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "openwebtext-10k")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

"""
Open Web Text contains many rows of text strings sourced from the internet
"""

if args.is_test:
    ds = load_dataset("stas/openwebtext-10k", split='train')

enc = tiktoken.get_encoding('gpt2')
# For our test data, sharding isn't necessary since ds == 10k (small)
# But it will for a production run, where ds > 10B (large)
shard_size = args.shard_size

# tokenize our texts
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(data):
    tokens = enc.encode_ordinary(data['text']) + [eot]
    tokens = np.array(tokens)
    tokens = tokens.astype(np.uint16)
    return tokens

num_procs = os.cpu_count() // 2

if __name__ == '__main__':
    with mp.Pool(processes=num_procs) as pool:

        """
        Data sharding via multiprocessing:
            - Chunksize separates our data into batches of 16, and gives each worker a batch
            - Each worker loops over the batch and tokenizes each string of text 

        """
        count = 0
        max_iter = 1
        cur_size = 0
        shard_index = 0
        # progress bar
        prog = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        # data buffer for our shard
        buffer = np.empty((shard_size,), dtype=np.uint16)

        for tokens in pool.imap(tokenize, ds, chunksize=16): # imap is just lazy map()

            if len(tokens) + cur_size > shard_size:
                new_shard = os.path.join(DATA_CACHE_DIR, f"openwebtext_{shard_index:06d}")
                cur_size += len(tokens)
                # gets whatever fits in the current shard
                remainder = shard_size - cur_size
                # update progress
                prog.update(remainder)

                buffer[cur_size:cur_size + remainder] = tokens[:remainder]
                np.save(new_shard, buffer)

                shard_index += 1
                # populate the next shard with the leftovers of the current doc
                # technically unnecessary to clear the buffer, but makes logical sense
                buffer = np.empty((shard_size,), dtype=np.uint16)
                buffer[:len(tokens) - remainder] = tokens[remainder:]
                # resets the cur_size
                cur_size = len(tokens) - remainder
                prog = None

            else:
                if prog is None:
                    prog = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                prog.update(len(tokens))

                buffer[cur_size:cur_size + len(tokens)] = tokens
                cur_size += len(tokens)
                prog.update(len(tokens))

        # write any remaining tokens
        if cur_size != 0:
            new_shard = os.path.join(DATA_CACHE_DIR, f"openwebtext_{shard_index:06d}")
            np.save(new_shard, buffer[:cur_size])

