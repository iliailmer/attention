import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def read_data(path: str = "tinyshakespeare.txt"):
    with open(path) as f:
        data = f.read()
    return data


def tinyshakespeare_batch(tokens, block_size, batch_size):
    _idx = torch.randint(len(tokens) - block_size, size=(batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in _idx])
    y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in _idx])
    return x, y


def tinyshakespeare_batch_words(tokens, block_size, batch_size):
    _idx = torch.randint(len(tokens) - block_size, size=(batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in _idx])
    y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in _idx])
    return x, y
