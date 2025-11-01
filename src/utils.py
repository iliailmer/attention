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


def tinyshakespeare_batch(data, block_size, batch_size, tokenizer):
    x = []
    y = []
    _idx = torch.randint(len(data) - block_size, size=(batch_size,))
    for i in _idx:
        x.append(tokenizer.encode(data[i : i + block_size]))
        y.append(tokenizer.encode(data[i + 1 : i + 1 + block_size]))
    return torch.stack(x), torch.stack(y)


def tinyshakespeare_batch_words(data, block_size, batch_size, tokenizer):
    # data = data.replace("\n", " ")
    x = []
    y = []
    words = tokenizer._tokenize(data)

    _idx = torch.randint(len(words) - block_size, size=(batch_size,))
    for i in _idx:
        x_str = "".join(words[i : i + block_size])
        y_str = "".join(words[i + 1 : i + 1 + block_size])
        x.append(tokenizer.encode(x_str))
        y.append(tokenizer.encode(y_str))
    return torch.stack(x), torch.stack(y)
