import torch


def read_data(path: str = "tinyshakespeare.txt"):
    with open(path, "r") as f:
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
