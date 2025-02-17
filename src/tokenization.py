from typing import Optional

from torch import Tensor


class Tokenizer:
    """Elementary tokenizer"""

    def __init__(self, from_text: str = "", vocab: Optional[list[str]] = None):
        if vocab:
            self.vocab = vocab
        else:
            if from_text:
                self.chars = sorted(list(set(from_text)))
                self.vocab_size = len(self.chars)
                self.stoi = {ch: i for i, ch in enumerate(self.chars)}
                self.itos = {i: ch for i, ch in enumerate(self.chars)}
            else:
                raise ValueError("No vocab or text provided")

    def encode(self, text: str) -> Tensor:
        return Tensor([self.stoi[c] for c in text]).long()

    def decode(self, tokens: Tensor) -> str:
        return "".join([self.itos[int(token)] for token in tokens])
