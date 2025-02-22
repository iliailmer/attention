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

    def decode(self, tokens) -> str:
        return "".join([self.itos[int(token)] for token in tokens])


class TokenizerByWord:
    """Elementary Tokenizer by Word"""

    def __init__(self, from_text: str = "", vocab: Optional[list[str]] = None):
        if vocab:
            self.vocab = vocab
        else:
            if from_text:
                self.words = sorted(list(set(from_text.replace("\n", " ").split())))
                self.vocab_size = len(self.words)
                self.stoi = {ch: i for i, ch in enumerate(self.words)}
                self.itos = {i: ch for i, ch in enumerate(self.words)}
            else:
                raise ValueError("No vocab or text provided")
        self.stoi["<NONE>"] = self.vocab_size + 1
        self.itos[self.vocab_size + 1] = "<NONE>"
        self.vocab_size += 1

    def encode(self, text: str) -> Tensor:
        text_ = text.replace("\n", " ").split()
        return Tensor([self.stoi.get(c, self.stoi["<NONE>"]) for c in text_]).long()

    def decode(self, tokens) -> str:
        return "".join([self.itos[int(token)] for token in tokens])
