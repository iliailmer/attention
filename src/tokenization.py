from typing import Optional

from torch import Tensor

SPECIAL_TOKENS = ": $.;',\n!&?-"


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
                self.words = sorted(self._tokenize(from_text))
                self.vocab_size = len(self.words)
                self.stoi = {ch: i for i, ch in enumerate(self.words)}
                self.itos = {i: ch for i, ch in enumerate(self.words)}
            else:
                raise ValueError("No vocab or text provided")
        # self.stoi["\n"] = self.vocab_size + 1
        # self.itos[self.vocab_size + 1] = "\n"
        # self.vocab_size += 1

        self.stoi["<NONE>"] = self.vocab_size + 1
        self.itos[self.vocab_size + 1] = "<NONE>"
        self.vocab_size += 1

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text while preserving newlines as separate tokens.
        """
        tokens = list()
        word = ""
        for char in text:
            if char in SPECIAL_TOKENS:
                if word:
                    tokens.append(word)
                    word = ""
                tokens.append(char)
            else:
                word += char
        if word:
            tokens.append(word)
        return tokens

    def encode(self, text: str) -> Tensor:
        text_ = self._tokenize(text)
        return Tensor([self.stoi.get(c, self.stoi["<NONE>"]) for c in text_]).long()

    def decode(self, tokens) -> str:
        return "".join([self.itos[int(token)] for token in tokens])
