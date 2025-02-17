import math
from typing import Optional

import torch  # noqa: F401
from torch import Tensor, nn
from torch.nn import functional as F


# NOTE: Batch x SentenceLength => Embedding => Batch x SentenceLength x EmbeddingSize
class SelfAttention(nn.Module):
    """Scaled dot product attention from "Attention is all you need" paper"""

    def __init__(self, embedding_size: int, head_size: int, block_size: int, is_decoder=False) -> None:
        super().__init__()
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.is_decoder = is_decoder
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # used in masking)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        q = self.query(x)  # B x SL x HS
        k = self.key(x)  # B x SL x HS
        v = self.value(x)  # B x SL x HS
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])  # B x SL x SL
        scores = F.softmax(scores, dim=0)
        if mask is not None:
            scores = scores + scores.masked_fill(mask == 0, Tensor("-inf"))
        elif self.is_decoder:
            T = k.shape[-1]  # SL
            scores = scores + scores.masked_fill(self.tril[:T, :T] == 0, Tensor("-inf"))
        scores = F.softmax(scores, dim=0)
        return scores @ v  # NOTE: Batch x SentenceLength x HeadSize


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(embedding_size, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_size)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)  # NOTE: Batch x SentenceLength x HeadSize*num_heads
        x = self.proj(x)  # NOTE: Batch x SentenceLength x embedding_size
        return x


class SelfAttentionEncoder(SelfAttention):
    def __init__(self, embedding_size: int, head_size: int, block_size: int):
        super().__init__(embedding_size, head_size, block_size, is_decoder=False)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(x, mask)


class MultiHeadEncoder(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionEncoder(embedding_size, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_size)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        return x


class SelfAttentionDecoder(SelfAttention):
    def __init__(self, embedding_size: int, head_size: int, block_size: int):
        super().__init__(embedding_size, head_size, block_size, is_decoder=True)


class MultiHeadDecoder(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionDecoder(embedding_size, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_size)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        x = self.proj(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        head_size: int,
        block_size: int,
        num_enc_heads: int,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = MultiHeadEncoder(embedding_size, head_size, block_size, num_enc_heads)
        self.ffn = nn.Sequential(
            nn.Linear(head_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, head_size),
        )
        self.layer_norm = nn.LayerNorm([head_size])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.encoder(x, mask)
        x = x + self.layer_norm(x)
        x = self.ffn(x)
        x = x + self.layer_norm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        head_size: int,
        block_size: int,
        num_heads: int,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.decoder = MultiHeadDecoder(embedding_size, head_size, block_size, num_heads)
        self.mha = MultiHeadAttention(embedding_size, head_size, block_size, num_heads)
        self.layer_norm = nn.LayerNorm(embedding_size)

        self.ffn = nn.Sequential(
            nn.Linear(head_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, head_size),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, enc_x: Optional[Tensor] = None) -> Tensor:
        x = self.decoder(x, mask)
        x = x + self.layer_norm(x)
        if enc_x is not None:
            x = torch.cat([x, enc_x], dim=1)
        x = self.mha(x)
        x = x + self.layer_norm(x)
        x = self.ffn(x)
        x = x + self.layer_norm(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        head_size: int,
        block_size: int,
        num_encoders: int,
        num_decoders: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(embedding_size, head_size, block_size, 2, 128) for _ in range(num_encoders)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embedding_size, head_size, block_size, 2, 128) for _ in range(num_decoders)]
        )
