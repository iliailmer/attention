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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        q = self.query(x)  # B x SL x HS
        k = self.key(x)  # B x SL x HS
        v = self.value(x)  # B x SL x HS
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])  # B x SL x SL
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        elif self.is_decoder:
            _, T, _ = k.shape  # B x SL x HS
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # pyright: ignore
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return scores @ v  # NOTE: Batch x SentenceLength x HeadSize


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(embedding_size, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)  # NOTE: Batch x SentenceLength x HeadSize*num_heads
        x = self.proj(x)  # NOTE: Batch x SentenceLength x embedding_size
        return self.dropout(x)


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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)


class SelfAttentionDecoder(SelfAttention):
    def __init__(self, embedding_size: int, head_size: int, block_size: int):
        super().__init__(embedding_size, head_size, block_size, is_decoder=True)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(x, mask)


class MultiHeadDecoder(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionDecoder(embedding_size, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_enc_heads: int, n_hidden: int = 128):
        super().__init__()
        self.encoder = MultiHeadEncoder(embedding_size, head_size, block_size, num_enc_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, embedding_size),
        )
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.encoder(x, mask)
        x = self.layer_norm(x)
        x = x + self.ffn(x)
        x = self.layer_norm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_heads: int, n_hidden: int = 128):
        super().__init__()
        self.decoder = MultiHeadDecoder(embedding_size, head_size, block_size, num_heads)
        self.mha = MultiHeadAttention(embedding_size, head_size, block_size, num_heads)
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)
        self.layer_norm_3 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, embedding_size),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, enc_x: Optional[Tensor] = None) -> Tensor:
        x = x + self.decoder(x, mask)
        x = self.layer_norm_1(x)
        if enc_x is not None:
            # NOTE: Decoder cross-attention should NOT concatenate x and enc_x.
            # Instead, x (query) attends to enc_x (key & value) in MultiHeadAttention
            # if enc_x is not None:
            # x_residual = x
            # x = self.mha(q=x, k=enc_x, v=enc_x)  # Query attends to encoder output
            # x = self.layer_norm_2(x + x_residual)

            x = torch.cat([x, enc_x], dim=1)
            x = self.mha(x)
            x = x + self.layer_norm_2(x)
        x = x + self.ffn(x)
        x = self.layer_norm_3(x)
        return x


class GPTModel(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_size: int, head_size: int, block_size: int, num_heads: int, num_blocks: int
    ) -> None:
        super().__init__()
        self.t_emembedding = nn.Embedding(vocab_size, embedding_size)
        self.p_embedding = nn.Embedding(block_size, embedding_size)
        self.decoder_blocks = nn.Sequential(
            *[
                DecoderBlock(embedding_size, head_size, block_size, num_heads=num_heads, n_hidden=4 * num_heads)
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.final = nn.Linear(embedding_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        _, T = x.shape
        x = self.t_emembedding(x) + self.p_embedding(torch.arange(T, device=x.device))
        x = self.decoder_blocks(x)
        x = self.layer_norm(x)
        logits = self.final(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens=256, block_size=128):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # last block
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # NOTE: this is the last token in the sequence
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1).to("mps")
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
