import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm


def flash_attention(q: Tensor, k: Tensor, v: Tensor, mask=None, is_decoder=False, tril=None, block_size=128):
    """
    Flash Attention: Computes attention using memory-efficient tiling.
    """
    batch_size, seq_len, head_dim = q.shape
    output = torch.zeros_like(q)  # Stores final results

    for i in range(0, seq_len, block_size):
        q_block = q[:, i : i + block_size]
        k_block = k[:, i : i + block_size]
        v_block = v[:, i : i + block_size]

        # Compute QK^T only for this block
        scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / (head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask[:, i : i + block_size, i : i + block_size] == 0, float("-inf"))
        elif is_decoder and tril is not None:
            if block_size > scores.shape[-1]:
                scores = scores.masked_fill(
                    tril[i : i + scores.shape[-1], i : i + scores.shape[-1]] == 0, float("-inf")
                )
            else:
                scores = scores.masked_fill(tril[i : i + block_size, i : i + block_size] == 0, float("-inf"))
        # Compute online softmax (normalize per tile)
        scores = scores - scores.max(dim=-1, keepdim=True).values  # Stabilize softmax
        scores = scores.exp()
        scores_sum = scores.sum(dim=-1, keepdim=True)
        scores = scores / (scores_sum + 1e-6)  # Avoid division by zero

        # Compute weighted sum
        output[:, i : i + block_size] = torch.matmul(scores, v_block)

    return output


class FlashAttentionMPS(nn.Module):
    def __init__(self, embed_dim, head_size, context_size, is_decoder=False, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.head_size = head_size
        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * head_size, bias=False)
        self.out_proj = torch.nn.Linear(head_size, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.is_decoder = is_decoder

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        attn_output = flash_attention(
            q, k, v, mask=mask, is_decoder=self.is_decoder, tril=self.tril, block_size=self.block_size
        )

        # Merge heads
        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, embed_dim]

        return self.out_proj(attn_output)


# NOTE: Batch x SentenceLength => Embedding => Batch x SentenceLength x EmbeddingSize
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, head_size: int, block_size: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [FlashAttentionMPS(embedding_size, head_size, context_size=block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)  # NOTE: Batch x SentenceLength x HeadSize*num_heads
        x = self.proj(x)  # NOTE: Batch x SentenceLength x embedding_size
        return self.dropout(x)


class SelfAttentionEncoder(FlashAttentionMPS):
    def __init__(self, embedding_size: int, head_size: int, block_size: int):
        super().__init__(embedding_size, head_size, context_size=block_size, is_decoder=False)

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


class SelfAttentionDecoder(FlashAttentionMPS):
    def __init__(self, embedding_size: int, head_size: int, block_size: int):
        super().__init__(embedding_size, head_size, context_size=block_size, is_decoder=True)

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


# benchmarking routines
def regular_attention(q, k, v):
    # simplified attention step, no masking
    scores = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(k.shape[-1]))  # B x SL x SL
    scores = F.softmax(scores, dim=-1)
    return scores @ v


def benchmark_fa(num_iter=1000):
    # Simulate input tensors
    batch_size, seq_len, head_dim = 1, 2**15, 64
    device = "mps"

    fa_times = []

    # Warm-up runs (5 iterations to trigger JIT optimization)
    for _ in range(5):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)
        _ = flash_attention(q, k, v, block_size=256)

    torch.mps.synchronize()  # Ensure GPU is ready

    for _ in tqdm(range(num_iter), desc="Benchmarking"):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)

        # Measure Flash Attention
        torch.mps.synchronize()
        start_time = time.perf_counter()
        _ = flash_attention(q, k, v, block_size=256)
        torch.mps.synchronize()
        end_time = time.perf_counter()
        fa_times.append(end_time - start_time)

    print(f"Average Flash Attention time: {np.mean(fa_times):.6f} ± {np.std(fa_times):.6f} s")


def benchmark_a(num_iter=1000):
    # Simulate input tensors
    batch_size, seq_len, head_dim = 1, 2**15, 64
    device = "mps"

    a_times = []

    # Warm-up runs (5 iterations to trigger JIT optimization)
    for _ in range(5):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)
        _ = regular_attention(q, k, v)

    torch.mps.synchronize()  # Ensure GPU is ready
    for _ in tqdm(range(num_iter), desc="Benchmarking"):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)

        # Measure regular attention
        torch.mps.synchronize()  # Sync before timing
        start_time = time.perf_counter()
        _ = regular_attention(q, k, v)
        torch.mps.synchronize()  # Sync after timing
        end_time = time.perf_counter()
        a_times.append(end_time - start_time)

    print(f"Average Regular Attention time: {np.mean(a_times):.6f} ± {np.std(a_times):.6f} s")
