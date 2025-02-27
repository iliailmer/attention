import time
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile
from tqdm.auto import tqdm


def flash_attention(q, k, v, mask=None, is_decoder=False, tril=None, block_size=128):
    """
    Flash Attention: Computes attention using memory-efficient tiling.
    """
    batch_size, seq_len, head_dim = q.shape
    output = torch.zeros_like(q)  # Stores final results

    # Iterate through the sequence in blocks
    for i in range(0, seq_len, block_size):
        q_block = q[:, i : i + block_size]  # Load tile of Q
        k_block = k[:, i : i + block_size]  # Load tile of K
        v_block = v[:, i : i + block_size]  # Load tile of V

        # Compute QK^T only for this block
        scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / (head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask[i : i + block_size, i : i + block_size] == 0, float("-inf"))
        elif is_decoder:
            scores = scores.masked_fill(tril[i : i + block_size, i : i + block_size] == 0, float("-inf"))
        # Compute online softmax (normalize per tile)
        scores = scores - scores.max(dim=-1, keepdim=True).values  # Stabilize softmax
        scores = scores.exp()
        scores = scores / scores.sum(dim=-1, keepdim=True)

        # Compute weighted sum
        output[:, i : i + block_size] = torch.matmul(scores, v_block)

    return output


class FlashAttentionMPS(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, is_decoder=False, block_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.block_size = block_size
        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # used in masking)
        self.is_decoder = is_decoder

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = [t.view(batch_size, seq_len, self.num_heads, -1) for t in (q, k, v)]

        attn_output = flash_attention(
            q, k, v, mask=mask, is_decoder=self.is_decoder, tril=self.tril, block_size=self.block_size
        )

        # Merge heads
        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


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
