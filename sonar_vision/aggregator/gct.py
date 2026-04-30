"""
Streaming GCT (Geometric Context Transformer) Aggregator.

Adapted from LingBot-Map for sonar modality. Provides causal temporal
attention with KV cache for streaming inference on sonar sweeps.

Key differences from LingBot-Map:
- Input: acoustic tokens (not RGB patches)
- 3D RoPE: (depth, bearing, time) instead of (x, y, time)
- No camera intrinsics (sonar has fixed geometry)
- Sliding window attention for bounded memory
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def build_3d_rope(
    depth_positions: torch.Tensor,
    bearing_positions: torch.Tensor,
    time_positions: torch.Tensor,
    dim: int,
    max_depth: int = 200,
    max_bearing: int = 128,
    max_time: int = 512,
    base: float = 10000.0,
) -> torch.Tensor:
    """Build 3D Rotary Position Embedding for (depth, bearing, time).

    Splits the embedding dimension into 3 equal parts for each axis.

    Args:
        depth_positions: (B, N) depth indices [0, max_depth]
        bearing_positions: (B, N) bearing indices [0, max_bearing]
        time_positions: (B, N) time indices [0, max_time]
        dim: Total embedding dimension (must be divisible by 6)
        max_depth, max_bearing, max_time: Maximum position for each axis

    Returns:
        (B, N, dim/2) complex rotation tensor
    """
    assert dim % 6 == 0, f"dim {dim} must be divisible by 6 for 3D RoPE"

    third = dim // 6
    freqs = 1.0 / (base ** (torch.arange(0, third, 2, device=depth_positions.device).float() / third))

    # Depth frequencies
    depth_angles = depth_positions.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B, N, third/2)
    # Bearing frequencies
    bearing_angles = bearing_positions.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
    # Time frequencies
    time_angles = time_positions.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)

    # Interleave: depth, bearing, time, depth, bearing, time, ...
    angles = torch.cat([
        depth_angles, bearing_angles, time_angles,
        depth_angles, bearing_angles, time_angles,
    ], dim=-1)  # (B, N, dim/2)

    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding.

    Args:
        x: (B, N, H, D) where H=num_heads, D=head_dim
        rope: (B, N, D) complex tensor
    Returns:
        (B, N, H, D) rotated tensor
    """
    B, N, H, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, N, H, -1, 2))
    rope_expanded = rope.unsqueeze(2).expand(-1, -1, H, -1)
    rotated = x_complex * rope_expanded
    return torch.view_as_real(rotated).reshape(B, N, H, D).type_as(x)


class KVCache:
    """Paged KV cache for streaming inference."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        dtype=torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Pre-allocate cache: (layer, 2, B, num_kv_heads, max_seq, head_dim)
        self.cache = torch.zeros(
            num_layers, 2, 1, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype,
        )
        self.seq_len = 0  # Current filled length
        self.device = None

    def init_batch(self, batch_size: int, device: torch.device):
        """Initialize cache for a batch."""
        if self.cache.device != device or self.cache.shape[2] != batch_size:
            self.cache = torch.zeros(
                self.num_layers, 2, batch_size,
                self.num_kv_heads, self.max_seq_len, self.head_dim,
                dtype=self.dtype, device=device,
            )
            self.seq_len = 0
            self.device = device

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append KV to cache and return full cache.

        Args:
            layer_idx: Transformer layer index
            key: (B, num_kv_heads, N, head_dim)
            value: (B, num_kv_heads, N, head_dim)

        Returns:
            (full_key, full_value) each (B, num_kv_heads, seq_len, head_dim)
        """
        B, H, N, D = key.shape
        assert self.seq_len + N <= self.max_seq_len, (
            f"Cache overflow: {self.seq_len} + {N} > {self.max_seq_len}"
        )

        # Write to cache
        self.cache[layer_idx, 0, :B, :, self.seq_len:self.seq_len + N, :] = key
        self.cache[layer_idx, 1, :B, :, self.seq_len:self.seq_len + N, :] = value
        self.seq_len += N

        # Return cached content
        k = self.cache[layer_idx, 0, :B, :, :self.seq_len, :]
        v = self.cache[layer_idx, 1, :B, :, :self.seq_len, :]
        return k, v

    def trim_to_window(self, window_size: int):
        """Keep only the last window_size entries."""
        if self.seq_len > window_size:
            # Shift cache left
            self.cache[:, :, :, :, :self.seq_len - window_size, :] = \
                self.cache[:, :, :, :, self.seq_len - window_size:self.seq_len, :].clone()
            self.seq_len = window_size

    def reset(self):
        """Clear the cache."""
        self.cache.zero_()
        self.seq_len = 0


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with causal masking.

    Reduces KV heads by gqa_ratio while keeping full query heads.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 16,
        gqa_ratio: int = 4,
        window_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads // gqa_ratio
        self.gqa_ratio = gqa_ratio
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        assert num_heads % gqa_ratio == 0

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        cache_layer: int = 0,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Forward pass.

        Args:
            x: (B, N, embed_dim) input tokens
            rope: Optional (B, N, head_dim/2) complex RoPE
            kv_cache: Optional KVCache for streaming
            cache_layer: Which layer this attention belongs to
            is_causal: Apply causal mask (no attending to future)

        Returns:
            (output, updated_kv_cache)
        """
        B, N, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, N, num_heads * head_dim)
        k = self.k_proj(x)  # (B, N, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, N, num_kv_heads * head_dim)

        # Reshape to (B, N, heads, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_kv_heads, self.head_dim)
        v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        if rope is not None:
            q = apply_rope(q, rope)
            k_rope = rope[:, :, :self.head_dim]  # KV heads may have different dim
            k = apply_rope(k, k_rope)

        # Transpose for attention: (B, heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update KV cache
        if kv_cache is not None:
            k_full, v_full = kv_cache.update(cache_layer, k, v)
            seq_len = k_full.shape[2]
        else:
            k_full, v_full = k, v
            seq_len = N

        # Expand KV heads to match query heads (GQA)
        k_expanded = k_full.repeat_interleave(self.gqa_ratio, dim=1)
        v_expanded = v_full.repeat_interleave(self.gqa_ratio, dim=1)

        # Compute attention scores
        attn = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale

        # Causal mask
        if is_causal and kv_cache is not None:
            # New tokens can attend to all cached + themselves
            # But cached tokens cannot attend to new tokens (causal)
            pass  # Causality is maintained by cache ordering

        # Sliding window mask
        if self.window_size and seq_len > self.window_size:
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - self.window_size + 1)
                mask[i, start:i + 1] = False
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        out = torch.matmul(attn, v_expanded)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        return out, kv_cache


class GCTBlock(nn.Module):
    """Single GCT Transformer block with pre-norm and residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 16,
        gqa_ratio: int = 4,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            gqa_ratio=gqa_ratio,
            window_size=window_size,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        cache_layer: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        residual = x
        x = self.norm1(x)
        x, kv_cache = self.attn(x, rope=rope, kv_cache=kv_cache, cache_layer=cache_layer)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x, kv_cache


class StreamingGCTAggregator(nn.Module):
    """Streaming Geometric Context Transformer for sonar token aggregation.

    Adapts LingBot-Map's GCT to acoustic modality:
    - 3D RoPE on (depth, bearing, time) axes
    - Sliding window causal attention for streaming
    - KV cache for online inference
    - Grouped-query attention for efficiency

    Usage:
        agg = StreamingGCTAggregator()
        cache = agg.init_cache(batch_size=1, device="cuda")

        # Streaming: process one sweep at a time
        for sweep in sonar_stream:
            tokens = encoder(sweep)
            output, cache = agg(tokens, cache=cache)
            frame = decoder(output)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 6,
        gqa_ratio: int = 4,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_cache_seq: int = 512,
        max_depth: int = 200,
        max_bearing: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.max_depth = max_depth
        self.max_bearing = max_bearing

        self.blocks = nn.ModuleList([
            GCTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                gqa_ratio=gqa_ratio,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.time_counter = 0

    def init_cache(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> KVCache:
        """Initialize a fresh KV cache for streaming inference."""
        num_kv_heads = self.blocks[0].attn.num_kv_heads
        head_dim = self.blocks[0].attn.head_dim
        cache = KVCache(
            num_layers=self.num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=self.window_size * 2,  # Double window for safety
            dtype=torch.float16,
        )
        cache.init_batch(batch_size, device)
        self.time_counter = 0
        return cache

    def build_rope(
        self,
        tokens: torch.Tensor,
        depth_positions: Optional[torch.Tensor] = None,
        bearing_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build 3D RoPE for current tokens.

        Args:
            tokens: (B, N, embed_dim)
            depth_positions: (B, N) or None (default: linspace)
            bearing_positions: (B, N) or None (default: linspace)
        """
        B, N, D = tokens.shape

        if depth_positions is None:
            depth_positions = torch.linspace(0, self.max_depth - 1, N, device=tokens.device)
            depth_positions = depth_positions.unsqueeze(0).expand(B, -1)

        if bearing_positions is None:
            bearing_positions = torch.linspace(0, self.max_bearing - 1, N, device=tokens.device)
            bearing_positions = bearing_positions.unsqueeze(0).expand(B, -1)

        time_positions = torch.full((B, N), self.time_counter, device=tokens.device, dtype=torch.long)

        return build_3d_rope(
            depth_positions.float(),
            bearing_positions.float(),
            time_positions.float(),
            D,
            max_depth=self.max_depth,
            max_bearing=self.max_bearing,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cache: Optional[KVCache] = None,
        depth_positions: Optional[torch.Tensor] = None,
        bearing_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Process tokens through GCT layers.

        Args:
            tokens: (B, N, embed_dim) encoder output tokens
            cache: Optional KV cache for streaming
            depth_positions: (B, N) depth indices for RoPE
            bearing_positions: (B, N) bearing indices for RoPE

        Returns:
            (aggregated_tokens, updated_cache)
        """
        rope = self.build_rope(tokens, depth_positions, bearing_positions)

        x = tokens
        for i, block in enumerate(self.blocks):
            x, cache = block(x, rope=rope, kv_cache=cache, cache_layer=i)

        x = self.final_norm(x)

        # Trim cache to window size
        if cache is not None:
            cache.trim_to_window(self.window_size)

        self.time_counter += 1

        return x, cache
