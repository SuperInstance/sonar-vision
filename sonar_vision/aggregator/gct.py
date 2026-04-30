"""Streaming GCT (Geometric Context Transformer) aggregator for sonar.

Adapted from LingBot-Map's streaming temporal aggregator.
Processes sequences of sonar patch tokens with causal temporal attention,
KV caching for efficient online inference, sliding-window attention over
time, 3D RoPE over (depth, bearing, time), and grouped-query attention.

Typical usage (streaming, one frame at a time):

    >>> aggregator = StreamingGCTAggregator(embed_dim=1024)
    >>> aggregator.init_cache(batch_size=1)
    >>> for sonar_frame in sonar_stream:
    ...     tokens, _ = encoder(sonar_frame)          # (1, 127, 1024)
    ...     out, cache = aggregator(tokens)            # (1, 127, 1024)
    ...     print(aggregator.cache_size())             # frames cached so far
"""

import math
from typing import Any, Dict, Optional, Tuple

# Type alias used by the pipeline for explicit cache typing.
KVCache = Dict[str, Any]

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_3d_rope(
    x: torch.Tensor,
    pos_t: torch.Tensor,
    pos_d: torch.Tensor,
    pos_b: torch.Tensor,
    base_t: float = 10000.0,
    base_d: float = 100.0,
    base_b: float = 100.0,
) -> torch.Tensor:
    """Apply 3D rotary positional embedding over (time, depth, bearing).

    Rotates pairs of dimensions in the last axis using a combined angle
    ``theta = pos_t * freq_t + pos_d * freq_d + pos_b * freq_b``.
    The frequency schedule follows standard RoPE with per-axis bases so
    that depth/bearing (small ranges) and time (long range) each receive
    appropriate angular scales.

    Args:
        x: Tensor of shape ``(batch, seq_len, num_heads, head_dim)``.
        pos_t: Time positions, shape ``(seq_len,)``.
        pos_d: Depth patch positions, shape ``(seq_len,)``.
        pos_b: Bearing patch positions, shape ``(seq_len,)``.
        base_t: RoPE base for the time dimension.
        base_d: RoPE base for the depth dimension.
        base_b: RoPE base for the bearing dimension.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    B, seq_len, num_heads, head_dim = x.shape

    # Merge batch and head dims so seq_len is the leading axis.
    x_reshaped = x.transpose(1, 2).reshape(B * num_heads, seq_len, head_dim)
    x_pairs = x_reshaped.float().reshape(B * num_heads, seq_len, head_dim // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]

    dim_idx = torch.arange(head_dim // 2, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (base_t ** (2 * dim_idx / head_dim))

    # Combined angle contributions from all three axes.
    angles = (
        pos_t[:, None] * inv_freq[None, :]
        + pos_d[:, None] * inv_freq[None, :] * (base_t / base_d)
        + pos_b[:, None] * inv_freq[None, :] * (base_t / base_b)
    )  # (seq_len, head_dim // 2)

    cos = torch.cos(angles)[None, :, :]  # (1, seq_len, head_dim // 2)
    sin = torch.sin(angles)[None, :, :]

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    y = torch.stack([y1, y2], dim=-1).flatten(-2)  # (B*num_heads, seq_len, head_dim)
    y = y.reshape(B, num_heads, seq_len, head_dim).transpose(1, 2)
    return y.type_as(x)


class CausalSlidingWindowGQA(nn.Module):
    """Grouped-query self-attention with causal temporal masking and sliding window.

    The sliding window and causal mask operate on **frame time indices**,
    not raw token indices. This preserves frame-level geometric coherence
    while bounding memory for long sonar streams.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int,
        tokens_per_frame: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.tokens_per_frame = tokens_per_frame

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Optional[Dict[str, torch.Tensor]] = None,
        new_positions: Optional[Dict[str, torch.Tensor]] = None,
        cached_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run attention for new tokens against a (possibly empty) KV cache.

        Args:
            hidden_states: ``(B, new_seq_len, embed_dim)``.
            past_kv: Optional dict with ``'k'`` and ``'v'`` of shape
                ``(B, num_kv_heads, cache_len, head_dim)``.
            new_positions: Dict with ``'t'``, ``'d'``, ``'b'`` tensors of
                shape ``(new_seq_len,)``.
            cached_positions: Dict with ``'t'``, ``'d'``, ``'b'`` tensors of
                shape ``(cache_len,)``.

        Returns:
            ``(output, new_kv)`` where ``output`` has shape
            ``(B, new_seq_len, embed_dim)`` and ``new_kv`` contains the
            rotated keys/values for the **new** tokens only, suitable for
            appending to an external cache.
        """
        B, new_len, _ = hidden_states.shape

        Q = self.q_proj(hidden_states).view(B, new_len, self.num_heads, self.head_dim)
        K_new = self.k_proj(hidden_states).view(B, new_len, self.num_kv_heads, self.head_dim)
        V_new = self.v_proj(hidden_states).view(B, new_len, self.num_kv_heads, self.head_dim)

        # Apply 3D RoPE to new queries and keys.
        Q = apply_3d_rope(Q, new_positions["t"], new_positions["d"], new_positions["b"])
        K_new = apply_3d_rope(
            K_new, new_positions["t"], new_positions["d"], new_positions["b"]
        )

        # Move heads to dimension 1 for batched matmuls.
        Q = Q.transpose(1, 2)  # (B, num_heads, new_len, head_dim)
        K_new = K_new.transpose(1, 2)
        V_new = V_new.transpose(1, 2)

        if past_kv is not None and past_kv["k"] is not None:
            K_past = past_kv["k"]  # (B, num_kv_heads, past_len, head_dim)
            V_past = past_kv["v"]
            past_times = cached_positions["t"]  # (past_len,)
            past_len = past_times.shape[0]

            # Sliding window: find earliest cached frame that still falls
            # inside the window for the newest query time.
            min_time = int(new_positions["t"].max().item()) - (self.window_size - 1)
            start_idx = torch.searchsorted(past_times, min_time, right=False)
            start_idx = min(int(start_idx.item()), past_len)

            if start_idx > 0:
                K_past = K_past[:, :, start_idx:]
                V_past = V_past[:, :, start_idx:]
                past_times_window = past_times[start_idx:]
                past_d_window = cached_positions["d"][start_idx:]
                past_b_window = cached_positions["b"][start_idx:]
            else:
                past_times_window = past_times
                past_d_window = cached_positions["d"]
                past_b_window = cached_positions["b"]

            K = torch.cat([K_past, K_new], dim=2)
            V = torch.cat([V_past, V_new], dim=2)
            pos_t = torch.cat([past_times_window, new_positions["t"]])
            pos_d = torch.cat([past_d_window, new_positions["d"]])
            pos_b = torch.cat([past_b_window, new_positions["b"]])
        else:
            K = K_new
            V = V_new
            pos_t = new_positions["t"]
            pos_d = new_positions["d"]
            pos_b = new_positions["b"]

        # GQA: repeat KV heads to match the number of query heads.
        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(repeat_factor, dim=1)
            V = V.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal temporal mask: a token may only attend to tokens whose
        # frame time is less than or equal to its own.
        mask = pos_t[None, :] <= new_positions["t"][:, None]  # (new_len, cache_len)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Guard against all-"-inf" rows (should not happen in normal use).
        attn = torch.nan_to_num(attn, 0.0)

        out = torch.matmul(attn, V)  # (B, num_heads, new_len, head_dim)
        out = out.transpose(1, 2).reshape(B, new_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        # Return only the new K/V (already rotated) for cache updating.
        return out, {"k": K_new, "v": V_new}


class GCTLayer(nn.Module):
    """Single transformer layer: pre-norm attention + pre-norm FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int,
        tokens_per_frame: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSlidingWindowGQA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            window_size=window_size,
            tokens_per_frame=tokens_per_frame,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Optional[Dict[str, torch.Tensor]] = None,
        new_positions: Optional[Dict[str, torch.Tensor]] = None,
        cached_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Layer forward with residual connections."""
        attn_out, new_kv = self.attn(
            self.norm1(hidden_states),
            past_kv=past_kv,
            new_positions=new_positions,
            cached_positions=cached_positions,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states, new_kv


class StreamingGCTAggregator(nn.Module):
    """Streaming geometric context transformer for sonar token aggregation.

    Processes sequences of sonar feature tokens with causal temporal
    attention, KV caching for efficient online inference, sliding-window
    attention over time, 3D RoPE over ``(depth, bearing, time)``, and
    grouped-query attention for memory efficiency.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of query attention heads.
        num_layers: Number of transformer layers.
        window_size: Attention sliding window size in **frames**.
        gqa_ratio: Ratio of query heads to KV heads
            (``num_kv_heads = num_heads // gqa_ratio``).
        max_cache_seq: Maximum number of frames to retain in the KV cache.
        num_patches_v: Number of depth patches per sonar sweep.
        num_patches_h: Number of bearing patches per sonar sweep.
        ffn_dim: Hidden dimension of the feed-forward network. Defaults
            to ``4 * embed_dim``.
        dropout: Dropout probability (unused in current implementation).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 6,
        window_size: int = 32,
        gqa_ratio: int = 4,
        max_cache_seq: int = 512,
        num_patches_v: Optional[int] = None,
        num_patches_h: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_bearing: Optional[int] = None,
        patch_size: int = 14,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.gqa_ratio = gqa_ratio
        self.max_cache_seq = max_cache_seq

        # Allow num_patches to be specified directly or inferred from
        # sonar geometry (max_depth, max_bearing) and patch_size.
        if num_patches_v is not None:
            self.num_patches_v = num_patches_v
        elif max_depth is not None:
            self.num_patches_v = max_depth // patch_size
        else:
            self.num_patches_v = 14

        if num_patches_h is not None:
            self.num_patches_h = num_patches_h
        elif max_bearing is not None:
            self.num_patches_h = max_bearing // patch_size
        else:
            self.num_patches_h = 9

        self.tokens_per_frame = self.num_patches_v * self.num_patches_h + 1  # +1 scale token
        self.num_kv_heads = num_heads // gqa_ratio
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = ffn_dim or 4 * embed_dim
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [
                GCTLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    window_size=window_size,
                    tokens_per_frame=self.tokens_per_frame,
                    ffn_dim=self.ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Precompute spatial positions for tokens within a single frame.
        # Token 0 is the scale token; remaining tokens are patches in
        # row-major order (depth slowest, bearing fastest).
        self.register_buffer(
            "frame_pos_d",
            torch.zeros(self.tokens_per_frame, dtype=torch.long),
        )
        self.register_buffer(
            "frame_pos_b",
            torch.zeros(self.tokens_per_frame, dtype=torch.long),
        )
        for i in range(self.tokens_per_frame - 1):
            d = i // self.num_patches_h
            b = i % self.num_patches_h
            self.frame_pos_d[i + 1] = d
            self.frame_pos_b[i + 1] = b

        # Internal cache used when past_cache is not provided explicitly.
        self._cache: Optional[Dict[str, Any]] = None

    def _build_positions(
        self, num_new_frames: int, start_time: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Construct 3D position tensors for incoming tokens.

        Returns a dict with keys ``'t'``, ``'d'``, ``'b'`` each of shape
        ``(num_new_frames * tokens_per_frame,)``.
        """
        pos_t = torch.arange(
            start_time, start_time + num_new_frames, device=device
        )
        pos_t = pos_t.repeat_interleave(self.tokens_per_frame)
        pos_d = self.frame_pos_d.repeat(num_new_frames).to(device)
        pos_b = self.frame_pos_b.repeat(num_new_frames).to(device)
        return {"t": pos_t, "d": pos_d, "b": pos_b}

    def init_cache(self, batch_size: int) -> Dict[str, Any]:
        """Initialize an empty KV cache for online inference.

        The returned cache dict can be passed to :meth:`forward` as
        ``past_cache``, or the module will keep and update its own
        internal cache automatically when ``past_cache=None``.

        Args:
            batch_size: Batch size for which the cache is allocated.

        Returns:
            Empty cache dictionary.
        """
        cache: Dict[str, Any] = {
            "kv": [
                {"k": None, "v": None} for _ in range(self.num_layers)
            ],
            "positions": {"t": None, "d": None, "b": None},
            "num_frames": 0,
            "next_time": 0,
            "batch_size": batch_size,
        }
        self._cache = cache
        return cache

    def cache_size(self) -> int:
        """Return the number of cached frames (time steps / sequences).

        Returns ``0`` if no cache has been initialized.
        """
        if self._cache is None:
            return 0
        return self._cache["num_frames"]

    def forward(
        self,
        tokens: torch.Tensor,
        past_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Streaming forward pass.

        Accepts one or more complete sonar frames (each frame comprising
        ``tokens_per_frame`` tokens) and produces aggregated tokens of
        the same shape.

        Args:
            tokens: ``(B, seq_len, embed_dim)`` where ``seq_len`` must be
                an integer multiple of ``tokens_per_frame``.
            past_cache: Optional cache dictionary from a previous call.
                If ``None``, the module uses and updates its internal cache.

        Returns:
            ``(output, new_cache)`` where ``output`` has shape
            ``(B, seq_len, embed_dim)`` and ``new_cache`` is the updated
            KV cache.
        """
        B, seq_len, D = tokens.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {D}"
            )
        if seq_len % self.tokens_per_frame != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be a multiple of "
                f"tokens_per_frame ({self.tokens_per_frame})"
            )

        num_new_frames = seq_len // self.tokens_per_frame

        # Resolve cache to use.
        if past_cache is None:
            if self._cache is None or self._cache["batch_size"] != B:
                past_cache = self.init_cache(B)
            else:
                past_cache = self._cache
        else:
            if past_cache["batch_size"] != B:
                raise ValueError(
                    f"Batch size mismatch: cache has "
                    f"{past_cache['batch_size']}, got {B}"
                )

        # Build 3D positions for the new tokens.
        new_positions = self._build_positions(
            num_new_frames, past_cache["next_time"], tokens.device
        )

        x = tokens
        new_cache_kv: list[Dict[str, torch.Tensor]] = []

        for layer_idx, layer in enumerate(self.layers):
            layer_past = past_cache["kv"][layer_idx]
            if layer_past is not None and layer_past["k"] is not None:
                cached_pos = past_cache["positions"]
            else:
                cached_pos = None

            x, new_kv = layer(
                x,
                past_kv=layer_past,
                new_positions=new_positions,
                cached_positions=cached_pos,
            )
            new_cache_kv.append(new_kv)

        x = self.norm(x)

        # ------------------------------------------------------------------
        # Assemble full cache: concatenate new K/V onto each layer's cache,
        # then optionally drop the oldest frames if we exceed max_cache_seq.
        # ------------------------------------------------------------------
        full_cache_kv: list[Dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            past_kv = past_cache["kv"][layer_idx]
            new_kv = new_cache_kv[layer_idx]
            if past_kv is not None and past_kv["k"] is not None:
                k = torch.cat([past_kv["k"], new_kv["k"]], dim=2)
                v = torch.cat([past_kv["v"], new_kv["v"]], dim=2)
            else:
                k, v = new_kv["k"], new_kv["v"]

            total_tokens = k.shape[2]
            max_tokens = self.max_cache_seq * self.tokens_per_frame
            if total_tokens > max_tokens:
                drop_tokens = total_tokens - max_tokens
                k = k[:, :, drop_tokens:]
                v = v[:, :, drop_tokens:]

            full_cache_kv.append({"k": k, "v": v})

        # Update position tensors similarly.
        if past_cache["positions"]["t"] is not None:
            full_pos_t = torch.cat(
                [past_cache["positions"]["t"], new_positions["t"]]
            )
            full_pos_d = torch.cat(
                [past_cache["positions"]["d"], new_positions["d"]]
            )
            full_pos_b = torch.cat(
                [past_cache["positions"]["b"], new_positions["b"]]
            )
        else:
            full_pos_t = new_positions["t"]
            full_pos_d = new_positions["d"]
            full_pos_b = new_positions["b"]

        total_tokens = full_pos_t.shape[0]
        max_tokens = self.max_cache_seq * self.tokens_per_frame
        if total_tokens > max_tokens:
            drop_tokens = total_tokens - max_tokens
            full_pos_t = full_pos_t[drop_tokens:]
            full_pos_d = full_pos_d[drop_tokens:]
            full_pos_b = full_pos_b[drop_tokens:]

        total_frames = past_cache["num_frames"] + num_new_frames
        if total_frames > self.max_cache_seq:
            total_frames = self.max_cache_seq

        new_cache: Dict[str, Any] = {
            "kv": full_cache_kv,
            "positions": {
                "t": full_pos_t,
                "d": full_pos_d,
                "b": full_pos_b,
            },
            "num_frames": total_frames,
            "next_time": past_cache["next_time"] + num_new_frames,
            "batch_size": B,
        }

        self._cache = new_cache
        return x, new_cache
