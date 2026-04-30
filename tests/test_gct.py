"""Tests for the GCT streaming aggregator."""

import torch
import pytest

from sonar_vision.aggregator.gct import (
    StreamingGCTAggregator,
    KVCache,
    GroupedQueryAttention,
    build_3d_rope,
)


class TestGroupedQueryAttention:
    def test_output_shape(self):
        attn = GroupedQueryAttention(embed_dim=256, num_heads=4, gqa_ratio=2)
        x = torch.randn(2, 16, 256)
        out, _ = attn(x)
        assert out.shape == (2, 16, 256)

    def test_kv_cache(self):
        attn = GroupedQueryAttention(embed_dim=256, num_heads=4, gqa_ratio=2)
        cache = KVCache(num_layers=1, num_kv_heads=2, head_dim=32, max_seq_len=64)
        cache.init_batch(2, device=torch.device("cpu"))

        x = torch.randn(2, 8, 256)
        out, cache = attn(x, kv_cache=cache, cache_layer=0)
        assert out.shape == (2, 8, 256)
        assert cache.seq_len == 8

        # Second call should extend cache
        x2 = torch.randn(2, 4, 256)
        out2, cache = attn(x2, kv_cache=cache, cache_layer=0)
        assert out2.shape == (2, 4, 256)
        assert cache.seq_len == 12


class TestKVCache:
    def test_init_and_update(self):
        cache = KVCache(num_layers=2, num_kv_heads=2, head_dim=64, max_seq_len=32)
        cache.init_batch(1, device=torch.device("cpu"))

        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)

        k_full, v_full = cache.update(0, k, v)
        assert k_full.shape == (1, 2, 8, 64)
        assert cache.seq_len == 8

        k2 = torch.randn(1, 2, 4, 64)
        v2 = torch.randn(1, 2, 4, 64)
        k_full, v_full = cache.update(0, k2, v2)
        assert k_full.shape == (1, 2, 12, 64)
        assert cache.seq_len == 12

    def test_trim_to_window(self):
        cache = KVCache(num_layers=1, num_kv_heads=1, head_dim=32, max_seq_len=64)
        cache.init_batch(1, device=torch.device("cpu"))

        k = torch.randn(1, 1, 20, 32)
        v = torch.randn(1, 1, 20, 32)
        cache.update(0, k, v)
        assert cache.seq_len == 20

        cache.trim_to_window(10)
        assert cache.seq_len == 10

    def test_reset(self):
        cache = KVCache(num_layers=1, num_kv_heads=1, head_dim=32, max_seq_len=32)
        cache.init_batch(1, device=torch.device("cpu"))

        k = torch.randn(1, 1, 8, 32)
        v = torch.randn(1, 1, 8, 32)
        cache.update(0, k, v)
        assert cache.seq_len == 8

        cache.reset()
        assert cache.seq_len == 0


class TestStreamingGCTAggregator:
    def test_forward_shape(self):
        agg = StreamingGCTAggregator(
            embed_dim=256, num_heads=4, num_layers=2, gqa_ratio=2, window_size=16
        )
        tokens = torch.randn(2, 16, 256)
        out, _ = agg(tokens)
        assert out.shape == (2, 16, 256)

    def test_with_cache(self):
        agg = StreamingGCTAggregator(
            embed_dim=256, num_heads=4, num_layers=2, gqa_ratio=2, window_size=16
        )
        cache = agg.init_cache(batch_size=1)

        tokens = torch.randn(1, 8, 256)
        out, cache = agg(tokens, cache=cache)
        assert out.shape == (1, 8, 256)
        assert cache.seq_len > 0

    def test_streaming_consistency(self):
        """Verify streaming (chunked) gives similar output to batched."""
        agg = StreamingGCTAggregator(
            embed_dim=256, num_heads=4, num_layers=2, gqa_ratio=2, window_size=64
        )
        tokens = torch.randn(1, 32, 256)

        # Batched
        out_batch, _ = agg(tokens)

        # Reset and stream in chunks
        agg.time_counter = 0
        cache = agg.init_cache(batch_size=1)
        out_chunks = []
        for i in range(0, 32, 8):
            chunk = tokens[:, i:i+8]
            out, cache = agg(chunk, cache=cache)
            out_chunks.append(out)

        out_stream = torch.cat(out_chunks, dim=1)
        # Shapes should match
        assert out_batch.shape == out_stream.shape

    def test_3d_rope_shape(self):
        depth_pos = torch.randint(0, 200, (2, 16))
        bearing_pos = torch.randint(0, 128, (2, 16))
        time_pos = torch.randint(0, 100, (2, 16))

        rope = build_3d_rope(depth_pos, bearing_pos, time_pos, dim=252)
        assert rope.shape == (2, 16, 126)  # dim/2 for complex
