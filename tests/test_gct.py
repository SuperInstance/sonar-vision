"""Tests for the GCT streaming aggregator.

Source notes:
  KVCache is a type alias: Dict[str, Any] — a plain dict, not a class.
  StreamingGCTAggregator.forward signature: forward(self, tokens, past_cache=None).
  cache["num_frames"] / cache["next_time"] are the relevant state keys.
  apply_3d_rope is the only rope helper exported from the module.
"""

import torch
import pytest
from sonar_vision.aggregator.gct import (
    StreamingGCTAggregator,
    KVCache,        # type alias = Dict[str, Any]
    apply_3d_rope,
)

# Small shared config: num_patches_v=2, num_patches_h=2 → tokens_per_frame=5
_DIM = 64
_HEADS = 4
_KV_RATIO = 2
_LAYERS = 2
_PV = 2
_PH = 2
_TPF = _PV * _PH + 1   # 5
_WIN = 4
_MAX_CACHE = 8


def make_agg(**kw) -> StreamingGCTAggregator:
    defaults = dict(
        embed_dim=_DIM, num_heads=_HEADS, num_layers=_LAYERS,
        gqa_ratio=_KV_RATIO, window_size=_WIN,
        num_patches_v=_PV, num_patches_h=_PH, max_cache_seq=_MAX_CACHE,
    )
    defaults.update(kw)
    return StreamingGCTAggregator(**defaults)


class TestApply3DRope:
    def test_output_shape(self):
        B, seq, heads, hd = 2, 10, 4, 16
        x = torch.randn(B, seq, heads, hd)
        pos = torch.arange(seq, dtype=torch.float32)
        y = apply_3d_rope(x, pos_t=pos, pos_d=pos, pos_b=pos)
        assert y.shape == x.shape

    def test_dtype_preserved(self):
        x = torch.randn(1, 5, 4, 16)
        pos = torch.arange(5, dtype=torch.float32)
        y = apply_3d_rope(x, pos_t=pos, pos_d=pos, pos_b=pos)
        assert y.dtype == x.dtype

    def test_zero_positions_identity(self):
        x = torch.randn(1, 4, 2, 8)
        pos = torch.zeros(4, dtype=torch.float32)
        y = apply_3d_rope(x, pos_t=pos, pos_d=pos, pos_b=pos)
        assert torch.allclose(y.float(), x.float(), atol=1e-5)

    def test_3d_rope_shape(self):
        B, seq, heads, hd = 3, 8, 4, 16
        x = torch.randn(B, seq, heads, hd)
        pos_t = torch.randint(0, 100, (seq,)).float()
        pos_d = torch.randint(0, 14, (seq,)).float()
        pos_b = torch.randint(0, 9, (seq,)).float()
        y = apply_3d_rope(x, pos_t=pos_t, pos_d=pos_d, pos_b=pos_b)
        assert y.shape == (B, seq, heads, hd)


class TestGCTForwardPass:
    def test_forward_shape(self):
        agg = make_agg()
        out, _ = agg(torch.randn(2, _TPF, _DIM))
        assert out.shape == (2, _TPF, _DIM)

    def test_forward_finite(self):
        agg = make_agg()
        out, _ = agg(torch.randn(1, _TPF, _DIM))
        assert torch.isfinite(out).all()

    def test_wrong_embed_dim_raises(self):
        agg = make_agg()
        with pytest.raises(ValueError, match="embed_dim"):
            agg(torch.randn(1, _TPF, _DIM + 1))

    def test_wrong_seq_len_raises(self):
        agg = make_agg()
        with pytest.raises(ValueError, match="tokens_per_frame"):
            agg(torch.randn(1, _TPF + 1, _DIM))

    def test_multi_frame_input(self):
        agg = make_agg()
        out, cache = agg(torch.randn(1, 2 * _TPF, _DIM))
        assert out.shape == (1, 2 * _TPF, _DIM)
        assert cache["num_frames"] == 2

    def test_streaming_consistency_shapes(self):
        agg = make_agg()
        tokens = torch.randn(1, 4 * _TPF, _DIM)
        out_batch, _ = agg(tokens)
        cache = agg.init_cache(batch_size=1)
        out_chunks = []
        for i in range(0, 4 * _TPF, _TPF):
            out, cache = agg(tokens[:, i:i+_TPF], past_cache=cache)
            out_chunks.append(out)
        out_stream = torch.cat(out_chunks, dim=1)
        assert out_batch.shape == out_stream.shape


class TestKVCacheLifecycle:
    def test_init_and_update(self):
        agg = make_agg()
        cache = agg.init_cache(batch_size=1)
        assert cache["num_frames"] == 0
        assert cache["next_time"] == 0
        for layer_kv in cache["kv"]:
            assert layer_kv["k"] is None

        tokens = torch.randn(1, _TPF, _DIM)
        _, cache = agg(tokens, past_cache=cache)
        assert cache["num_frames"] == 1
        assert cache["next_time"] == 1
        for layer_kv in cache["kv"]:
            assert layer_kv["k"] is not None
            assert layer_kv["k"].shape[2] == _TPF

        _, cache = agg(tokens, past_cache=cache)
        assert cache["num_frames"] == 2
        for layer_kv in cache["kv"]:
            assert layer_kv["k"].shape[2] == 2 * _TPF

    def test_trim_via_max_cache_seq(self):
        max_cache = 3
        agg = make_agg(max_cache_seq=max_cache, num_layers=1)
        tokens = torch.randn(1, _TPF, _DIM)
        cache = agg.init_cache(1)
        for _ in range(max_cache + 1):
            _, cache = agg(tokens, past_cache=cache)
        assert cache["num_frames"] == max_cache
        assert cache["next_time"] == max_cache + 1
        for layer_kv in cache["kv"]:
            assert layer_kv["k"].shape[2] <= max_cache * _TPF

    def test_reset_via_init_cache(self):
        agg = make_agg()
        tokens = torch.randn(1, _TPF, _DIM)
        cache = agg.init_cache(1)
        _, cache = agg(tokens, past_cache=cache)
        assert agg.cache_size() == 1
        agg.init_cache(batch_size=1)
        assert agg.cache_size() == 0

    def test_cache_size_method(self):
        agg = make_agg()
        assert agg.cache_size() == 0
        _, _ = agg(torch.randn(1, _TPF, _DIM))
        assert agg.cache_size() == 1

    def test_batch_size_mismatch_raises(self):
        agg = make_agg()
        cache = agg.init_cache(batch_size=2)
        with pytest.raises(ValueError, match="[Bb]atch"):
            agg(torch.randn(1, _TPF, _DIM), past_cache=cache)


class TestSlidingWindowBehavior:
    def test_cache_bounded(self):
        max_cache = 4
        agg = make_agg(max_cache_seq=max_cache)
        tokens = torch.randn(1, _TPF, _DIM)
        cache = agg.init_cache(1)
        for _ in range(max_cache + 3):
            _, cache = agg(tokens, past_cache=cache)
        assert cache["num_frames"] == max_cache

    def test_next_time_unbounded(self):
        max_cache = 3
        n = 7
        agg = make_agg(max_cache_seq=max_cache)
        tokens = torch.randn(1, _TPF, _DIM)
        cache = agg.init_cache(1)
        for _ in range(n):
            _, cache = agg(tokens, past_cache=cache)
        assert cache["next_time"] == n
        assert cache["num_frames"] == max_cache

    def test_output_finite_after_overflow(self):
        agg = make_agg(max_cache_seq=2, num_layers=1)
        tokens = torch.randn(1, _TPF, _DIM)
        cache = agg.init_cache(1)
        out = None
        for _ in range(6):
            out, cache = agg(tokens, past_cache=cache)
        assert out.shape == (1, _TPF, _DIM)
        assert torch.isfinite(out).all()

    def test_position_tensors_bounded(self):
        max_cache = 3
        agg = make_agg(max_cache_seq=max_cache, num_layers=1)
        tokens = torch.randn(1, _TPF, _DIM)
        cache = agg.init_cache(1)
        for _ in range(max_cache + 4):
            _, cache = agg(tokens, past_cache=cache)
        max_len = max_cache * _TPF
        assert cache["positions"]["t"].shape[0] <= max_len
        assert cache["positions"]["d"].shape[0] <= max_len
        assert cache["positions"]["b"].shape[0] <= max_len
