"""Tests for v2 model modules: frequency, multiscale, temporal attention."""

from __future__ import annotations

import torch
import pytest


def test_frequency_preprocess_shape():
    from qalf.models.frequency import FrequencyPreprocess

    module = FrequencyPreprocess()
    x = torch.randn(4, 3, 224, 224)
    y = module(x)
    assert y.shape == (4, 3, 224, 224), f"Expected (4, 3, 224, 224), got {y.shape}"


def test_frequency_preprocess_srm_fixed():
    """SRM kernels must not have requires_grad (non-learnable)."""
    from qalf.models.frequency import FrequencyPreprocess

    module = FrequencyPreprocess()
    assert not module.srm_weight.requires_grad


def test_frequency_preprocess_adapter_learnable():
    from qalf.models.frequency import FrequencyPreprocess

    module = FrequencyPreprocess()
    learnable = [p for p in module.adapter.parameters() if p.requires_grad]
    assert len(learnable) > 0, "Adapter should have learnable parameters"


def test_multiscale_aggregation_shape():
    from qalf.models.multiscale import MultiScaleAggregation

    module = MultiScaleAggregation(
        low_channels=24, mid_channels=112, high_channels=1280, embedding_dim=192
    )
    low = torch.randn(4, 24, 14, 14)
    mid = torch.randn(4, 112, 7, 7)
    high = torch.randn(4, 1280, 4, 4)
    y = module(low, mid, high)
    assert y.shape == (4, 192), f"Expected (4, 192), got {y.shape}"


def test_temporal_attention_pooling_shape():
    from qalf.models.temporal import TemporalAttentionPooling

    module = TemporalAttentionPooling(embedding_dim=192, hidden_dim=64, dropout=0.1)
    x = torch.randn(2, 10, 192)
    y = module(x)
    assert y.shape == (2, 192), f"Expected (2, 192), got {y.shape}"


def test_temporal_attention_weights_sum_to_one():
    from qalf.models.temporal import TemporalAttentionPooling

    module = TemporalAttentionPooling(embedding_dim=192)
    x = torch.randn(3, 8, 192)
    scores = module.attention_mlp(x)
    weights = torch.softmax(scores, dim=1)
    sums = weights.sum(dim=1).squeeze(-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_qalf_v1_backward_compat():
    """V1 defaults: no frequency, no multiscale, no temporal attention."""
    from qalf.models import QALFModel

    model = QALFModel(embedding_dim=192, dropout=0.3)
    batch = {"texture": torch.randn(2, 8, 3, 160, 160)}
    output = model(batch)
    assert output["logit"].shape == (2,)
    assert output["embedding"].shape == (2, 192)


def test_qalf_v2_full():
    """V2 with all modules enabled."""
    from qalf.models import QALFModel

    model = QALFModel(
        embedding_dim=192,
        dropout=0.3,
        frequency_preprocess=True,
        multiscale=True,
        temporal_attention=True,
    )
    batch = {"texture": torch.randn(1, 10, 3, 224, 224)}
    output = model(batch)
    assert output["logit"].shape == (1,)
    assert output["embedding"].shape == (1, 192)


def test_qalf_v2_partial_freq_only():
    from qalf.models import QALFModel

    model = QALFModel(embedding_dim=192, frequency_preprocess=True)
    batch = {"texture": torch.randn(1, 8, 3, 160, 160)}
    output = model(batch)
    assert output["logit"].shape == (1,)


def test_qalf_v2_partial_temporal_only():
    from qalf.models import QALFModel

    model = QALFModel(embedding_dim=192, temporal_attention=True)
    batch = {"texture": torch.randn(1, 8, 3, 160, 160)}
    output = model(batch)
    assert output["logit"].shape == (1,)


def test_qalf_v2_partial_multiscale_only():
    from qalf.models import QALFModel

    model = QALFModel(embedding_dim=192, multiscale=True)
    batch = {"texture": torch.randn(1, 8, 3, 160, 160)}
    output = model(batch)
    assert output["logit"].shape == (1,)


def test_qalf_v2_param_count():
    """V2 full should be under 6M parameters."""
    from qalf.models import QALFModel

    model = QALFModel(
        embedding_dim=192,
        dropout=0.3,
        frequency_preprocess=True,
        multiscale=True,
        temporal_attention=True,
    )
    total = sum(p.numel() for p in model.parameters())
    assert total < 6_000_000, f"Expected < 6M params, got {total:,}"


def test_qalf_v2_gradient_flow():
    """All new modules should receive gradients."""
    from qalf.models import QALFModel

    model = QALFModel(
        embedding_dim=192,
        dropout=0.0,
        texture_pretrained=False,
        frequency_preprocess=True,
        multiscale=True,
        temporal_attention=True,
    )
    batch = {"texture": torch.randn(1, 4, 3, 64, 64)}
    output = model(batch)
    output["logit"].sum().backward()
    # Check frequency adapter
    for p in model.texture_encoder.frequency.adapter.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Frequency adapter should receive gradient"
    # Check temporal attention
    for p in model.texture_encoder.temporal_pool.attention_mlp.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Temporal attention should receive gradient"
    # Check multiscale
    for p in model.texture_encoder.multiscale_agg.merge.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Multiscale merge should receive gradient"
