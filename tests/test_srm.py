import torch

from qalf.models.srm import FixedSRMPreprocess


def test_srm_shape_and_fixed_kernels():
    module = FixedSRMPreprocess()
    x = torch.randn(2, 3, 32, 32)
    y = module(x)
    assert y.shape == x.shape
    assert module.kernel_bank.shape == (3, 1, 5, 5)
    assert module.kernel_bank.requires_grad is False
    assert module.residual_scale.requires_grad is True
    assert torch.allclose(module.kernel_bank.sum(dim=(-1, -2)), torch.zeros(3, 1))


def test_srm_zero_scale_is_identity():
    module = FixedSRMPreprocess()
    module.residual_scale.data.zero_()
    x = torch.randn(1, 3, 16, 16)
    assert torch.equal(module(x), x)
