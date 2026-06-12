import torch

from anvil_audio.inference.sampling import sample


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


def test_legacy_sample_verbose_is_cpu_safe():
    x = torch.randn(1, 2, 16)

    out = sample(_ZeroVelocity(), x, steps=10, eta=0.0, verbose=True)

    assert out.shape == x.shape
