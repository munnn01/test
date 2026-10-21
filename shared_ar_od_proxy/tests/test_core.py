"""Catch detached proxy gradients, temporal lookahead and real-codec accounting errors."""
import importlib

import pytest
import torch

torch.set_num_threads(2)


def test_bridge_uses_real_values_and_proxy_jacobian():
    bridge = importlib.import_module("ar_od_proxy.models").real_value_proxy_grad
    y = torch.tensor([2.0], requires_grad=True)
    x = bridge(3 * y, torch.tensor([100.0]))
    rate = bridge(2 * y, torch.tensor([7.0]))
    assert x.item() == 100 and rate.item() == 7
    (x.sum() + 4 * rate.sum()).backward()
    assert y.grad.item() == 11.0  # An identity STE would give 5 instead.


def test_npp_starts_identity_and_handles_non_square_video():
    cls = importlib.import_module("ar_od_proxy.models").VideoNPP
    x = torch.rand(1, 3, 3, 22, 26)
    y = cls(width=4)(x, "h264", 35)
    assert y.shape == x.shape
    torch.testing.assert_close(y, x)


def test_npp_is_causal_after_nonzero_residual_initialization():
    cls = importlib.import_module("ar_od_proxy.models").VideoNPP
    model = cls(width=4)
    with torch.no_grad():
        model.rgb.weight.normal_(0, 0.05)
    x = torch.rand(1, 3, 3, 24, 24)
    other = x.clone()
    other[:, 2] = 0
    a, b = model(x, "h265", 40), model(other, "h265", 40)
    torch.testing.assert_close(a[:, :2], b[:, :2])


def test_frozen_proxy_retains_reconstruction_and_rate_input_gradients():
    cls = importlib.import_module("ar_od_proxy.models").VideoProxy
    proxy = cls(width=4).requires_grad_(False).eval()
    x = torch.rand(1, 3, 3, 24, 24, requires_grad=True)
    out = proxy(x, "h264", 35)
    assert out.reconstruction.shape == x.shape
    assert out.bpp.shape == (1,) and torch.isfinite(out.bpp).all()
    assert out.bpp.item() > 0
    grad_rate = torch.autograd.grad(out.bpp.sum(), x, retain_graph=True)[0]
    assert torch.isfinite(grad_rate).all() and grad_rate.abs().sum() > 0
    out.reconstruction.mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is None for p in proxy.parameters())


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_actual_codec_preserves_frames_and_counts_stream_headers(codec):
    cls = importlib.import_module("ar_od_proxy.codec").RealCodec
    torch.manual_seed(4)
    x = torch.rand(1, 4, 3, 32, 48)
    out = cls(preset="ultrafast", gop=4).roundtrip(x, codec, 35)
    assert out.reconstruction.shape == x.shape
    assert len(out.streams) == 1 and len(out.streams[0]) > 0
    assert out.bpp.item() == pytest.approx(8 * len(out.streams[0]) / (4 * 32 * 48))
    assert not out.reconstruction.requires_grad
    assert out.reconstruction.min() >= 0 and out.reconstruction.max() <= 1


def test_codec_rejects_odd_yuv420_dimensions():
    cls = importlib.import_module("ar_od_proxy.codec").RealCodec
    with pytest.raises(ValueError, match="even"):
        cls().roundtrip(torch.rand(1, 1, 3, 31, 32), "h264", 35)
