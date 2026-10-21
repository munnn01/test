"""Causal NPP, conditional video proxy and explicit value/gradient replacement."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def real_value_proxy_grad(proxy: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    if proxy.shape != real.shape:
        raise ValueError("real and proxy tensor shapes must agree")
    return proxy + (real.to(proxy) - proxy).detach()


def validate_video(x):
    if x.ndim != 5 or x.shape[2] != 3 or min(x.shape) < 1:
        raise ValueError("expected nonempty B,T,3,H,W video")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("video must contain finite floating-point RGB values")


class Condition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, x, codec, qp):
        if codec not in ("h264", "h265") or type(qp) is not int or not 0 <= qp <= 51:
            raise ValueError("unsupported codec or integer QP")
        q = x.new_tensor([[qp / 51.0, float(codec == "h265")]])
        return 1 + 0.1 * self.net(q).tanh().view(1, -1, 1, 1)


class ConvGRU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, x, hidden):
        if hidden is None:
            hidden = torch.zeros_like(x)
        reset, update = self.gates(torch.cat((x, hidden), 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat((x, reset * hidden), 1)).tanh()
        return (1 - update) * hidden + update * candidate


class VideoNPP(nn.Module):
    """One bounded RGB residual; reset causal context at every clip/GOP boundary."""
    def __init__(self, width=12, strength=0.12):
        super().__init__()
        self.strength = strength
        self.spatial = nn.Conv2d(3, width, 3, padding=1)
        self.down = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.temporal = ConvGRU(2 * width)
        self.condition = Condition(2 * width)
        self.pixel = nn.Sequential(nn.Conv2d(3, width, 1), nn.SiLU(), nn.Conv2d(width, 3, 1))
        self.rgb = nn.Conv2d(3 * width, 3, 3, padding=1)
        nn.init.zeros_(self.rgb.weight)
        nn.init.zeros_(self.rgb.bias)
        nn.init.zeros_(self.pixel[-1].weight)
        nn.init.zeros_(self.pixel[-1].bias)

    def forward(self, x, codec, qp):
        validate_video(x)
        scale = self.condition(x, codec, qp)
        hidden, output = None, []
        for frame in x.unbind(1):
            shallow = F.silu(self.spatial(frame))
            encoded = F.silu(self.down(shallow)) * scale
            hidden = self.temporal(encoded, hidden)
            up = F.interpolate(hidden, frame.shape[-2:], mode="bilinear", align_corners=False)
            residual = self.pixel(frame) + self.rgb(torch.cat((shallow, up), 1))
            output.append((frame + self.strength * residual.tanh()).clamp(0, 1))
        return torch.stack(output, 1)


@dataclass
class ProxyOutput:
    reconstruction: torch.Tensor
    bpp: torch.Tensor


class VideoProxy(nn.Module):
    """Learned surrogate; its entropy values are calibrated against actual bits.

    This does not claim to replicate codec decisions or their true derivative.
    The previous proxy reconstruction is used as temporal prediction context.
    """
    def __init__(self, width=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(6, width, 5, 2, 2), nn.SiLU(),
                                     nn.Conv2d(width, 2 * width, 3, 2, 1), nn.SiLU())
        self.temporal = ConvGRU(2 * width)
        self.condition = Condition(2 * width)
        self.decoder = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(width, 3, 3, padding=1))
        self.entropy_scale = nn.Conv2d(2 * width, 2 * width, 1)
        self.entropy_mean = nn.Parameter(torch.zeros(1, 2 * width, 1, 1))

    def forward(self, x, codec, qp):
        validate_video(x)
        condition = self.condition(x, codec, qp)
        step = 0.05 * 2 ** ((qp - 30) / 6)
        hidden, previous, frames, bits = None, torch.zeros_like(x[:, 0]), [], []
        for frame in x.unbind(1):
            encoded = self.encoder(torch.cat((frame, previous), 1)) * condition
            hidden = self.temporal(encoded, hidden)
            rounded = (hidden / step).round() * step
            quantized = hidden + (rounded - hidden).detach()
            delta = self.decoder(quantized)
            delta = F.interpolate(delta, frame.shape[-2:], mode="bilinear", align_corners=False)
            previous = (frame + 0.2 * delta.tanh()).clamp(0, 1)
            frames.append(previous)
            scale = F.softplus(self.entropy_scale(hidden)) + 0.05
            upper = ((quantized + step / 2 - self.entropy_mean) / scale).sigmoid()
            lower = ((quantized - step / 2 - self.entropy_mean) / scale).sigmoid()
            bits.append(-(upper - lower).clamp_min(1e-9).log2().flatten(1).sum(1))
        bpp = torch.stack(bits, 1).sum(1) / (x.shape[1] * x.shape[3] * x.shape[4])
        return ProxyOutput(torch.stack(frames, 1), bpp)
