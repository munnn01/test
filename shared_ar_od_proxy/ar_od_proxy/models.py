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


class RegionalVideoNPP(VideoNPP):
    """Causal learned spatial filter; RGB/codec/QP are its only inference inputs."""
    def __init__(self, width=12, strength=0.12, filter_limit=0.9):
        super().__init__(width, strength)
        if not 0 < filter_limit <= .9:
            raise ValueError('filter limit must be positive and at most .9')
        self.filter_limit = float(filter_limit)
        self.filter_gate = nn.Sequential(nn.Conv2d(5, width, 3, padding=1), nn.SiLU(),
                                         nn.Conv2d(width, 1, 3, padding=1))
        self.filter_strength = nn.Parameter(torch.zeros(8))

    @classmethod
    def from_base(cls, base):
        model = cls(base.spatial.out_channels, base.strength).to(next(base.parameters()).device)
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        expected = {'filter_strength'} | {'filter_gate.' + k for k in model.filter_gate.state_dict()}
        if unexpected or set(missing) != expected:
            raise ValueError('regional initialization requires a plain VideoNPP')
        return model

    def forward(self, x, codec, qp):
        base = super().forward(x, codec, qp)
        b, t, c, h, w = x.shape
        flat = x.reshape(b*t, c, h, w)
        low = F.avg_pool2d(F.pad(flat, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
        contrast = (low-flat).abs().mean(1, keepdim=True)
        motion = torch.cat((torch.zeros_like(x[:, :1, :1]),
                            (x[:, 1:]-x[:, :-1]).abs().mean(2, keepdim=True)), 1)
        gate = self.filter_gate(torch.cat((flat, contrast, motion.reshape(b*t, 1, h, w)), 1)).sigmoid()
        strength = self.filter_limit * self.filter_strength[CalibratedRateHead.point(codec, qp)].tanh()
        residual = (strength * gate * (low-flat)).reshape_as(x)
        return (base+residual).clamp(0, 1)


@dataclass
class ProxyOutput:
    reconstruction: torch.Tensor
    bpp: torch.Tensor


class CalibratedRateHead(nn.Module):
    """Positive content regression plus stream overhead, for the fixed QP grid.

    Unlike a latent entropy model, this predicts the complete real stream rate.
    Nonnegative content coefficients make its smooth sensitivity interpretable;
    fidelity still requires held-out real-codec finite-step measurements.
    """
    qps = (30, 35, 40, 45)

    def __init__(self):
        super().__init__()
        self.log_header = nn.Parameter(torch.tensor([1024.] * 4 + [18000.] * 4).log())
        self.raw_bias = nn.Parameter(torch.full((8,), -12.))
        self.raw_weights = nn.Parameter(torch.full((8, 27), -3.))

    @classmethod
    def point(cls, codec, qp):
        if codec not in ("h264", "h265") or qp not in cls.qps:
            raise ValueError("rate calibration supports h264/h265 at QP 30/35/40/45 only")
        return 4 * int(codec == "h265") + cls.qps.index(qp)

    @staticmethod
    def features(x):
        # Fixed linear Y/Cb/Cr transform. Derivatives stay connected to RGB.
        matrix = x.new_tensor([[.299, .587, .114], [-.168736, -.331264, .5],
                               [.5, -.418688, -.081312]])
        yuv = torch.einsum("oc,btchw->btohw", matrix, x)
        spatial = []
        for stride in (1, 2, 4, 8):
            dx = (yuv[..., stride:] - yuv[..., :-stride]).abs().mean((-1, -2))
            dy = (yuv[..., stride:, :] - yuv[..., :-stride, :]).abs().mean((-1, -2))
            spatial.append((dx + dy) / 2)
        spatial = torch.cat(spatial, 2)
        count = x.shape[1]
        intra = spatial[:, 0] / count
        if count > 1:
            inter = spatial[:, 1:].sum(1) / count
            motion = (yuv[:, 1:] - yuv[:, :-1]).abs().mean((1, 3, 4)) * (count - 1) / count
        else:
            inter, motion = torch.zeros_like(intra), x.new_zeros(x.shape[0], 3)
        return torch.cat((intra, inter, motion), 1)

    def from_features(self, features, pixels, point):
        # pixels is the actual T*H*W denominator, including T=1 for COCO.
        return (self.log_header[point].exp() / pixels + F.softplus(self.raw_bias[point])
                + (F.softplus(self.raw_weights[point]) * features).sum(-1))

    def forward(self, x, codec, qp):
        return self.from_features(self.features(x), x.shape[1] * x.shape[3] * x.shape[4],
                                  self.point(codec, qp))


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
        self.rate_head = CalibratedRateHead()

    def forward(self, x, codec, qp):
        return self._forward(x, codec, qp, legacy=False)

    def forward_legacy(self, x, codec, qp):
        """Reproduce v1 only for paired diagnostics of a migrated checkpoint."""
        return self._forward(x, codec, qp, legacy=True)

    def _forward(self, x, codec, qp, legacy):
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
            if legacy:
                scale = F.softplus(self.entropy_scale(hidden)) + 0.05
                upper = ((quantized + step / 2 - self.entropy_mean) / scale).sigmoid()
                lower = ((quantized - step / 2 - self.entropy_mean) / scale).sigmoid()
                bits.append(-(upper - lower).clamp_min(1e-9).log2().flatten(1).sum(1))
        bpp = (torch.stack(bits, 1).sum(1) / (x.shape[1] * x.shape[3] * x.shape[4])
               if legacy else self.rate_head(x, codec, qp))
        return ProxyOutput(torch.stack(frames, 1), bpp)
