import importlib

import pytest
import torch
from torch import nn

from ar_od_proxy.codec import RealCodec
from ar_od_proxy.models import VideoNPP, VideoProxy


class TinyTasks:
    """Small real differentiable task functions, only for a software integration test."""
    def __init__(self):
        self.ar = nn.Conv2d(3, 2, 1).requires_grad_(False)
        self.od = nn.Conv2d(3, 2, 1).requires_grad_(False)

    def losses(self, x, record, refs):
        return {"ar_gt": self.ar(x[:, 0]).square().mean(), "od_gt": self.od(x[:, -1]).square().mean(),
                "ar_source": x.sum() * 0, "od_source": x.sum() * 0}


def test_real_codec_training_step_updates_only_npp():
    engine = importlib.import_module("ar_od_proxy.engine")
    npp, proxy, tasks = VideoNPP(width=4), VideoProxy(width=4).requires_grad_(False), TinyTasks()
    before_npp = {k: v.clone() for k, v in npp.state_dict().items()}
    frozen = {k: v.clone() for k, v in proxy.state_dict().items()}
    opt = torch.optim.Adam(npp.parameters(), lr=1e-3)
    result = engine.npp_step(npp, proxy, RealCodec(preset="ultrafast", gop=4), tasks,
                             torch.rand(1, 4, 3, 32, 32), {"task": "ar"}, {}, opt,
                             {"rate": 1., "ar": 1., "od": 1., "pre": .5, "temp": .1})
    assert result["npp_grad_norm"] > 0 and result["actual_bpp"] > 0
    assert any(not torch.equal(v, before_npp[k]) for k, v in npp.state_dict().items())
    for k, v in proxy.state_dict().items():
        torch.testing.assert_close(v, frozen[k])
    assert all(p.grad is None for m in (proxy, tasks.ar, tasks.od) for p in m.parameters())


def test_checkpoint_rejects_incompatible_model_config(tmp_path):
    engine = importlib.import_module("ar_od_proxy.engine")
    cfg = {"width": 4, "qps": [30, 35, 40, 45], "codecs": ["h264", "h265"], "frames": 16, "size": 64}
    p = tmp_path / "checkpoint.pt"
    engine.save_checkpoint(p, VideoNPP(width=4), VideoProxy(width=4), cfg, "test", 1)
    a, b, meta = engine.load_checkpoint(p, cfg)
    assert meta["stage"] == "test" and isinstance(a, VideoNPP)
    with pytest.raises(ValueError, match="config"):
        engine.load_checkpoint(p, dict(cfg, width=8))


def test_schedule_balances_both_codecs_and_exact_four_qps():
    engine = importlib.import_module("ar_od_proxy.engine")
    cfg = {"codecs": ["h264", "h265"], "qps": [30, 35, 40, 45]}
    groups = [engine.operating_point(i, cfg) for i in range(16)]
    for codec in ("h264", "h265"):
        for qp in (30, 35, 40, 45):
            assert groups.count((codec, qp)) == 2
