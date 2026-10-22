"""Rate representability, overhead units and honest sensitivity accounting."""
import importlib

import pytest
import torch


def test_proxy_can_represent_low_qp30_rates_without_logistic_floor():
    cls = importlib.import_module("ar_od_proxy.models").VideoProxy
    proxy = cls(width=4)
    with torch.no_grad():
        for parameter in proxy.parameters():
            parameter.zero_()
        # A representable 100-bit header and negligible content rate.
        assert hasattr(proxy, "rate_head"), "rate needs a head without the v1 entropy floor"
        proxy.rate_head.log_header.fill_(torch.log(torch.tensor(100.)).item())
        proxy.rate_head.raw_weights.fill_(-20)
        proxy.rate_head.raw_bias.fill_(-20)
    actual = proxy(torch.full((1, 16, 3, 64, 64), .5), "h264", 30).bpp.item()
    assert actual == pytest.approx(100 / (16 * 64 * 64), abs=1e-6)


def test_rate_header_is_amortized_over_actual_frames_and_pixels():
    module = importlib.import_module("ar_od_proxy.models")
    assert hasattr(module, "CalibratedRateHead"), "missing calibrated rate model"
    head = module.CalibratedRateHead()
    with torch.no_grad():
        head.raw_weights.fill_(-30)
        head.raw_bias.fill_(-30)
    image = torch.zeros(1, 1, 3, 32, 32)
    video = image.expand(-1, 16, -1, -1, -1)
    assert head(image, "h265", 35).item() / head(video, "h265", 35).item() == pytest.approx(16)


def test_rate_content_gradient_prefers_less_spatial_detail():
    module = importlib.import_module("ar_od_proxy.models")
    assert hasattr(module, "CalibratedRateHead"), "missing content-sensitive rate model"
    head = module.CalibratedRateHead().requires_grad_(False)
    torch.manual_seed(31)
    x = (.2 + .6 * torch.rand(1, 2, 3, 32, 32)).requires_grad_()
    smooth = torch.nn.functional.avg_pool2d(x.flatten(0, 1), 3, 1, 1).reshape_as(x)
    gradient = torch.autograd.grad(head(x, "h264", 40).sum(), x)[0]
    assert (gradient * (smooth - x)).sum().item() < 0


def test_ties_are_not_counted_as_gradient_agreement():
    module = importlib.import_module("ar_od_proxy.calibration")
    result = module.direction_agreement([0., 1., -1.], [1., 2., 2.], tolerance=.01)
    assert result == {"total": 3, "informative": 2, "ties": 1, "agree": 1, "agreement": .5}


def test_old_checkpoint_migration_is_explicit_and_reproduces_legacy_rate(tmp_path):
    models = importlib.import_module("ar_od_proxy.models")
    engine = importlib.import_module("ar_od_proxy.engine")
    old = models.VideoProxy(width=4)
    legacy = {k: v for k, v in old.state_dict().items() if not k.startswith("rate_head.")}
    npp = models.VideoNPP(width=4)
    path = tmp_path / "v1.pt"
    torch.save({"format": 1, "proxy": legacy, "npp": npp.state_dict(), "config": {"width": 4}}, path)
    with pytest.raises(ValueError, match="legacy"):
        engine.load_checkpoint(path)
    _, restored, _ = engine.load_checkpoint(path, migrate_legacy=True)
    x = torch.rand(1, 2, 3, 32, 32)
    torch.testing.assert_close(old.forward_legacy(x, "h265", 45).bpp,
                               restored.forward_legacy(x, "h265", 45).bpp)


def test_calibration_fits_rates_using_train_pairs_only():
    models = importlib.import_module("ar_od_proxy.models")
    module = importlib.import_module("ar_od_proxy.calibration")
    assert hasattr(module, "fit_rate"), "missing train-only calibration"
    torch.manual_seed(18)
    teacher = models.CalibratedRateHead()
    with torch.no_grad():
        teacher.raw_weights.fill_(0.)
    samples = torch.rand(24, 2, 3, 32, 32)
    features = teacher.features(samples).detach()
    points = torch.arange(24) % 8
    pixels = torch.full((24,), 2048.)
    target = teacher.from_features(features, pixels, points).detach()
    head = models.CalibratedRateHead()
    before = (head.from_features(features, pixels, points).log() - target.log()).square().mean()
    module.fit_rate(head, features, pixels, points, target, steps=400, paired=False)
    after = (head.from_features(features, pixels, points).log() - target.log()).square().mean()
    assert after < before / 20


def test_finite_step_summary_keeps_rate_and_reconstruction_tolerances_separate():
    module = importlib.import_module("ar_od_proxy.calibration")
    assert hasattr(module, "summarize"), "missing held-out grouped report"
    rows = [{"task": "ar", "codec": "h264", "qp": 30, "model": "new", "pixels": 1000,
             "actual_bpp": .2, "proxy_bpp": .22, "l1": .01, "psnr": 35., "codec_psnr": 30.,
             "directions": [{"rate_actual": .001, "rate_proxy": -.1,
                             "recon_actual": 1e-6, "recon_proxy": 2e-6}],
             "descent": [{"epsilon": 1/255, "actual_change": 0., "proxy_change": -.1}]}]
    group = module.summarize(rows)[0]
    assert group["rate_direction"]["ties"] == 1
    assert group["reconstruction_direction"]["agreement"] == 1
    assert group["descent"][0]["ties"] == 1
    assert group["descent"][0]["improved"] == 0
