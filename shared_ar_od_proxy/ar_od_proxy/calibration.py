"""Train-only rate calibration and held-out finite-step codec sensitivity audit."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from .codec import RealCodec
from .engine import emit, load_checkpoint, save_checkpoint, source_hashes
from .models import real_value_proxy_grad


def blur(x):
    flat = x.flatten(0, 1)
    return F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode="replicate"), 3, 1).reshape_as(x)


def fit_rate(head, features, pixels, points, target, steps=4000, paired=True):
    """Only precomputed TRAIN features/rates enter this optimizer."""
    head.requires_grad_(True)
    optimizer = torch.optim.Adam(head.parameters(), lr=.04)
    target_log = target.log()
    for step in range(steps):
        prediction = head.from_features(features, pixels, points).clamp_min(1e-9).log()
        loss = F.smooth_l1_loss(prediction, target_log)
        if paired:
            p, t = prediction.reshape(-1, 3), target_log.reshape(-1, 3)
            loss = loss + 2 * F.smooth_l1_loss(p[:, 1:] - p[:, :1], t[:, 1:] - t[:, :1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
    head.requires_grad_(False)
    return float(loss.detach())


def calibrate(npp, proxy, codec, samples, config, out, reconstruction_steps=1024, rate_steps=4000):
    device = next(proxy.parameters()).device
    cases, features, pixels, points, rates = [], [], [], [], []
    grid = list(itertools.product(config["codecs"], config["qps"]))
    npp.requires_grad_(False).eval()
    # Complete grid for each source, paired perturbations of exactly the same input.
    for record, x, _ in samples:
        for name, qp in grid:
            with torch.no_grad():
                y = npp(x.to(device), name, qp)
                smooth = blur(y)
                for amount in (0., .75, -.5):
                    candidate = (y + amount * (smooth - y)).clamp(0, 1)
                    actual = codec.roundtrip(candidate, name, qp)
                    features.append(proxy.rate_head.features(candidate).cpu())
                    pixels.append(candidate.shape[1] * candidate.shape[3] * candidate.shape[4])
                    points.append(proxy.rate_head.point(name, qp))
                    rates.append(actual.bpp.cpu())
                    if amount == 0:
                        cases.append((y.cpu(), (actual.reconstruction.cpu() * 255).round().byte(), name, qp))
                    emit(out / "calibration_train.jsonl", {"source": record["id"], "codec": name,
                         "qp": qp, "blur_amount": amount, "actual_bpp": float(actual.bpp.mean()),
                         "stream_sha256": hashlib.sha256(actual.streams[0]).hexdigest()})
    feature = torch.cat(features).to(device)
    pixel = torch.tensor(pixels, device=device, dtype=feature.dtype)
    point = torch.tensor(points, device=device)
    target = torch.cat(rates).to(device)
    loss = fit_rate(proxy.rate_head, feature, pixel, point, target, steps=rate_steps)
    torch.save({"features": feature.cpu(), "pixels": pixel.cpu(), "points": point.cpu(),
                "target_bpp": target.cpu()}, out / "train_rate_targets.pt")
    emit(out / "training.jsonl", {"stage": "rate_calibration", "steps": rate_steps,
         "train_cases": len(rates), "train_log_loss": loss})
    # Isolate reconstruction fitting: the calibrated rate head is frozen.
    proxy.requires_grad_(True).train()
    proxy.rate_head.requires_grad_(False)
    proxy.entropy_scale.requires_grad_(False)
    proxy.entropy_mean.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in proxy.parameters() if p.requires_grad), lr=.0003)
    for step in range(reconstruction_steps):
        y, decoded, name, qp = cases[step % len(cases)]
        predicted = proxy(y.to(device), name, qp).reconstruction
        actual = decoded.to(device).float() / 255
        l1 = F.l1_loss(predicted, actual)
        loss = l1 + 5 * F.mse_loss(predicted, actual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 64 == 0 or step + 1 == reconstruction_steps:
            emit(out / "training.jsonl", {"stage": "reconstruction_calibration", "step": step + 1,
                 "loss": float(loss.detach()), "l1": float(l1.detach())})
    proxy.requires_grad_(False).eval()
    for parameter in proxy.parameters():
        parameter.grad = None
    save_checkpoint(out / "calibrated.pt", npp, proxy, config, "proxy_calibration", reconstruction_steps)


def psnr(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def normalized(direction):
    rms = direction.square().mean().sqrt()
    if not torch.isfinite(rms):
        raise ValueError("nonfinite input gradient/direction")
    return direction / rms.clamp_min(1e-12)


def stream_record(packet):
    return {"bytes": len(packet.streams[0]), "sha256": hashlib.sha256(packet.streams[0]).hexdigest()}


def audit(npp, baseline, calibrated, codec, samples, config, out):
    """Same DEV inputs and actual streams for before/after; never optimize here."""
    device = next(calibrated.parameters()).device
    rows = []
    for sample_index, (record, x, _) in enumerate(samples):
        for name, qp in itertools.product(config["codecs"], config["qps"]):
            with torch.no_grad():
                source = x.to(device)
                y = npp(source, name, qp).detach()
                real = codec.roundtrip(y, name, qp)
            pixels = y.shape[1] * y.shape[3] * y.shape[4]
            generator = torch.Generator(device=device).manual_seed(config["seed"] + sample_index)
            noise = torch.randn(y.shape[0] * y.shape[1], 3, 8, 8, device=device, generator=generator)
            noise = F.interpolate(noise, y.shape[-2:], mode="bilinear", align_corners=False).reshape_as(y)
            directions = {"blur": normalized(blur(y) - y), "random_smooth": normalized(noise)}
            # Encode paired probes once and share the exact results across both models.
            probes = []
            with torch.no_grad():
                for direction_name, direction in directions.items():
                    eps = 4 / 255
                    plus, minus = (y + eps * direction).clamp(0, 1), (y - eps * direction).clamp(0, 1)
                    rp, rm = codec.roundtrip(plus, name, qp), codec.roundtrip(minus, name, qp)
                    probes.append((direction_name, plus, minus, rp, rm))
            for label, proxy in (("legacy", baseline), ("new", calibrated)):
                forward = proxy.forward_legacy if label == "legacy" else proxy.forward
                input_tensor = y.clone().requires_grad_()
                predicted = forward(input_tensor, name, qp)
                grad_rate = torch.autograd.grad(predicted.bpp.sum(), input_tensor, retain_graph=True)[0]
                # Evaluate reconstruction Jacobian at the real forward value, as in NPP training.
                bridge = real_value_proxy_grad(predicted.reconstruction, real.reconstruction)
                reconstruction_loss = F.mse_loss(bridge, y)
                grad_reconstruction = torch.autograd.grad(reconstruction_loss, input_tensor)[0]
                with torch.no_grad():
                    row = {"source": record["id"], "task": record["task"], "codec": name, "qp": qp,
                           "model": label, "pixels": pixels, "base_stream": stream_record(real),
                           "actual_bpp": float(real.bpp[0]), "proxy_bpp": float(predicted.bpp.detach()[0]),
                           "l1": float((predicted.reconstruction.detach() - real.reconstruction).abs().mean()),
                           "psnr": psnr(F.mse_loss(predicted.reconstruction.detach(), real.reconstruction)),
                           "codec_psnr": psnr(F.mse_loss(real.reconstruction, y)),
                           "rate_gradient_rms": float(grad_rate.square().mean().sqrt()),
                           "reconstruction_gradient_rms": float(grad_reconstruction.square().mean().sqrt()),
                           "directions": [], "descent": []}
                    for direction_name, plus, minus, rp, rm in probes:
                        pp, pm = forward(plus, name, qp), forward(minus, name, qp)
                        row["directions"].append({"name": direction_name, "epsilon": 4 / 255,
                            "plus_stream": stream_record(rp), "minus_stream": stream_record(rm),
                            "rate_actual": float((rp.bpp - rm.bpp)[0]),
                            "rate_proxy": float((pp.bpp - pm.bpp)[0]),
                            "recon_actual": float(F.mse_loss(rp.reconstruction, y) - F.mse_loss(rm.reconstruction, y)),
                            "recon_proxy": float((grad_reconstruction * (plus - minus)).sum())})
                    for epsilon in (1 / 255, 4 / 255):
                        candidate = (y - epsilon * normalized(grad_rate)).clamp(0, 1)
                        candidate_real = codec.roundtrip(candidate, name, qp)
                        candidate_proxy = forward(candidate, name, qp)
                        row["descent"].append({"epsilon": epsilon,
                             "actual_change": float((candidate_real.bpp - real.bpp)[0]),
                             "actual_change_percent": float(100 * (candidate_real.bpp / real.bpp - 1)[0]),
                             "proxy_change": float((candidate_proxy.bpp - predicted.bpp.detach())[0]),
                             "stream": stream_record(candidate_real)})
                emit(out / "diagnostics.jsonl", row)
                rows.append(row)
    return rows


def direction_agreement(actual, predicted, tolerance):
    if len(actual) != len(predicted):
        raise ValueError("paired sensitivity lengths differ")
    informative = [(a, p) for a, p in zip(actual, predicted) if abs(a) > tolerance]
    agree = sum(a * p > 0 for a, p in informative)
    count = len(informative)
    return {"total": len(actual), "informative": count, "ties": len(actual) - count,
            "agree": agree, "agreement": agree / count if count else None}


def summarize(rows):
    groups = {}
    for row in rows:
        key = row["task"], row["codec"], row["qp"], row["model"]
        groups.setdefault(key, []).append(row)
    results = []
    for (task, codec, qp, model), group in sorted(groups.items()):
        rate_actual, rate_proxy, recon_actual, recon_proxy = [], [], [], []
        for row in group:
            for probe in row["directions"]:
                rate_actual.append(probe["rate_actual"] * row["pixels"])
                rate_proxy.append(probe["rate_proxy"] * row["pixels"])
                recon_actual.append(probe["recon_actual"])
                recon_proxy.append(probe["recon_proxy"])
        descent = []
        epsilons = sorted({d["epsilon"] for r in group for d in r["descent"]})
        for epsilon in epsilons:
            candidates = [(d, r["pixels"], r["actual_bpp"]) for r in group for d in r["descent"]
                          if d["epsilon"] == epsilon]
            improvements = [d["actual_change"] * pixels < -1 for d, pixels, _ in candidates]
            ties = [abs(d["actual_change"] * pixels) <= 1 for d, pixels, _ in candidates]
            descent.append({"epsilon": epsilon, "count": len(candidates), "improved": sum(improvements),
                "ties": sum(ties), "worsened": len(candidates) - sum(ties) - sum(improvements),
                "mean_rate_change_percent": statistics.mean(100 * d["actual_change"] / bpp for d, _, bpp in candidates),
                "proxy_decreased": sum(d["proxy_change"] < 0 for d, _, _ in candidates)})
        results.append({"task": task, "codec": codec, "qp": qp, "model": model, "count": len(group),
            "mape_percent": statistics.mean(100 * abs(r["proxy_bpp"] / r["actual_bpp"] - 1) for r in group),
            "median_ratio": statistics.median(r["proxy_bpp"] / r["actual_bpp"] for r in group),
            "mean_l1": statistics.mean(r["l1"] for r in group),
            "mean_psnr": statistics.mean(r["psnr"] for r in group),
            "mean_codec_psnr": statistics.mean(r["codec_psnr"] for r in group),
            "rate_direction": direction_agreement(rate_actual, rate_proxy, 1.),
            "reconstruction_direction": direction_agreement(recon_actual, recon_proxy, 1e-10),
            "descent": descent})
        by_direction = {}
        for name in sorted({d.get("name", "unspecified") for r in group for d in r["directions"]}):
            probes = [(d, r["pixels"]) for r in group for d in r["directions"]
                      if d.get("name", "unspecified") == name]
            by_direction[name] = {
                "rate": direction_agreement([d["rate_actual"] * p for d, p in probes],
                                            [d["rate_proxy"] * p for d, p in probes], 1.),
                "reconstruction": direction_agreement([d["recon_actual"] for d, _ in probes],
                                                       [d["recon_proxy"] for d, _ in probes], 1e-10)}
        results[-1]["by_direction"] = by_direction
    return results


def main():
    from .run import configuration, manifest_for, samples_for
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ar-root", "coco-root", "annotations", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("GPU required unless --allow-cpu is explicit")
    cfg = configuration(args.config)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    meta = {"complete": False, "source_commit": os.environ.get("AROD_SOURCE_COMMIT"),
            "source_hashes": source_hashes(), "config": cfg, "device": device,
            "baseline_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "scope": "Frozen existing NPP; train-only proxy calibration, dev-only diagnostics. No new AR/OD score claim.",
            "rate_steps": 4000, "reconstruction_steps": 1024}
    try:
        npp, baseline, checkpoint = load_checkpoint(args.checkpoint, cfg, device, migrate_legacy=True)
        if checkpoint["format"] != 1:
            raise ValueError("paired v1 comparison requires a legacy checkpoint")
        npp.requires_grad_(False).eval()
        baseline.requires_grad_(False).eval()
        proxy = copy.deepcopy(baseline)
        manifest = manifest_for(args, cfg)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        calibrate(npp, proxy, RealCodec(preset=cfg["preset"], gop=cfg["frames"], fps=cfg["fps"]),
                  samples_for(manifest["train"], cfg), cfg, args.out)
        rows = audit(npp, baseline, proxy,
                     RealCodec(preset=cfg["preset"], gop=cfg["frames"], fps=cfg["fps"]),
                     samples_for(manifest["dev"], cfg), cfg, args.out)
        protocol = {k: v for k, v in meta.items() if k != "complete"}
        (args.out / "summary.json").write_text(json.dumps({"groups": summarize(rows), "protocol": protocol}, indent=2))
        meta["complete"] = True
    except BaseException as error:
        meta["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        meta["wall_seconds"] = time.time() - started
        (args.out / "run.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
        print(json.dumps({"complete": meta["complete"], "seconds": meta["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
