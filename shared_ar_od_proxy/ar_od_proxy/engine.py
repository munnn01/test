"""Proxy calibration, frozen-task optimization and checkpoint contracts."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .models import VideoNPP, RegionalVideoNPP, VideoProxy, real_value_proxy_grad


def operating_point(step, config):
    grid = list(itertools.product(config["codecs"], config["qps"]))
    return grid[step % len(grid)]


def training_case(step, count, config):
    grid_size = len(config["codecs"]) * len(config["qps"])
    codec, qp = operating_point(step, config)
    return (step // grid_size) % count, codec, qp


def source_hashes():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def save_checkpoint(path, npp, proxy, config, stage, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"format": 2, "npp": npp.state_dict(), "proxy": proxy.state_dict(),
                "config": config, "stage": stage, "step": step, "source_hashes": source_hashes()}, temporary)
    temporary.replace(path)


def load_checkpoint(path, config=None, device="cpu", migrate_legacy=False):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") not in (1, 2):
        raise ValueError("unsupported checkpoint format")
    legacy = checkpoint["format"] == 1
    if legacy and not migrate_legacy:
        raise ValueError("legacy proxy rate head requires explicit migrate_legacy=True and calibration")
    stored = checkpoint["config"]
    if config is not None and stored != config:
        raise ValueError("checkpoint config differs from requested config")
    kind = stored.get('npp_kind', 'video')
    if kind not in ('video', 'regional'):
        raise ValueError('unknown NPP model kind')
    model = RegionalVideoNPP if kind == 'regional' else VideoNPP
    kwargs = {'filter_limit': stored.get('filter_limit', .9)} if kind == 'regional' else {}
    npp, proxy = model(stored["width"], **kwargs).to(device), VideoProxy(stored["width"]).to(device)
    npp.load_state_dict(checkpoint["npp"], strict=True)
    missing, unexpected = proxy.load_state_dict(checkpoint["proxy"], strict=not legacy)
    if legacy and (unexpected or set(missing) != {"rate_head.log_header", "rate_head.raw_bias", "rate_head.raw_weights"}):
        raise ValueError("unexpected legacy checkpoint keys")
    return npp, proxy, checkpoint


def load_calibrated_start(path, config, device="cpu"):
    npp, proxy, saved = load_checkpoint(path, device=device)
    if saved["format"] != 2 or saved["stage"] != "proxy_calibration":
        raise ValueError("training requires a calibrated format-2 proxy checkpoint")
    if config.get("proxy_refresh_every") != 0:
        raise ValueError("disable proxy refresh to preserve the calibrated proxy")
    maximum = config.get("max_npp_update_rms")
    if (not isinstance(maximum, (int, float)) or not math.isfinite(maximum)
            or not 0 < maximum <= 1 / 255):
        raise ValueError("calibrated training requires finite positive NPP update RMS at most 1/255")
    allowed = {"learning_rate", "proxy_refresh_every", "max_npp_update_rms"}
    different = {k for k in set(config) | set(saved["config"])
                 if config.get(k) != saved["config"].get(k)}
    if different - allowed:
        raise ValueError("calibrated checkpoint config differs: " + str(sorted(different - allowed)))
    proxy.requires_grad_(False).eval()
    return npp, proxy, saved


def bounded_optimizer_step(npp, optimizer, x, previous, name, qp, maximum):
    """Backtrack the Adam displacement; bound the CURRENT case's output change."""
    if not 0 < maximum <= 1 / 255:
        raise ValueError("NPP update RMS must be positive and at most 1/255")
    parameters = list(npp.parameters())
    before = [p.detach().clone() for p in parameters]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    proposed = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        for attempt in range(17):
            scale = .5 ** attempt
            if attempt:
                for p, old, new in zip(parameters, before, proposed):
                    p.copy_(old + scale * (new - old))
            change = (npp(x, name, qp) - previous).square().mean().sqrt()
            if torch.isfinite(change) and change.item() <= maximum:
                return float(change), scale
        for p, old in zip(parameters, before):
            p.copy_(old)
    optimizer.load_state_dict(optimizer_before)
    raise RuntimeError("NPP update failed finite output RMS bound; parameters and optimizer restored")


def emit(log, row):
    if any(isinstance(v, float) and not torch.isfinite(torch.tensor(v)) for v in row.values()):
        raise RuntimeError("nonfinite research metric")
    line = json.dumps(row, allow_nan=False)
    print(line, flush=True)
    if log is not None:
        with Path(log).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def fit_proxy(npp, proxy, codec, samples, config, steps, out, stage):
    """Fit on actual-codec targets, including current-NPP distribution refreshes."""
    proxy.requires_grad_(True).train()
    npp.eval()
    optimizer = torch.optim.Adam(proxy.parameters(), lr=config["learning_rate"])
    groups = len(config["codecs"]) * len(config["qps"])
    cache = {}
    device = next(proxy.parameters()).device
    for step in range(steps):
        sample_index, name, qp = training_case(step, len(samples), config)
        # Changing source each complete grid gives every source all four QPs.
        key = (sample_index, name, qp)
        x = samples[sample_index][1].to(device)
        with torch.no_grad():
            y = npp(x, name, qp)
            if key not in cache:
                result = codec.roundtrip(y, name, qp)
                cache[key] = ((result.reconstruction.cpu() * 255).round().byte(), result.bpp.cpu())
                # Bound host-memory use, independently of dataset length.
                if len(cache) > 64:
                    cache.pop(next(iter(cache)))
            target, actual = cache[key]
            target, actual = target.to(device).float() / 255, actual.to(device)
        pred = proxy(y.detach(), name, qp)
        reconstruction = F.l1_loss(pred.reconstruction, target)
        rate = F.smooth_l1_loss((pred.bpp + 1e-8).log(), (actual + 1e-8).log())
        loss = 10 * reconstruction + rate
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 8 == 0 or step + 1 == steps:
            emit(Path(out) / "training.jsonl", {"stage": stage, "step": step + 1, "codec": name,
                 "qp": qp, "loss": float(loss.detach()), "reconstruction_l1": float(reconstruction.detach()),
                 "actual_bpp": float(actual.mean()), "proxy_bpp": float(pred.bpp.detach().mean())})
    proxy.requires_grad_(False).eval()
    # Clear optimizer-era gradients before the frozen proxy is reused.
    for parameter in proxy.parameters():
        parameter.grad = None
    save_checkpoint(Path(out) / f"{stage}.pt", npp, proxy, config, stage, steps)


def npp_step(npp, proxy, codec, tasks, x, record, refs, optimizer, weights, name="h264", qp=35,
             max_output_rms=None):
    if any(p.requires_grad for p in proxy.parameters()):
        raise ValueError("freeze proxy parameters before NPP optimization")
    if any(p.requires_grad for m in (tasks.ar, tasks.od) for p in m.parameters()):
        raise ValueError("freeze task parameters before NPP optimization")
    start = time.perf_counter()
    y = npp(x, name, qp)
    actual = codec.roundtrip(y, name, qp)
    predicted = proxy(y, name, qp)
    decoded = real_value_proxy_grad(predicted.reconstruction, actual.reconstruction)
    rate = real_value_proxy_grad(predicted.bpp, actual.bpp)
    terms = tasks.losses(decoded, record, refs)
    preservation = F.mse_loss(y, x)
    delta = y - x
    temporal = x.sum() * 0
    if x.shape[1] > 1:
        # Stabilize residuals in quiet regions without suppressing source motion.
        motion = (x[:, 1:] - x[:, :-1]).abs().mean(2, keepdim=True).detach()
        temporal = ((delta[:, 1:] - delta[:, :-1]).square() * (-10 * motion).exp()).mean()
    loss = (weights["rate"] * rate.mean()
            + weights["ar"] * (terms["ar_gt"] + .2 * terms["ar_source"])
            + weights["od"] * (terms["od_gt"] + .5 * terms["od_source"])
            + weights["pre"] * preservation + weights["temp"] * temporal)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite NPP loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(npp.parameters(), 1.0, error_if_nonfinite=True)
    if max_output_rms is None:
        optimizer.step()
        output_rms, step_scale = None, 1.
    else:
        output_rms, step_scale = bounded_optimizer_step(npp, optimizer, x, y.detach(), name, qp, max_output_rms)
    return {"loss": float(loss.detach()), "actual_bpp": float(actual.bpp.mean()),
            "proxy_bpp": float(predicted.bpp.detach().mean()), "npp_grad_norm": float(norm),
            "preservation": float(preservation.detach()), "temporal": float(temporal.detach()),
            "seconds": time.perf_counter() - start, "output_update_rms": output_rms, "step_scale": step_scale,
            **{k: float(v.detach()) for k, v in terms.items()}}


def fit_npp(npp, proxy, codec, tasks, samples, config, out):
    optimizer = torch.optim.Adam(npp.parameters(), lr=config["learning_rate"])
    device = next(npp.parameters()).device
    # Precompute source targets once; these are never recomputed on a candidate.
    references = [tasks.references(x.to(device), record, gt) for record, x, gt in samples]
    grid_size = len(config["codecs"]) * len(config["qps"])
    for step in range(config["npp_steps"]):
        refresh = config["proxy_refresh_every"]
        if step and refresh and step % refresh == 0:
            fit_proxy(npp, proxy, codec, samples, config, config["proxy_refresh_steps"], out,
                      f"proxy_refresh_{step}")
        npp.train()
        index, name, qp = training_case(step, len(samples), config)
        record, x, _ = samples[index]
        row = npp_step(npp, proxy, codec, tasks, x.to(device), record, references[index], optimizer,
                       config["loss_weights"], name, qp, max_output_rms=config.get("max_npp_update_rms"))
        emit(Path(out) / "training.jsonl", {"stage": "npp", "step": step + 1,
             "source": record["id"], "task": record["task"], "codec": name, "qp": qp, **row})
        if (step + 1) % 16 == 0:
            save_checkpoint(Path(out) / "last.pt", npp, proxy, config, "npp", step + 1)
    save_checkpoint(Path(out) / "last.pt", npp, proxy, config, "npp", config["npp_steps"])
