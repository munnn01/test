"""Proxy calibration, frozen-task optimization and checkpoint contracts."""
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .models import VideoNPP, VideoProxy, real_value_proxy_grad


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
    torch.save({"format": 1, "npp": npp.state_dict(), "proxy": proxy.state_dict(),
                "config": config, "stage": stage, "step": step, "source_hashes": source_hashes()}, temporary)
    temporary.replace(path)


def load_checkpoint(path, config=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != 1:
        raise ValueError("unsupported checkpoint format")
    stored = checkpoint["config"]
    if config is not None and stored != config:
        raise ValueError("checkpoint config differs from requested config")
    npp, proxy = VideoNPP(stored["width"]).to(device), VideoProxy(stored["width"]).to(device)
    npp.load_state_dict(checkpoint["npp"], strict=True)
    proxy.load_state_dict(checkpoint["proxy"], strict=True)
    return npp, proxy, checkpoint


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


def npp_step(npp, proxy, codec, tasks, x, record, refs, optimizer, weights, name="h264", qp=35):
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
    optimizer.step()
    return {"loss": float(loss.detach()), "actual_bpp": float(actual.bpp.mean()),
            "proxy_bpp": float(predicted.bpp.detach().mean()), "npp_grad_norm": float(norm),
            "preservation": float(preservation.detach()), "temporal": float(temporal.detach()),
            "seconds": time.perf_counter() - start,
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
                       config["loss_weights"], name, qp)
        emit(Path(out) / "training.jsonl", {"stage": "npp", "step": step + 1,
             "source": record["id"], "task": record["task"], "codec": name, "qp": qp, **row})
        if (step + 1) % 16 == 0:
            save_checkpoint(Path(out) / "last.pt", npp, proxy, config, "npp", step + 1)
    save_checkpoint(Path(out) / "last.pt", npp, proxy, config, "npp", config["npp_steps"])
