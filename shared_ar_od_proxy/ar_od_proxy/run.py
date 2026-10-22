"""Run a real-data mixed-label pilot, evaluate a checkpoint, or encode one clip."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from .codec import RealCodec
from .data import kinetics_records, coco_records, select_records, load_record
from .engine import fit_proxy, fit_npp, source_hashes, load_checkpoint, load_calibrated_start, save_checkpoint
from .evaluate import evaluate
from .models import VideoNPP, VideoProxy
from .tasks import TaskHeads


def configuration(path):
    target = Path(path) if path is not None else Path(__file__).parent / "configs/pilot.json"
    cfg = json.loads(target.read_text(encoding="utf-8"))
    if cfg["qps"] != [30, 35, 40, 45] or cfg["codecs"] != ["h264", "h265"]:
        raise ValueError("protocol requires H.264/H.265 and exactly QPs 30,35,40,45")
    if cfg["frames"] < 16 or cfg["size"] < 64 or cfg["size"] % 2 or cfg["width"] < 4:
        raise ValueError("invalid spatial/temporal model settings")
    if any(type(cfg[k]) is not int or cfg[k] < 1 for k in ("train_count", "dev_count", "proxy_steps", "npp_steps")):
        raise ValueError("positive source counts and training budgets required")
    if "max_npp_update_rms" in cfg and not 0 < cfg["max_npp_update_rms"] <= 1 / 255:
        raise ValueError("NPP output update RMS must be positive and at most 1/255")
    return cfg


def samples_for(records, cfg):
    result = []
    for record in records:
        x, gt = load_record(record, cfg["frames"], cfg["size"])
        result.append((record, x, gt))
    return result


def manifest_for(args, cfg):
    ar, od = kinetics_records(args.ar_root), coco_records(args.coco_root, args.annotations)
    manifest = {}
    for split, count in (("train", cfg["train_count"]), ("dev", cfg["dev_count"])):
        # Interleave datasets without inventing AR labels for still images.
        manifest[split] = list(itertools.chain.from_iterable(zip(select_records(ar, split, count),
                                                                select_records(od, split, count))))
    if {r["id"] for r in manifest["train"]} & {r["id"] for r in manifest["dev"]}:
        raise ValueError("TRAIN/DEV source leakage")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "train-calibrated", "evaluate", "encode"))
    parser.add_argument("--config", type=Path, help="Defaults to the packaged four-QP pilot config")
    parser.add_argument("--ar-root", type=Path)
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--codec", choices=("h264", "h265"), default="h264")
    parser.add_argument("--qp", type=int, choices=(30, 35, 40, 45), default=35)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="Explicit local execution; Kaggle never passes this flag")
    args = parser.parse_args()
    cfg = configuration(args.config)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required; use --allow-cpu explicitly for local work")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(2)
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    args.out.mkdir(parents=True)
    codec = RealCodec(preset=cfg["preset"], gop=cfg["frames"], fps=cfg["fps"])
    meta = {"config": cfg, "source_hashes": source_hashes(), "source_commit": os.environ.get("AROD_SOURCE_COMMIT"),
            "device": device, "torch": str(torch.__version__), "start_time_unix": time.time(),
            "complete": False, "task_weights": {"ar": "R3D_18_KINETICS400_V1", "od": "MobileNetV3_320_FPN_COCO_V1"}}
    status = args.out / "run.json"
    try:
        if args.action == "encode":
            if args.checkpoint is None or args.input is None:
                raise ValueError("encode requires --checkpoint and --input")
            npp, _, _ = load_checkpoint(args.checkpoint, cfg, device)
            x, _ = load_record({"task": "ar", "path": str(args.input)}, cfg["frames"], cfg["size"])
            with torch.no_grad():
                packet = codec.roundtrip(npp(x.to(device), args.codec, args.qp), args.codec, args.qp)
            path = args.out / ("clip.264" if args.codec == "h264" else "clip.265")
            path.write_bytes(packet.streams[0])
            meta.update(bytes=len(packet.streams[0]), bpp=float(packet.bpp[0]),
                        scope="One centered contiguous clip; not a full-video streaming wrapper")
        else:
            if not all((args.ar_root, args.coco_root, args.annotations)):
                raise ValueError("train/evaluate require real Kinetics and COCO paths")
            manifest = manifest_for(args, cfg)
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            status.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            tasks = TaskHeads(device=device, size=cfg["size"], od_frames=cfg["od_frames"])
            if args.action == "train-calibrated":
                if args.checkpoint is None:
                    raise ValueError("train-calibrated requires --checkpoint")
                npp, proxy, saved = load_calibrated_start(args.checkpoint, cfg, device)
                input_manifest = json.loads((args.checkpoint.parent / "manifest.json").read_text())
                if any([r["id"] for r in manifest[s]] != [r["id"] for r in input_manifest[s]]
                       for s in ("train", "dev")):
                    raise ValueError("calibration source-ID manifest differs from requested data")
                meta["warm_start"] = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    "stage": saved["stage"], "source_hashes": saved["source_hashes"], "config": saved["config"],
                    "proxy_frozen": True, "optimizer": "new Adam state; continue NPP weights"}
                save_checkpoint(args.out / "start.pt", npp, proxy, cfg, "calibrated_start", 0)
                dev = samples_for(manifest["dev"], cfg)
                start_out = args.out / "start_evaluation"
                start_out.mkdir()
                evaluate(npp, proxy, codec, tasks, dev, cfg, args.annotations, start_out)
                del dev
                train = samples_for(manifest["train"], cfg)
                fit_npp(npp, proxy, codec, tasks, train, cfg, args.out)
                del train
            elif args.action == "train":
                train = samples_for(manifest["train"], cfg)
                npp, proxy = VideoNPP(cfg["width"]).to(device), VideoProxy(cfg["width"]).to(device)
                fit_proxy(npp, proxy, codec, train, cfg, cfg["proxy_steps"], args.out, "proxy_warmup")
                fit_npp(npp, proxy, codec, tasks, train, cfg, args.out)
                del train
            else:
                if args.checkpoint is None:
                    raise ValueError("evaluate requires --checkpoint")
                npp, proxy, saved = load_checkpoint(args.checkpoint, cfg, device)
                if saved["source_hashes"] != source_hashes():
                    raise ValueError("evaluation source differs from training checkpoint")
            dev = samples_for(manifest["dev"], cfg)
            evaluate(npp, proxy, codec, tasks, dev, cfg, args.annotations, args.out)
        meta["complete"] = True
    except BaseException as error:
        meta["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        meta["wall_seconds"] = time.time() - meta["start_time_unix"]
        status.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"complete": meta["complete"], "output": str(args.out), "wall_seconds": meta["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
