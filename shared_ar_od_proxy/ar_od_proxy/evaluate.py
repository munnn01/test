"""Actual-bitstream DEV curves; GT accuracy and source agreement stay separate."""
from collections import defaultdict
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torchvision.ops import box_iou


def agreement(prediction, reference, threshold=.5):
    keep = prediction["scores"] >= .5
    boxes, labels = prediction["boxes"][keep], prediction["labels"][keep]
    matched, tp = set(), 0
    if len(boxes) and len(reference["boxes"]):
        ious = box_iou(boxes, reference["boxes"])
        for i in range(len(boxes)):
            candidates = [j for j in range(len(reference["boxes"])) if j not in matched
                          and labels[i] == reference["labels"][j] and ious[i, j] >= threshold]
            if candidates:
                j = max(candidates, key=lambda j: float(ious[i, j]))
                matched.add(j)
                tp += 1
    fp, fn = len(boxes) - tp, len(reference["boxes"]) - tp
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / denominator if denominator else 1.0}


def coco_metrics(annotation, ids, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    # Keep progress logs concise; retain the numeric official metrics in JSON.
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
        if predictions:
            result = gt.loadRes(predictions)
        else:
            result = COCO()
            result.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
            result.createIndex()
        evaluator = COCOeval(gt, result, "bbox")
        evaluator.params.imgIds = list(ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
    return {name: float(v) if v >= 0 else None for name, v in zip(names, evaluator.stats[:6])}


@torch.no_grad()
def evaluate(npp, proxy, codec, tasks, samples, config, annotation, out):
    out = Path(out)
    stream_dir = out / "streams"
    stream_dir.mkdir(exist_ok=True)
    device = next(npp.parameters()).device
    npp.eval()
    proxy.eval()
    rows, detections = [], defaultdict(list)
    for sample_number, (record, cpu, gt) in enumerate(samples):
        x = cpu.to(device)
        refs = tasks.references(x, record, gt)
        for name in config["codecs"]:
            for qp in config["qps"]:
                for arm in ("identity", "npp"):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    y = x if arm == "identity" else npp(x, name, qp)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    preprocessing_seconds = time.perf_counter() - started
                    actual = codec.roundtrip(y, name, qp)
                    decoded = actual.reconstruction  # The SAME tensor goes to both heads.
                    predicted = proxy(y, name, qp)
                    extension = "264" if name == "h264" else "265"
                    stream_name = f"{sample_number:04d}_{name}_q{qp}_{arm}.{extension}"
                    (stream_dir / stream_name).write_bytes(actual.streams[0])
                    row = {"source": record["id"], "task": record["task"], "codec": name, "qp": qp,
                           "arm": arm, "bytes": len(actual.streams[0]), "bpp": float(actual.bpp[0]),
                           "stream": "streams/" + stream_name,
                           "stream_sha256": hashlib.sha256(actual.streams[0]).hexdigest(),
                           "decoded_sha256": hashlib.sha256((decoded.cpu() * 255).round().byte().numpy().tobytes()).hexdigest(),
                           "preprocessing_seconds": preprocessing_seconds, "codec_seconds": actual.seconds,
                           "proxy_bpp": float(predicted.bpp[0]),
                           "proxy_reconstruction_l1": float((predicted.reconstruction - decoded).abs().mean())}
                    outputs = tasks.detections(decoded, refs["indices"])
                    if record["task"] == "ar":
                        logits = tasks.ar_logits(decoded)
                        row["ar_correct"] = int(logits.argmax(-1).item() == record["ar_label"])
                        row["ar_top5_correct"] = int(record["ar_label"] in logits.topk(5).indices[0].tolist())
                        counts = [agreement(prediction, target) for prediction, target in zip(outputs, refs["od"])]
                        row["od_source_agreement"] = {k: sum(c[k] for c in counts) for k in ("tp", "fp", "fn")}
                        row["simultaneous_od_gt_available"] = False
                    else:
                        row["image_id"] = record["image_id"]
                        prediction = outputs[0]
                        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
                            if score < .001:
                                continue
                            x1, y1, x2, y2 = box.cpu().tolist()
                            sx, sy = record["width"] / config["size"], record["height"] / config["size"]
                            detections[(name, qp, arm)].append({"image_id": record["image_id"],
                                "category_id": int(label), "score": float(score),
                                "bbox": [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]})
                    rows.append(row)
                    with (out / "evaluation.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({"stage": "evaluation", "completed_sources": sample_number + 1, "total": len(samples)}), flush=True)
    summary = {"scope": "DEV pilot; Kinetics AR GT plus OD source agreement, separate COCO image GT",
               "joint_ar_od_gt_confirmed": False, "bd_rate": None, "curves": [],
               "limitations": ["No simultaneous AR+OD ground-truth video set", "Small pilot cohort",
                               "Pretrained-backbone exposure to original dataset training splits is not excluded"]}
    image_ids = [r[0]["image_id"] for r in samples if r[0]["task"] == "od"]
    for name in config["codecs"]:
        for qp in config["qps"]:
            for arm in ("identity", "npp"):
                group = [r for r in rows if (r["codec"], r["qp"], r["arm"]) == (name, qp, arm)]
                video = [r for r in group if r["task"] == "ar"]
                images = [r for r in group if r["task"] == "od"]
                tp, fp, fn = [sum(r["od_source_agreement"][k] for r in video) for k in ("tp", "fp", "fn")]
                item = {"codec": name, "qp": qp, "arm": arm,
                        "ar_video_bpp": float(np.mean([r["bpp"] for r in video])),
                        "coco_image_bpp": float(np.mean([r["bpp"] for r in images])),
                        "ar_top1": float(np.mean([r["ar_correct"] for r in video])),
                        "ar_top5": float(np.mean([r["ar_top5_correct"] for r in video])),
                        "od_video_source_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                        "coco": coco_metrics(annotation, image_ids, detections[(name, qp, arm)])}
                summary["curves"].append(item)
    (out / "coco_predictions.json").write_text(json.dumps({f"{c}_q{q}_{a}": p for (c, q, a), p in detections.items()}), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
