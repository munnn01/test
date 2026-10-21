"""Real Kinetics/COCO sources, source-group splits and explicit label masks."""
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import torch


def canonical(name):
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def video_source_id(path):
    stem = Path(path).stem
    match = re.match(r"^([A-Za-z0-9_-]{11})(?:$|_\d+)", stem)
    return "kinetics/" + (match.group(1) if match else stem)


def partition(identifier):
    bucket = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "dev" if bucket == 8 else "test"


def kinetics_records(root):
    from torchvision.models.video import R3D_18_Weights
    names = R3D_18_Weights.KINETICS400_V1.meta["categories"]
    mapping = {canonical(name): i for i, name in enumerate(names)}
    records, seen = [], set()
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in (".mp4", ".avi", ".webm", ".mkv"):
            continue
        category = next((canonical(p.name) for p in path.parents if canonical(p.name) in mapping), None)
        if category is None:
            continue
        source = video_source_id(path)
        if source in seen:
            continue
        seen.add(source)
        records.append({"id": source, "task": "ar", "path": str(path), "ar_label": mapping[category]})
    if not records:
        raise ValueError(f"no Kinetics videos with canonical K400 labels in {root}")
    return records


def coco_records(root, annotations):
    meta = json.loads(Path(annotations).read_text(encoding="utf-8"))
    objects = {}
    for ann in meta["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w > 0 and h > 0:
            objects.setdefault(ann["image_id"], []).append(([x, y, x + w, y + h], ann["category_id"]))
    return [{"id": f"coco2017/{im['id']}", "task": "od", "path": str(Path(root) / im["file_name"]),
             "image_id": im["id"], "width": im["width"], "height": im["height"],
             "objects": objects.get(im["id"], [])} for im in meta["images"]]


def select_records(records, split, count):
    chosen = [r for r in records if partition(r["id"]) == split]
    chosen.sort(key=lambda r: hashlib.sha256(("aro-proxy-v1/" + r["id"]).encode()).hexdigest())
    if count < 1 or len(chosen) < count:
        raise ValueError(f"need {count} {split} records, found {len(chosen)}")
    result = chosen[:count]
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("duplicate source groups")
    return result


def load_record(record, frames, size):
    if size < 32 or size % 2 or frames < 1:
        raise ValueError("size must be even and >=32; frames must be positive")
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    targets = None
    if record["task"] == "od":
        with Image.open(path) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError("COCO image dimensions differ from annotations")
            array = np.array(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))[None]
        boxes, labels = [], []
        for box, label in record["objects"]:
            scaled = np.array(box, dtype=np.float32) * ([size / record["width"], size / record["height"]] * 2)
            scaled = np.clip(scaled, 0, size)
            if scaled[2] > scaled[0] and scaled[3] > scaled[1]:
                boxes.append(scaled.tolist())
                labels.append(label)
        targets = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                   "labels": torch.tensor(labels, dtype=torch.int64)}
    elif record["task"] == "ar":
        cap = cv2.VideoCapture(str(path))
        result = []
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {path}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, (total - frames) // 2))
            for _ in range(frames):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"video has fewer than {frames} decodable contiguous frames: {path}")
                result.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size)))
        finally:
            cap.release()
        array = np.stack(result)
    else:
        raise ValueError("unknown record task")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float()[None] / 255
    return tensor, targets
