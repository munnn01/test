import importlib
import json

import pytest
import torch
from PIL import Image


def test_source_partition_groups_all_segments_of_one_youtube_video():
    data = importlib.import_module("ar_od_proxy.data")
    a = data.video_source_id("/train/playing guitar/abcdefghijk_000001_000011.mp4")
    b = data.video_source_id("/test/other class/abcdefghijk_000031_000041.mp4")
    assert a == b == "kinetics/abcdefghijk"
    assert data.partition(a) == data.partition(b)


def test_coco_remains_single_frame_and_scales_boxes(tmp_path):
    data = importlib.import_module("ar_od_proxy.data")
    path = tmp_path / "000000000001.jpg"
    Image.new("RGB", (100, 50), (80, 90, 100)).save(path)
    ann = {"images": [{"id": 1, "file_name": path.name, "width": 100, "height": 50}],
           "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 5, 30, 20],
                            "category_id": 17, "iscrowd": 0}], "categories": []}
    ap = tmp_path / "instances.json"
    ap.write_text(json.dumps(ann))
    rec = data.coco_records(tmp_path, ap)[0]
    clip, targets = data.load_record(rec, frames=16, size=64)
    assert clip.shape == (1, 1, 3, 64, 64)
    assert rec.get("ar_label") is None
    torch.testing.assert_close(targets["boxes"], torch.tensor([[6.4, 6.4, 25.6, 32.0]]))
    assert targets["labels"].tolist() == [17]


def test_missing_source_fails_instead_of_synthetic_fallback(tmp_path):
    data = importlib.import_module("ar_od_proxy.data")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        data.load_record({"task": "ar", "path": str(tmp_path / "missing.mp4")}, 16, 64)


def test_detector_training_loss_keeps_frozen_weights_and_bn_unchanged():
    tasks = importlib.import_module("ar_od_proxy.tasks")
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    detector = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None,
                                               min_size=64, max_size=64,
                                               rpn_pre_nms_top_n_train=64,
                                               rpn_post_nms_top_n_train=32).eval().requires_grad_(False)
    before = {n: b.clone() for n, b in detector.named_buffers()}
    x = torch.rand(3, 64, 64, requires_grad=True)
    target = {"boxes": torch.tensor([[10., 12., 40., 50.]]), "labels": torch.tensor([1])}
    loss = tasks.detection_loss(detector, [x], [target])
    loss.backward()
    assert torch.isfinite(loss) and x.grad.abs().sum() > 0
    assert not detector.training and all(p.grad is None for p in detector.parameters())
    for name, buffer in detector.named_buffers():
        torch.testing.assert_close(buffer, before[name])


def test_identical_dense_detections_have_perfect_source_agreement():
    tasks = importlib.import_module("ar_od_proxy.tasks")
    evaluation = importlib.import_module("ar_od_proxy.evaluate")
    prediction = {"boxes": torch.tensor([[i * 12., 0., i * 12. + 10., 10.] for i in range(21)]),
                  "labels": torch.ones(21, dtype=torch.long), "scores": torch.full((21,), .9)}

    class SourceHeads(tasks.TaskHeads):
        def __init__(self):
            self.od_frames = 1

        def detections(self, x, indices=None):
            return [prediction]

        def ar_logits(self, x):
            return torch.zeros(1, 400)

    refs = SourceHeads().references(torch.zeros(1, 16, 3, 32, 32), {"task": "ar"}, None)
    assert evaluation.agreement(prediction, refs["od"][0])["f1"] == 1.0
