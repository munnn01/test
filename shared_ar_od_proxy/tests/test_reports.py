import importlib
import json

import pytest
import torch


def test_detection_agreement_penalizes_wrong_classes_and_missing_objects():
    evaluation = importlib.import_module("ar_od_proxy.evaluate")
    reference = {"boxes": torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]),
                 "labels": torch.tensor([1, 2])}
    prediction = {"boxes": reference["boxes"].clone(), "labels": torch.tensor([1, 3]),
                  "scores": torch.tensor([.9, .8])}
    assert evaluation.agreement(prediction, reference) == {"tp": 1, "fp": 1, "fn": 1, "f1": .5}


def test_official_coco_ap_uses_original_image_geometry(tmp_path):
    evaluation = importlib.import_module("ar_od_proxy.evaluate")
    annotation = {"info": {}, "images": [{"id": 7, "file_name": "x.jpg", "width": 100, "height": 50}],
                  "categories": [{"id": 17, "name": "cat"}],
                  "annotations": [{"id": 1, "image_id": 7, "category_id": 17,
                                   "bbox": [10, 5, 30, 20], "area": 600, "iscrowd": 0}]}
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(annotation))
    predictions = [{"image_id": 7, "category_id": 17, "bbox": [10., 5., 30., 20.], "score": .9}]
    result = evaluation.coco_metrics(path, [7], predictions)
    assert result["mAP"] == pytest.approx(1.)


def test_each_training_source_gets_all_eight_codec_qp_pairs():
    engine = importlib.import_module("ar_od_proxy.engine")
    cfg = {"codecs": ["h264", "h265"], "qps": [30, 35, 40, 45]}
    cases = [engine.training_case(i, 32, cfg) for i in range(256)]
    assert len(set(cases)) == 256
    assert {i for i, _, _ in cases} == set(range(32))
    assert len([c for c in cases if c[0] == 0]) == 8
