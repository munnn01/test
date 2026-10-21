"""Frozen pretrained task heads with differentiable input paths."""
import torch
from torch import nn
import torch.nn.functional as F


def detection_loss(model, images, targets):
    # TorchVision selects pre-NMS detector losses in training mode. Preserve all
    # module modes and prevent batch-normalization state updates while doing so.
    modes = {module: module.training for module in model.modules()}
    try:
        model.train()
        for module in model.modules():
            if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
                module.eval()
        return sum(model(images, targets).values())
    finally:
        for module, training in modes.items():
            module.training = training


class TaskHeads:
    def __init__(self, device="cuda", size=256, pretrained=True, od_frames=1):
        from torchvision.models.video import r3d_18, R3D_18_Weights
        from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
                                                   FasterRCNN_MobileNet_V3_Large_320_FPN_Weights)
        self.device, self.od_frames = device, od_frames
        self.ar = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None).to(device).eval()
        self.od = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 if pretrained else None,
            weights_backbone=None, min_size=size, max_size=size,
            rpn_pre_nms_top_n_train=600, rpn_post_nms_top_n_train=200,
            rpn_pre_nms_top_n_test=600, rpn_post_nms_top_n_test=200).to(device).eval()
        self.ar.requires_grad_(False)
        self.od.requires_grad_(False)
        self.mean = torch.tensor([.43216, .394666, .37645], device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([.22803, .22145, .216989], device=device).view(1, 3, 1, 1, 1)

    def ar_logits(self, x):
        if x.shape[1] < 16:
            raise ValueError("AR requires at least 16 decoded frames; COCO has no AR target")
        indices = torch.linspace(0, x.shape[1] - 1, 16, device=x.device).long()
        frames = x[:, indices]
        b, t, c, h, w = frames.shape
        frames = F.interpolate(frames.reshape(b * t, c, h, w), (112, 112), mode="bilinear", align_corners=False)
        frames = frames.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        return self.ar((frames - self.mean) / self.std)

    def indices(self, x):
        if self.od_frames == 1:
            return [x.shape[1] // 2]
        return torch.linspace(0, x.shape[1] - 1, min(self.od_frames, x.shape[1])).long().tolist()

    @torch.no_grad()
    def detections(self, x, indices=None):
        if x.shape[0] != 1:
            raise ValueError("task adapter currently requires batch size 1")
        self.od.eval()
        return self.od([x[0, i] for i in (self.indices(x) if indices is None else indices)])

    @torch.no_grad()
    def references(self, x, record, gt):
        if record["task"] == "od":
            return {"od": [{k: v.to(x.device) for k, v in gt.items()}], "indices": [0]}
        refs = []
        for det in self.detections(x):
            keep = (det["scores"] >= .5).nonzero().flatten()
            refs.append({"boxes": det["boxes"][keep].detach(), "labels": det["labels"][keep].detach()})
        return {"ar_logits": self.ar_logits(x).detach(), "od": refs, "indices": self.indices(x)}

    def losses(self, x, record, refs):
        zero = x.sum() * 0
        terms = {name: zero for name in ("ar_gt", "ar_source", "od_gt", "od_source")}
        if record["task"] == "ar":
            logits = self.ar_logits(x)
            terms["ar_gt"] = F.cross_entropy(logits, torch.tensor([record["ar_label"]], device=x.device))
            terms["ar_source"] = F.kl_div(logits.log_softmax(-1), refs["ar_logits"].softmax(-1), reduction="batchmean")
        od = detection_loss(self.od, [x[0, i] for i in refs["indices"]], refs["od"])
        terms["od_gt" if record["task"] == "od" else "od_source"] = od
        return terms
