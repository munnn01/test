# Shared AR + OD proxy pipeline

Approved scope: implement the architecture discussed in this chat, place it in one standalone folder, publish to munnn01/test, and execute a private Kaggle GPU pilot. Date: 2026-10-22, Asia/Bangkok.

One causal, codec/QP-conditioned NPP produces one RGB clip. A real x264/x265 encode/decode supplies exact forward reconstruction and paid elementary-stream bits. A recurrent learned proxy supplies reconstruction and rate derivatives through detached value replacement. Both frozen task networks consume the same decoded clip. Only NPP changes during the NPP optimization stage.

NPP: pointwise residual branch plus a compact spatial encoder/decoder and ConvGRU bottleneck, bounded RGB residual, identity initialization. Proxy: codec/QP-conditioned spatial analysis, recurrent temporal context, straight-through latent quantization, reconstruction synthesis and logistic entropy rate head. Context resets at each closed GOP; no B frames or future frames. Codec and proxy use the same clip boundary. This is a proposed surrogate, not an exact differentiable implementation of x264/x265.

Task supervision: Kinetics supplies AR labels and source-detector pseudo boxes on the same video; COCO supplies real boxes and no AR target. A COCO image remains T=1. Ground-truth and source-consistency results are named separately. Mixed Kinetics/COCO supervision does not establish simultaneous two-task GT performance on the same video. Evaluation reports actual bytes, AR accuracy, COCO mAP and on-video detection agreement separately. No automatic improvement claim or BD-rate extrapolation.

First Kaggle pilot: seed 20261022, frames=16, RGB size=256, width=12, both codecs, QP 30/35/40/45, 16 TRAIN sources per dataset and 8 DEV sources per dataset. Proxy fitting precedes NPP fitting; a short refresh uses current NPP outputs. Holdout source grouping is independent of annotation class and filesystem mount. Initial test is a development screen, not a confirmatory experiment.

Operational requirements: private notebook, free T4 request, one configured Kaggle account, no tokens or datasets in GitHub/notebook payloads, immutable published commit in the notebook, source hashes, streaming logs, actual checkpoints and decoded stream artifacts. External credentials are read only by the local launcher and passed through subprocess environment. Outputs are excluded from publication. Fail rather than silently replacing missing real data, pretrained weights or GPU with synthetic inputs.

Implementation isolation: the current workspace is not a Git repository. A fresh standalone folder and a new GitHub branch provide isolation; existing LAB checkouts and the default GitHub branch are preserved. The user explicitly authorized implementation, publishing and Kaggle submission, so no additional design or publication approval is needed.
