# Completed SAM recovery results: 2026-11-02

All 4 workers COMPLETE; all 12 learned-arm evaluations complete; all 8 new SAM arms trained 768 updates. Native BD-rate recomputed with PCHIP analytical integration and numerical quadrature for 20 curves.

| Task / codec | NPP control | SAM foreground | SAM foreground + boundary | Gaussian 0.5 | Gaussian 0.8 |
|---|---:|---:|---:|---:|---:|
| is-h264 | +5.06% | -7.52% | +16.85% | -16.96% | -7.52% |
| is-h265 | +3.09% | +7.10% | -0.35% | -2.44% | -6.49% |
| od-h264 | +13.74% | +9.99% | +10.10% | -5.23% | -9.73% |
| od-h265 | +12.81% | +17.46% | +32.38% | +8.07% | -11.71% |

Negative BD-rate saves bits at matched AP; positive BD-rate increases bits. Same 64 COCO VAL images,4 QPs 30/35/40/45, independent one-frame streams. Each native value uses that method overlap with codec-only.

## Common AP interval across every method

| Task / codec | NPP control | SAM foreground | SAM + boundary | Gaussian 0.5 | Gaussian 0.8 |
|---|---:|---:|---:|---:|---:|
| is-h264 | +4.87% | -4.98% | +16.85% | -11.67% | -3.98% |
| is-h265 | +2.51% | +5.92% | -0.35% | -6.34% | -9.02% |
| od-h264 | +11.60% | +9.99% | +7.27% | -1.85% | -8.33% |
| od-h265 | +12.40% | +17.46% | +32.27% | +8.18% | -13.65% |

Best Gaussian remains better than every learned arm on the common AP interval for all 4 configurations. Foreground SAM saves bits only on IS-H264; boundary SAM saves a very small amount on IS-H265. No learned arm reaches -15% native BD-rate. Gaussian 0.5 IS-H264 reaches -16.96% on this subset only.

## Integrity and scope

Source hashes match submitted payloads; cloud exit0 and complete study flags checked. All arms share exact warmNPP, frozen proxy, cohort and task identities within each worker. All 20 codec-only curves match the original control anchors exactly.20 BD-rates recomputed; analytical integration agrees with numerical quadrature. Both proxy gates pass on all 12 learned arms. Every SAM cache contains 64 sources; guidance has nonzero coverage, and each trained SAM NPP has nonzero parameter change.

Control was preserved rather than retrained. No new training jobs launched in this result check. Original NPP and 4 QPs retained. Tracking remains deferred. This is a short 768-update, single-seed 64 VAL study; not a complete paper reproduction. No confidence intervals were estimated; the -0.35% IS-H265 boundary value should not be treated as significant improvement.
