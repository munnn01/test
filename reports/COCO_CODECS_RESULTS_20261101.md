# COCO H264/H265 results — 2026-11-01

Negative BD-rate is better than each codec-only baseline at matched AP. H265 rows are diagnostic checkpoints from aborted runs, not completed matched training.

| Task / codec | Control steps | Improved steps | Control BD-rate | Improved BD-rate | Matched schedule complete |
|---|---:|---:|---:|---:|---|
| [is-h264](https://www.kaggle.com/code/tranthihongdieu/coco-is-h264-paired-20261101) | 768 | 768 | +5.75% | +12.67% | True |
| [is-h265](https://www.kaggle.com/code/dngbolm/coco-is-h265-paired-20261101) | 100 | 550 | +67.64% | +53.42% | False |
| [od-h264](https://www.kaggle.com/code/hoangminhhuy123/coco-od-h264-paired-20261101) | 768 | 768 | +8.03% | -2.98% | True |
| [od-h265](https://www.kaggle.com/code/ngynanhthuw/coco-od-h265-paired-20261101) | 50 | 50 | +75.26% | +46.32% | False |

## is-h264

| Arm | Last pre-codec dev PSNR | Rate-gradient alignment | Proxy/real rate ratio range |
|---|---:|---:|---:|
| control | 22.33 dB | 8/8 (ties 0) | 0.778–3.054 |
| foreground | 23.08 dB | 8/8 (ties 0) | 0.778–3.054 |

| QP | Codec bpp / AP | Control bpp / AP | Improved bpp / AP |
|---|---:|---:|---:|
| 30 | 0.8265 / 45.85 | 0.4614 / 42.16 | 0.4714 / 42.48 |
| 35 | 0.4848 / 42.70 | 0.2646 / 39.40 | 0.2705 / 38.40 |
| 40 | 0.2784 / 39.80 | 0.1592 / 32.21 | 0.1631 / 32.92 |
| 45 | 0.1521 / 32.70 | 0.0977 / 22.55 | 0.1005 / 23.95 |

## is-h265

| Arm | Last pre-codec dev PSNR | Rate-gradient alignment | Proxy/real rate ratio range |
|---|---:|---:|---:|
| control | 19.35 dB | 8/8 (ties 0) | 1.889–14.589 |
| foreground | 17.08 dB | 8/8 (ties 0) | 1.889–14.589 |

| QP | Codec bpp / AP | Control bpp / AP | Improved bpp / AP |
|---|---:|---:|---:|
| 30 | 0.7478 / 45.69 | 0.6715 / 40.28 | 0.6588 / 38.37 |
| 35 | 0.4576 / 44.81 | 0.3825 / 37.30 | 0.3583 / 37.37 |
| 40 | 0.2779 / 40.11 | 0.2232 / 32.87 | 0.2113 / 30.99 |
| 45 | 0.1677 / 32.90 | 0.1344 / 19.82 | 0.1330 / 22.21 |

## od-h264

| Arm | Last pre-codec dev PSNR | Rate-gradient alignment | Proxy/real rate ratio range |
|---|---:|---:|---:|
| control | 21.57 dB | 8/8 (ties 0) | 0.778–3.054 |
| foreground | 21.44 dB | 8/8 (ties 0) | 0.778–3.054 |

| QP | Codec bpp / AP | Control bpp / AP | Improved bpp / AP |
|---|---:|---:|---:|
| 30 | 0.8265 / 51.08 | 0.4225 / 46.78 | 0.4237 / 44.88 |
| 35 | 0.4848 / 49.78 | 0.2438 / 43.16 | 0.2433 / 43.59 |
| 40 | 0.2784 / 45.42 | 0.1490 / 34.34 | 0.1483 / 36.79 |
| 45 | 0.1521 / 34.70 | 0.0934 / 23.00 | 0.0933 / 23.11 |

## od-h265

| Arm | Last pre-codec dev PSNR | Rate-gradient alignment | Proxy/real rate ratio range |
|---|---:|---:|---:|
| control | 19.44 dB | 8/8 (ties 0) | 1.889–14.590 |
| foreground | 19.90 dB | 8/8 (ties 0) | 1.889–14.590 |

| QP | Codec bpp / AP | Control bpp / AP | Improved bpp / AP |
|---|---:|---:|---:|
| 30 | 0.7478 / 51.33 | 0.6479 / 46.91 | 0.6549 / 47.32 |
| 35 | 0.4576 / 50.61 | 0.3844 / 43.14 | 0.3829 / 44.47 |
| 40 | 0.2779 / 45.60 | 0.2309 / 37.43 | 0.2279 / 40.47 |
| 45 | 0.1677 / 39.34 | 0.1413 / 30.40 | 0.1392 / 30.92 |

## Checks and limitations

Shared warm-up/proxy hashes, task weights and cohorts match within each pair. Byte-based bpp checked. Valid PCHIP BD-rates recomputed and independently checked by numerical quadrature.

- 64-image pilot; no confidence interval or repeated seeds.
- H265 arms stopped below20dB pre-codec fidelity: diagnostic BD-rates are not completed matched768-step results.
- Individual BD-rates integrate different overlapping AP intervals; do not subtract them as a direct SAM effect.
- Gradient directions measured on two original TRAIN-dev images, not NPP outputs; do not establish gradient fidelity after adaptation.

## Additional overhead diagnostic

A CPU flat-image probe at QP45 produced H265:128px2350bytes/1.1475bpp,512px2426bytes/0.0740bpp. H264:128px620bytes/0.3027bpp,512px687bytes/0.0210bpp. The nearly fixed byte cost strongly affects128px proxy fitting. This is a flat-stream overhead diagnostic, not exact header separation or a Kaggle-version parity claim.

Hypothesis: fitting content entropy to128px paid rates absorbs stream overhead into the differentiable content estimate; this helps explain full-resolution H265 proxy rates around1.1bpp versus actual~0.1bpp. It does not by itself prove rate gradients caused the collapse. Before another NPP sweep, fit/validate rate at deployment geometry, model paid fixed overhead separately from content gradients, probe paired x/NPP(x), and measure rate/task/fidelity gradient norms.
