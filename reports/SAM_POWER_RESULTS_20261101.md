# SAM-power result check: 2026-11-01

All 4 controls trained768 updates and completed8 rate/AP points on64 VAL. All 4 workers then failed on Gaussian evaluation due to StopIteration; neither SAM arm began. Positive BD-rate means bitrate increased at matched AP.

| Task / codec | Control BD-rate | Final content rate median / p90 error | Finite direction |
|---|---:|---:|---:|
| is-h264 | +5.06% | 6.52% / 23.99% | 16/16 |
| is-h265 | +3.09% | 6.19% / 23.58% | 16/16 |
| od-h264 | +13.74% | 6.31% / 22.75% | 16/16 |
| od-h265 | +12.81% | 6.02% / 19.79% | 16/16 |

## Rate/task gradient norm ratio, median over sparse logged probes

| Worker | QP30 | QP35 | QP40 | QP45 |
|---|---:|---:|---:|---:|
| is-h264 | 2.0991 | 0.2554 | 0.0472 | 0.0273 |
| is-h265 | 5.2259 | 0.5737 | 0.0721 | 0.0232 |
| od-h264 | 4.7296 | 1.1164 | 0.1941 | 0.0481 |
| od-h265 | 4.5241 | 0.7180 | 0.2821 | 0.0427 |

Rate/task balance changes strongly by QP. This suggests future QP-specific objective tuning, but no causal improvement is established. Finish the unchanged three-arm comparison first.

## Verification

Downloaded report identities and source binding checked; all 4 BD-rates independently recomputed with SciPy PCHIP + numerical quadrature. The Gaussian failure was reproduced through actual processed_video and corrected by a frozen device anchor. Recovery preserves learned NPP/proxy/optimizer/RNG and reruns no control task updates. Each private predecessor is attached only to its own owner.
