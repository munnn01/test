# A3 OT: prospective sequence-disjoint source reservation

This prepares sources only. No new tracker policy has been frozen or measured yet. A2 MOT20 results remain unchanged. All raw frame download/hash work happens in a private Kaggle CPU job; no local model/codec execution occurs.

The official [MOT17 archive](https://motchallenge.net/data/MOT17/) currently publishes downloads and archived results; official submission/evaluation is closed. This project uses public TRAIN annotations for an internal prospective split, not an official challenge TEST claim. The three detector variants share physical videos, so only the `FRCNN` directory is used once per video. Supplied detector text files are ignored: our frozen detector must infer every source/decoded frame.

| Internal split | Physical sequences | Contiguous prefix |
|---|---|---|
| FIT DEV | MOT17-02, MOT17-05 | frames1–64 each |
| Validation DEV | MOT17-04, MOT17-11 | frames1–64 each |
| Reserved TEST | MOT17-09, MOT17-10, MOT17-13 | frames1–512 each |

Exactly QP30/35/40/45 on H.264/H.265 remain required. Sequence cadence/geometry come from original `seqinfo.ini`; later codec/tracker recipes must explicitly account for the native14/25/30fps. No sequence may cross these splits. Source protocol [configs/v34r1-a3-mot17-sources.json.gz](../configs/v34r1-a3-mot17-sources.json.gz) freezes original archive byte length5,860,214,001, selected ZIP offsets/sizes/CRC and seqinfo SHA before source inference. Its decompressed SHA256 is `a18e3455002e52ac1225e7fab34f54fc9a14a16405fe29483ad912358b1f6143`.

The Kaggle source job fetches only declared members, checks ZIP CRC/size, rejects exact duplicated source JPEGs and saves per-file SHA plus a private cache. Raw videos/images/GT are excluded from the public Git repository. Future results must distinguish source-track pseudo-HOTA from GT HOTA; use cloned common past state, commit one selected trial, and preserve all4 raw curve points. Source preparation alone is no evidence of bitrate savings or target attainment.
