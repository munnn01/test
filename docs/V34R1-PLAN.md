# V34r1 implementation and verification plan

Request: clone all source branches into D:/TEST, evaluate and improve AR, OD and OT, use AutoResearchClaw and primary papers, verify individual blocks, and publish the improved v34r1 branch to github.com/munnn01/test.

The prospective target is PCHIP task-quality BD-rate strictly below -15% on each of AR/OD/OT × H.264/H.265. Freeze **QP 30,35,40,45 only**, matching the user's latest instruction. Retain the original anchor's codec, preset, GOP, input geometry, task weights and paid header accounting. Do not select QP subsets or interpolators after seeing new TEST results.

## Design

Keep published v34 runtime bytes and artifact provenance intact. Implement a separate `adaptive_vcm.v34r1` namespace: strict endpoint assessment, bounded multiscale filter candidates, source-only OD scoring, candidate selection with identity fallback and explicit cost accounting, and stateful tracking consistency. Reuse frozen primary controllers as baseline/shortlist proposals; extra fixed filter candidates are not newly trained neural outputs. Ground truth is used only in declared TRAIN/DEV selection and final scoring, never sender candidate choice on TEST.

## Work units

1. Inventory all 29 source branches and verify existing published artifact hashes; prepare CUDA/test runtime and explicit data manifests.
2. Add strict four-QP/six-endpoint metrics with failing tests for the exact -15 boundary, missing endpoints, nonmonotone curves, no overlap, and weak CIs. Preserve the legacy -10 protocol for historical reproduction.
3. Add separately versioned stronger geometry-preserving region filters; test identity, finite uint8 shape, bounded edits and causal masks. Record the exact candidate family and configuration hash.
4. Correct source-only OD behavior in the new namespace. Test missing/permuted GT invariance. Add source-only candidate guards with actual measured rate, finite checks, fallback and evaluation cost; test selection contracts before benchmarks.
5. Add temporal tracking-state snapshots and commit-only-selected behavior; verify candidate order invariance, boundary associations, and sequence reset.
6. Run block tests and real x264/x265 roundtrips; evaluate TRAIN/DEV candidate recall and freeze the final policy before TEST. Score AR top-1, official COCO AP, and pinned TrackEval HOTA/IDF1/MOTA. Retain raw paired rows and diagnostics.
7. Run the six prospective endpoints if data/compute allow. Compute paired uncertainty at video/image/sequence level, and distinguish point attainment from an upper CI strictly below -15. A missing or failing endpoint prevents an all-task success claim.
8. Publish tested code, reproducible instructions, cited research and actual results/limitations on the new v34r1 branch. Never overwrite or relabel historical evidence as a new successful run.

## Data scope

The user-supplied Kaggle dataset `qktttttttttt/kineticscleaned`, version 4, contains 10,843 MP4 files under `train/` and one removal CSV (16,485,084,692 bytes). Its path is not evidence of official independent K400 validation membership. Freeze IDs, category mapping, exclusion history and split provenance before claims. COCO and MOT sources are obtained from their official distribution or traceable public mirrors. Raw data and task weights stay outside Git.

## Review focus

- Source-only OD must work without GT and must not change its selected bitstream when GT changes.
- Additional filter candidates must preserve geometry and identity and pay actual bitstream/header costs.
- Four-QP protocol and all six endpoints must be required by the success report.
- Temporal candidate trials start from equal tracker state; only the chosen trial commits state.
- Existing published runtime/checkpoint/result hashes remain verifiable.
