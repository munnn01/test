# A7 bounded probe: stabilize H.264 choices across four QPs

User2026-10-18 amended success to at least one valid codec BD strictly below−15% per task and additionally requested improving undefined codecs. Preserve A6 and its original two-codec gate; annotate the later criterion separately. A6 H265 is−16.4884% on reusedCAL; H264 is undefined. All real work remains Kaggle-only, QP30/35/40/45 exactly.

The H264 inversion is QP30HOTA53.479675 versus QP35HOTA53.938814; most reversal occurs on MOT17-11. Many selected profiles differ between QPs. Packet/pixel/old prediction and tracker controls pass, so this is an observed task-quality curve, not evidence of an I/O bug. Profile switching is a hypothesis; residual detector/tracker noise may remain even after coupling.

Probe one change: select one common profile among the same21 actions for all four QPs in each16-frame window. Keep four separate native-FPS causal tracker states, clone all trials from the corresponding common past, choose using mean log paid ratio plus lambda times mean identity-relative source-HOTA damage, and commit that profile once to each QP state. Require paid<=identity at every QP; identity always available. Empty source reference forces identity. No GT, future, curve sorting, smoothing, point removal, or interpolation repair enters the sender.

Use the fixed lambda grid0/1/2/4/8/16/32 on H264 FIT02/05 only. A deployable candidate must have valid raw FIT BD and negative endpoint-scaled FIT utility; choose best utility, smaller lambda tie; otherwise identity. Lock policy/hash before exactly one reusedCAL04/11 replay. Report all FIT results and raw four points, including failures. H265 remains delegated to the frozen A6 lambda32 policy as a separately identified reference. Verify old A6 H264 independent32 and anchor tracker files/metrics exactly using its authenticated archive.

The common-profile policy requires counterfactual candidates at all four QPs even for a request to transmit one QP:84 measured candidates per window/codec. Offline caches already contain these, so this probe has no new encodes or detector inference, but deployment cost is not free. Report this cost explicitly.

This is development on four consumed videos, not independent confirmation. It may fail to restore GT monotonicity; never promise defined BD. The frozen8-video source reservation remains5 MOT17 plus3 metadata-ranked DanceTrack videos, with old2/new6 reported separately. Under the user's amended criterion, unchanged A6 H265 qualifies for expansion; any A7 policy must be frozen before new-video GT evaluation. No retrospective success criterion is called prespecified.

Execution: pure choice/goal tests RED→GREEN; one read-only review of the CPU execution unit; commit/push immutable source; private Kaggle CPU reuse the authenticated cache dataset; audit saved arithmetic and raw metrics, publish all outcomes. Then execute the already authorized source-freeze and8-video evaluation separately.
