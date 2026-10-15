# AutoResearchClaw ACP review

Actual output from researchclaw.llm.acp_client.ACPClient using the existing Codex login, 2026-10-15. This is a design critique, not experimental evidence. The summary was explicitly approved by the user before transmission. The broader CLI pipeline attempt stopped at TOPIC_INIT after an ACP timeout; it did not execute or validate experiments.

Three largest remaining risks

1. Source consistency does not establish GT correctness. Matching source detections and features removes the unintended GT dependency, but can preserve source errors or penalize beneficial changes. The anchor-plus-slack constraint and identity fallback provide no guarantee of COCO mAP, HOTA, action accuracy, or bitrate improvement.

2. Temporal validity and generalization remain weak. Unverified K400 JPEG cadence could change action inputs and the meaning of GOP16. MOT20-01’s first 256 frames provide a narrow tracking test; correlated frames cannot substitute for independent sequences. Fresh TEST data reduces tuning leakage but does not resolve these limitations.

3. The strict six-target objective remains vulnerable to noisy curves and selection effects. Four QPs, noisy OD DEV measurements, and limited quality overlap can make BD-rate unstable or undefined. Rejecting curves can bias conclusions if failed cases disappear from reporting. Actual-byte candidate selection also needs a defined encoding scope and computation budget.

Three concrete checks or improvements

1. Audit annotation isolation and tracking state. With fixed inputs and configuration, remove or permute GT and require identical source-only candidate scores, selections, and encoded outputs. Keep GT available only for evaluation. Test cloned-state probes against a deterministic replay: rejected candidates must leave committed state unchanged, and the selected candidate must advance it exactly once. Apply common correctness fixes across arms so ablations isolate guard and action-family effects.

2. Verify source provenance before freezing TEST execution. Establish K400 frame indices, timestamps, extraction cadence, and model sampling; re-extract from official videos if these cannot be established. Check split exclusions using canonical IDs alongside hashes. Add independent MOT sequences for broader claims; otherwise explicitly limit tracking conclusions to this sequence segment.

3. Predeclare estimation, failures, and success criteria. Name the BD-rate reference, aggregation rules, overlap threshold, slack, and handling of ties, nonmonotonicity, and undefined curves. Interpolate log-rate against quality on a prespecified common interval; publish raw four-point curves and every rejection. Treat undefined targets as “not demonstrated,” not omitted successes. Stabilize policy choices on expanded OD DEV, then freeze. Report candidate-search encodes, end-to-end latency, and peak memory. Any claim that all six targets pass should address joint uncertainty using independent sampling units; one MOT sequence cannot supply population-level tracking confidence.
