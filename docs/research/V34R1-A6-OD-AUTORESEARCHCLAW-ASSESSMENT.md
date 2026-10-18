# AutoResearchClaw actual OD result assessment

Summary-only actual ACP invocation, no local real task work. Prompt SHA256: d3f3cab1e610e4f4ce0433970e015d91b3ff915fc5f76d4c017c7b4ef8edb24d

The reported checks support computational consistency and a cohort-held-out development evaluation. Native/portable parity and replacement checks strengthen implementation credibility. The 32,256 rows remain correlated observations from 192 images in three cohorts. The selected policy’s pooled OOF BD-rates, −21.1466% for H264 and −14.3034% for H265, are valid but fail the joint requirement: neither this policy nor any of the 14 prespecified choices achieves both strictly below −15%.

Deployment efficacy remains unsupported. Policy selection used these OOF results, so they are development evidence rather than independent confirmation. Prediction MAE does not establish compression benefit. Transfer from 128-image fold models to the final 192-image model is unmeasured, and teacher scale/composition may shift. No prospective pilot or OD256 confirmation exists, and the different-cohort A4 results do not establish an A6 improvement.

The single next discriminating measurement is the locked final model’s actual H264/H265 BD-rate pair on the untouched prospective 64-image pilot, using exactly QP30/35/40/45, full paid 40-byte headers, and valid common-support PCHIP without repair or extrapolation. Point-gate success requires both values strictly below −15% plus all eight mandatory provenance blocks; confidence-interval assessment remains separate.