# Dataset feedback: near-boundary reference labels and incompatible MIB constraints

Subject: FloorSet / ICCAD 2026 Problem C - reproducible dataset issues and request for clarification

Dear Problem C organizers,

I am studying the FloorSet dataset and would like to report two issues that substantially affect the usefulness of the reference layouts as supervision and the interpretation of solution quality. I have attached concrete layout visualizations and reproducible case identifiers.

## 1. Reference blocks often stop just short of their required boundary

In a fixed local development panel of 100 layouts drawn from the released training data, 91 reference layouts violate at least one boundary requirement, with 210 violating blocks in total. For 97 of these blocks, across 50 layouts, all required-edge gaps are at most 1% of the corresponding bounding-box width or height. This is a descriptive threshold; the evaluator's touching tolerance remains 1e-6.

For example, in worker_8/layouts_1232.th, sample 102 (zero-based), block 74 has a top-right boundary requirement (code 6). Its reference XYWH is (85, 319, 23, 24), while the bounding-box top is 344. Moving only this block upward by 1 unit, to (85, 320, 23, 24), satisfies the corner requirement without overlap or any change to dimensions or other blocks. Two further one-unit examples are documented in the attachment.

These appear to be easily correctable reference-label defects: the required boundary is only a small distance away, and a direct legal movement is available. Such labels can teach a supervised model to preserve avoidable boundary errors. Could you explain why these near-boundary gaps remain in the released references, and whether a corrected or optionally repaired reference-label release would be possible?

<!-- pagebreak -->

## 2. MIB groups can conflict mathematically with hard area requirements

In worker_0/layouts_1344.th, sample 82, MIB group 1 contains blocks 33, 42 and 63 with target areas 72, 240 and 56. Their permitted area intervals are [71.28, 72.72], [237.60, 242.40] and [55.44, 56.56]. Identical width and height imply identical area, so these hard area requirements conflict with the intended MIB same-shape requirement.

We also accounted for the evaluator's rounding of widths and heights to four decimal places, rather than relying only on exact-area reasoning. For this example the conservatively relaxed intervals remain disjoint. On the same 100-layout panel, our existing solutions incur 393 MIB violations; a conservative certificate establishes at least 374 unavoidable violations under the published evaluator. The resulting arithmetic-mean neutral-runtime quality lower bound is 1.180663, even when all other losses are optimistically ignored. This is a local-panel bound, not a claim about the hidden set.

We understand from Q&A A4 that reference solutions may violate soft constraints, and from A5/A6 that hard constraints take priority. However, a soft target that cannot be satisfied even in principle is different from a suboptimal reference. Here, at least 95.2% of the observed MIB violations are already covered by the unavoidable lower bound. Could you confirm whether these incompatible MIB groups are intentional, explain their origin, and consider providing compatibility flags or a constraint-consistent dataset variant?

## Clarification and possible collaboration

Q&A A23/A24 state that the released and hidden datasets use the same generation method. Does the hidden set contain the same types of near-boundary labels and MIB/area conflicts, or were any additional repairs or consistency checks applied?

For research validation, would it be possible to obtain access to the hidden benchmark if its release is permitted? If it cannot be shared, could you instead run our frozen solver once on your side and provide quality-only and runtime results, ideally with boundary, grouping and MIB violation counts?

I would also be glad to contribute a reproducible audit and reference-repair checks to help improve the dataset. The attached evidence uses saved development records only; it does not require access to hidden test data.

Thank you for your time and for making FloorSet available.

Best regards,
Chengxiao Yu

Attachment: dataset_issues_evidence_20261018.pdf
