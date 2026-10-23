**Baseline policy**

No accepted numerical baseline is shipped. Initial CI uses `observe`: complete, valid evaluation results are reported even when accuracy is low or an answer reaches the token budget. Request errors, missing examples, malformed results, and missing artifacts fail in every mode.

After calibration, a baseline JSON document may contain a `cases` mapping keyed by the six case IDs. Each entry requires:

- `comparison_key`: copy from the accepted case's `result.json`; this identifies the recipe, datasets, runtime/image, and evaluator.
- `metrics`: accepted `score`, `mean_ttft_ms`, `mean_tpot_ms`, and `output_throughput` values, using the declared repeat aggregation.
- `accuracy_floor` and `accuracy_drop`: reviewed fractions in [0, 1]. One answer among 200 changes accuracy by 0.005.
- `performance_tolerance`: a reviewed relative fraction in [0, 1].

Record the accepted source tuple, calibration run/artifact, reviewer, and rationale alongside those entries. Use `run.py plan --gate-mode enforce` and `run.py execute --baseline PATH` only after reviewing this data. A missing/incompatible baseline fails enforcement; adding a baseline file does not automatically enable numerical gates. Profiler timings are diagnostic and excluded from numerical gates.
