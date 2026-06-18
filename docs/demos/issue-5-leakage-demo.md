# Issue #5 — Train/test data leakage fix: outcome evidence

*2026-06-17T21:17:09Z*

```bash
uv run python /tmp/issue5demo/ac1_cv_leakfree.py 2>/dev/null
```

```output
Preprocessor fitted BEFORE cross_val_score?  False
cross_val_score ran 5 folds (each refits preprocessing in isolation)
Preprocessor still unfitted AFTER (clones run per fold)?  True
=> Imputation/encoding never sees test-fold rows. AC1 satisfied.
```

AC3 — A dataset below the held-out floor (n<10) is evaluated in-sample and labelled honestly: evaluation=in_sample, n_too_small=True, no test_indices, and the summary says "on the training data", NOT "held-out test set".

```bash
uv run python /tmp/issue5demo/ac3_insample_honesty.py 2>/dev/null
```

```output
n = 6 rows (< 10)
metrics.evaluation : in_sample
metrics.n_too_small: True
test_indices       : None
summary            : R² = 1.00 (excellent fit — 1.0 would be perfect) on the training data (dataset too small for a held-out test — treat as optimistic). On average, predictions are off by 0.00 units (MAE).
=> No false 'held-out' claim on memorised data. AC3 satisfied.
```

AC2 — End-to-end through the real API: upload -> features -> train (leak-free) -> query diagnostics. Training persists held-out test_indices + a train-fold preprocessor sidecar; every diagnostic re-scores on that held-out slice and returns evaluation="held_out".

```bash
uv run python /tmp/issue5demo/ac2_heldout_endpoint.py 2>/dev/null | grep -vE "^\s*$"
```

```output
Training summary: 82.5% accuracy on the held-out test set. F1 = 0.83 (balances precision and recall; 1.0 is perfect).
Stored metrics.evaluation: held_out
GET /api/validate/c8e5da7a-b069-46d7-8e9c-c49239cac7ea/metrics -> 200  evaluation=held_out
GET /api/models/c8e5da7a-b069-46d7-8e9c-c49239cac7ea/calibration-check -> 200  evaluation=held_out
GET /api/models/c8e5da7a-b069-46d7-8e9c-c49239cac7ea/segment-performance?col=region -> 200  evaluation=held_out
=> Diagnostics report evaluation='held_out' — scored on rows the model never trained on. AC2 satisfied.
```

AC4 — The leakage regression suite pins all of the above: the CV preprocessor refits per fold, the reported held-out metric equals a manual leak-free baseline, the old global-fit path provably diverges, n<10 is tagged in_sample, and serving encodes unseen categories with the training sentinel.

```bash
uv run pytest tests/test_train_test_leakage.py tests/test_preprocessing.py -q -p no:cacheprovider 2>/dev/null | tail -4
```

```output
.................                                                        [100%]
17 passed in 2.21s
```

All four acceptance criteria demonstrated with live outcome evidence: AC1 leak-free CV (preprocessor unfitted/refit per fold), AC2 held-out diagnostics end-to-end (evaluation=held_out), AC3 honest in-sample labelling for tiny data, AC4 regression suite green (17 passed). Legacy models without a sidecar fall back to in_sample, tagged with a caveat — never silently inflated.
