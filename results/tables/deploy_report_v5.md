# Deployment report (2026-11-22 05:46 UTC)

- Registry version: 5 (sagemaker-xgboost:3.0-5)
- Approval reason: Same trees as version 4 (test AUC-PR 0.8467); adds reason codes
- Endpoint: fraud-xgb-v5-20261122-053937 on 1 x ml.m5.xlarge (deleted at the end of the run)

| Check | Result | Detail |
| --- | --- | --- |
| served 0-100 score matches the registry calibration | PASS | largest gap 0.00 points |
| every alert comes with reason codes | PASS | 4 of 4 alerts have reasons |
| smoke test: one score per row | PASS | 10 scores for 10 rows |
| smoke test: frauds alerted, no false alarms | PASS | 4 of 5 frauds alerted, 0 false alarms |
| full test: one score per row | PASS | 42,722 scores for 42,722 rows |
| reproduces test_auc_pr | PASS | endpoint 0.8467 vs registry 0.8467 |
| reproduces dr_5to1 | PASS | endpoint 0.8514 vs registry 0.8514 |
| reproduces vdr_5to1 | PASS | endpoint 0.5007 vs registry 0.5007 |
| latency p99 within budget | PASS | p99 15.7 ms vs budget 50 ms |

Single-transaction latency over 190 calls (round trip from Studio): p50 13.1 ms, p95 14.8 ms, p99 15.7 ms, max 28.4 ms.
Alerts at 5:1 on the test set: 389 (63 of 74 frauds caught, 326 false alarms).
