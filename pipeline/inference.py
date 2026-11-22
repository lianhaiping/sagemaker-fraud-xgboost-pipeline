"""
inference.py - the serving code packaged INSIDE the model: every score comes with its reason codes.

WHERE THIS RUNS
    Not on your machine and not during training. It is packaged in model.tar.gz next to the trees:
        model.tar.gz
        ├── xgboost-model          the trained trees (saved by train.py, re-saved at registration)
        └── code/
            ├── inference.py       this file
            └── scoring.json       calibration (Platt a, b), score scale, feature names, cutoff,
                                   and whether to explain every row or only alerts
    The registry entry tells AWS's XGBoost serving container to use it, through two settings:
        SAGEMAKER_PROGRAM = inference.py
        SAGEMAKER_SUBMIT_DIRECTORY = /opt/ml/model/code
    When the endpoint starts, the container calls model_fn once; for every request it calls
    input_fn -> predict_fn -> output_fn, in that order.

THE REQUEST AND THE ANSWER
    Request:  Content-Type text/csv, one transaction per line, 29 numbers (V1..V28, Amount), no label
    Answer:   application/json
        {"predictions": [{"raw_score": 0.9998, "fraud_probability": 0.9973, "score_0_100": 100.0,
                          "alert_5to1": true,
                          "reasons": [{"feature": "V14", "contribution": 3.21}, ...]}, ...]}

HOW THE REASON CODES ARE COMPUTED (TreeSHAP)
    XGBoost's predict(..., pred_contribs=True) splits each transaction's raw log-odds into one exact
    contribution per feature plus a bias term (they add up to the model's output). This is TreeSHAP,
    computed from the trees themselves: no second model, no extra library. The reasons are the
    features with the largest POSITIVE contributions, i.e. what pushed this transaction towards fraud.
    Calibration does not change their order: Platt scaling is a monotone transform of the log-odds.
    Note: on this dataset the features are anonymized PCA components, so a reason reads "V14", not a
    plain-language statement; a production system maps each feature to an approved reason text.
"""
import json
import os
import pickle

import numpy as np
import xgboost as xgb

TOP_K = 3                  # reasons returned per transaction


def model_fn(model_dir):
    """Load the trees and the scoring settings once, when the endpoint starts.

    model_dir is /opt/ml/model, where SageMaker unpacked model.tar.gz.
    """
    booster = xgb.Booster()
    path = os.path.join(model_dir, "xgboost-model")
    try:
        booster.load_model(path)              # the format the pipeline saves (XGBoost 2.1 or newer)
    except Exception:
        with open(path, "rb") as f:           # very old models were pickled
            booster = pickle.load(f)
    with open(os.path.join(model_dir, "code", "scoring.json")) as f:
        scoring = json.load(f)
    return {"booster": booster, "scoring": scoring}


def input_fn(request_body, content_type):
    """CSV text -> a NumPy array of floats, one row per transaction."""
    if content_type != "text/csv":
        raise ValueError(f"Unsupported content type {content_type}: send text/csv")
    text = request_body.decode("utf-8") if isinstance(request_body, (bytes, bytearray)) else request_body
    rows = [[float(v) for v in line.split(",")] for line in text.strip().splitlines() if line.strip()]
    return np.array(rows, dtype=float)


def predict_fn(X, model):
    """Score each transaction and explain it.

    1. raw_score: the model's output (a probability on the weighted training scale, not a true rate)
    2. fraud_probability: Platt calibration, sigmoid(a + b * logit(raw_score)), fitted on validation
    3. score_0_100: 50 + 3 * log2(odds / 0.001), capped: 50 = odds 1:1000, +3 points doubles the odds
    4. alert_5to1: raw_score at or above the cutoff chosen on validation for 5 false positives per fraud
    5. reasons: the TOP_K features with the largest positive TreeSHAP contribution (log-odds units)
    """
    b, sc = model["booster"], model["scoring"]
    dm = xgb.DMatrix(X)
    raw = b.predict(dm)
    # TreeSHAP for every row, or only for rows that alert (scoring.json "explain_alerts_only").
    # It costs about 2 ms per row, about 50 times the scoring itself: negligible for one card
    # authorization, but it dominates a batch of 1,000 rows if every row is explained.
    contribs = np.zeros((len(raw), X.shape[1]))
    rows = np.where(raw >= sc["thr_5to1"])[0] if sc.get("explain_alerts_only") else np.arange(len(raw))
    if len(rows):
        contribs[rows] = b.predict(xgb.DMatrix(X[rows]), pred_contribs=True)[:, :-1]   # last column = bias
    r = np.clip(raw.astype(float), 1e-9, 1 - 1e-9)
    log_odds = sc["platt_a"] + sc["platt_b"] * np.log(r / (1 - r))
    prob = 1 / (1 + np.exp(-log_odds))
    score = sc["anchor_score"] + sc["pdo"] * (log_odds - np.log(sc["anchor_odds"])) / np.log(2)
    score = np.clip(score, 0, 100)
    names = sc["features"]
    out = []
    for i in range(len(raw)):
        top = np.argsort(-contribs[i])[:TOP_K]                 # largest contributions first
        out.append({
            "raw_score": float(raw[i]),
            "fraud_probability": float(prob[i]),
            "score_0_100": round(float(score[i]), 1),
            "alert_5to1": bool(raw[i] >= sc["thr_5to1"]),
            "reasons": [{"feature": names[j], "contribution": round(float(contribs[i, j]), 4)}
                        for j in top if contribs[i, j] > 0],     # only features that push towards fraud
        })
    return out


def output_fn(prediction, accept):
    """Return the answer as JSON (the content type travels with it)."""
    return json.dumps({"predictions": prediction}), "application/json"
