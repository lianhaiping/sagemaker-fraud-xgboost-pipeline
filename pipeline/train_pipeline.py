"""
train_pipeline.py - the training pipeline as a SWEEP (the "orchestrator").

THE TWO QUESTIONS IT ANSWERS
    Story 1, volume: how many legitimate ("good") transactions does an XGBoost fraud model need?
        Six training sets that keep ALL frauds and a growing number of goods: 17, 50, 100, 200 and
        500 goods per fraud, and the full data (about 579). The fraud weight (scale_pos_weight) is
        scaled with the ratio so every model sees the SAME effective balance, about 17 goods per
        weighted fraud. Only the volume of goods changes.
    Story 2, balance: how much should the frauds be weighted?
        Full data in every case, fraud weight 1, 6, 35, 116 and 579, i.e. an effective balance of
        579:1 (natural), 100:1, 17:1, 5:1 and 1:1 (fully balanced). Only the weight changes.
    "full-w35" (full data, weight 35) belongs to both stories: it is the shared reference point.

WHAT EVERY CONFIGURATION GETS (identical treatment, so differences are real)
    - the same train.py, the same 7-parameter search space, the same 20 random draws (tuner seed 42),
      4 trials at a time
    - its best settings retrained with XGBoost seeds 1, 2 and 3 (training-noise check)
    - every seed model scored on validation AND test: AUC-PR, AUC-PR without duplicated rows,
      detection rates at 1:1 / 5:1 / 10:1 false positives per fraud (cutoffs chosen on validation),
      false positives per fraud at 80% detection, and calibration BEFORE and AFTER Platt scaling
    - a bootstrap interval of its test AUC-PR difference against the reference (full-w35)

REUSING EARLIER RUNS
    "r50" (50:1, weight 3) and "full-w35" are exactly challengers A and B of the earlier A-vs-B run
    (registry versions 3 and 4). If their tuning and seed jobs still exist in your account, they are
    reused instead of retrained (same numbers, about 15 minutes saved). In an account without them,
    or with --no-reuse, they are simply trained like the others, so the study is reproducible by anyone.

STEPS (the section headers it prints)
    S0  setup, data download and split, duplicate check, upload of each configuration's training file
    S1  random searches, run 3 at a time (12 machines; the account quota is 15)
    S2  each configuration's best settings retrained with 3 seeds, in waves of up to 15 jobs
    S3  every seed model scored on validation and test; summary and bootstrap intervals
    S4  the two charts: AUC-PR vs goods kept (Story 1) and AUC-PR and calibration vs weight (Story 2)
    S5  the best configuration (chosen on VALIDATION) gated against the current champion on test and
        saved as a registration candidate (written to the registry only with --register). The gate needs
        a material lift (CFG["min_lift"]) AND a 95% bootstrap interval above zero

WHERE IT RUNS
    In a SageMaker Studio terminal (File -> New -> Terminal), from the REPOSITORY ROOT (the folder
    above pipeline/), so outputs/, data/ and models/ are created there. The heavy work runs on
    SageMaker training machines, inside AWS's XGBoost container, running pipeline/train.py.

HOW TO RUN
    python pipeline/train_pipeline.py                     # the sweep: train, score, chart (registry untouched)
    python pipeline/train_pipeline.py --no-reuse          # retrain r50 and full-w35 too
    python pipeline/train_pipeline.py --register          # also register the best configuration
    python pipeline/train_pipeline.py --register-only outputs/<run>/register/<config>   # register a saved candidate
    python pipeline/train_pipeline.py --repackage-version 4   # no training: register a copy of version 4's
                                                              # trees WITH the reason-code serving code
    Keep it running if the browser disconnects:
        nohup python pipeline/train_pipeline.py > sweep.log 2>&1 &     then:  tail -f sweep.log

COST AND TIME
    8 new configurations x (20 trials + 3 seeds) = 184 short training jobs on ml.m5.xlarge:
    about 45-60 minutes and a few dollars. No endpoint is created.

REASON CODES ARE PART OF THE MODEL PACKAGE
    Every package this pipeline registers holds the trees AND the serving code:
        model.tar.gz = xgboost-model + code/inference.py + code/scoring.json
    and its registry entry sets SAGEMAKER_PROGRAM=inference.py, so the endpoint returns each score with
    its top 3 TreeSHAP reason codes (see pipeline/inference.py). The trees and their scores are unchanged.

OUTPUTS (outputs/<timestamp>/)
    sweep_seeds.csv     one row per seed model: every metric on validation and test
    sweep_summary.csv   one row per configuration: 3-seed means, seed ranges, bootstrap interval
    sweep_summary.md    the two stories as tables
    volume_curve.png    Story 1 chart;   balance_curve.png   Story 2 chart
    tuning_top10_<config>.csv per configuration, and register/<config>/ for the best one
"""
# ============================================================================ imports
import argparse            # command-line options
import json                # save and load the registration candidate
import os                  # folders and file paths
import tarfile             # unpack and build model.tar.gz archives
import time                # timestamps and waiting between status checks

import matplotlib
matplotlib.use("Agg")      # "Agg" draws charts into files: a terminal has no screen to show them on
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter    # plain tick labels on log axes (1, 10, 100)
import numpy as np
import pandas as pd
import boto3               # low-level AWS API client (sm = boto3.client("sagemaker"))
import sagemaker           # SageMaker Python SDK v2 (higher-level helpers)
import xgboost as xgb      # local XGBoost (2.1.4): loads the trained models for evaluation
from sagemaker import get_execution_role
from sagemaker.analytics import HyperparameterTuningJobAnalytics   # tuning results as a DataFrame
from sagemaker.inputs import TrainingInput                         # describes a data channel in S3
from sagemaker.s3 import S3Downloader                              # copy files from S3
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter
from sagemaker.xgboost import XGBoost      # script-mode estimator: runs OUR train.py in AWS's XGBoost image
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression                # Platt calibration
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split

# ============================================================================ settings (edit here)
# Every configuration of the sweep. ratio = goods kept per fraud (None = all training rows);
# weight = scale_pos_weight. Names may only use letters, digits and "-" (they become job names).
CONFIGS = [
    # ---- Story 1, volume: the weight follows the ratio, so ratio / weight stays about 17
    {"name": "r17",       "story": "volume",  "ratio": 17,   "weight": 1},
    {"name": "r50",       "story": "volume",  "ratio": 50,   "weight": 3},     # = challenger A (v3)
    {"name": "r100",      "story": "volume",  "ratio": 100,  "weight": 6},
    {"name": "r200",      "story": "volume",  "ratio": 200,  "weight": 12},
    {"name": "r500",      "story": "volume",  "ratio": 500,  "weight": 29},
    {"name": "full-w35",  "story": "both",    "ratio": None, "weight": 35},    # = challenger B (v4), the reference
    # ---- Story 2, balance: full data, only the weight changes
    {"name": "full-w1",   "story": "balance", "ratio": None, "weight": 1},     # natural 579:1
    {"name": "full-w6",   "story": "balance", "ratio": None, "weight": 6},     # about 100:1
    {"name": "full-w116", "story": "balance", "ratio": None, "weight": 116},   # about 5:1
    {"name": "full-w579", "story": "balance", "ratio": None, "weight": 579},   # about 1:1 (fully balanced)
]
REFERENCE = "full-w35"     # every configuration's test AUC-PR is compared with this one
# The 0-100 score scale used by the serving code: 50 = fraud odds 1:1000, +3 points doubles the odds
CFG_SCORE = {"pdo": 3, "anchor_score": 50, "anchor_odds": 1 / 1000}

CFG = {
    "framework_version": "1.7-1",            # AWS XGBoost container version that runs train.py
    "serve_version": "3.0-5",                # serving image recorded in the registry (reads 2.1 models)
    "instance_type": "ml.m5.xlarge",         # training machine type
    "machine_quota": 15,                     # ml.m5.xlarge training machines allowed at once (account quota)
    "tuners_at_once": 3,                     # 3 searches x 4 trials = 12 machines, under the quota
    "max_jobs": 20,                          # random-search budget per configuration (trials)
    "max_parallel": 4,                       # trials running at once per configuration
    "tuner_seed": 42,                        # the same 20 random draws for every configuration
    "seeds": [1, 2, 3],                      # XGBoost seeds for the retrain (training-noise check)
    "max_run": 1800,                         # cost guardrail: stop any job after 30 minutes
    "fixed": {"eta": 0.05, "num_round": 2000, "early_stopping_rounds": 100},   # same for every trial
    "sample_seed": 42,                       # fixes which goods each sampled configuration keeps
                                             # (the same call as the A-vs-B run, so r50 = challenger A's data)
    "bootstrap": 1000,                       # test resamples for the intervals against the reference
    "detection_for_fp": 0.80,                # "false positives per fraud at 80% detection" (test)
    "ratios": [1, 5, 10],                    # false-positive ratios for the operations metrics
    "group": "fraud-xgb-models",             # model package group in the registry
    "prefix": "fraud-xgb/sweep",             # S3 folder for this pipeline's files
    "min_auc_pr_first": 0.70,                # gate when the registry has no champion yet
    # Release gate against a champion: the candidate must beat it by at least this much (3-seed mean,
    # test AUC-PR) AND the 95% bootstrap interval of the difference must lie entirely above zero.
    # (The first sweep showed why: "mean beats champion" alone approved a 0.0004 gain, i.e. noise.)
    "min_lift": 0.005,
    # Reason codes in the serving code: False = for every transaction (what v5 runs);
    # True = only for transactions that alert, which keeps batch scoring fast (TreeSHAP costs about
    # 2 ms per row, about 50 times the scoring itself; single transactions add only about 1 ms).
    "explain_alerts_only": False,
    # Jobs from the earlier A-vs-B run, reused when they exist (see REUSING EARLIER RUNS above)
    "reuse": {
        "r50": {"tuning_job": "fraud-d5A-261116-0540",
                "seed_jobs": ["fraud-d5A-seed1-2026-11-16-05-47-39-959",
                              "fraud-d5A-seed2-2026-11-16-05-47-40-463",
                              "fraud-d5A-seed3-2026-11-16-05-47-41-174"]},
        "full-w35": {"tuning_job": "fraud-d5B-261116-0540",
                     "seed_jobs": ["fraud-d5B-seed1-2026-11-16-05-47-41-971",
                                   "fraud-d5B-seed2-2026-11-16-05-47-43-209",
                                   "fraud-d5B-seed3-2026-11-16-05-47-44-638"]},
    },
}

# The searched settings and their ranges (unchanged from the A-vs-B run)
RANGES = {
    "max_depth": IntegerParameter(3, 8),
    "min_child_weight": ContinuousParameter(1, 10, scaling_type="Logarithmic"),
    "subsample": ContinuousParameter(0.6, 1.0),
    "colsample_bytree": ContinuousParameter(0.5, 1.0),
    "gamma": ContinuousParameter(0, 5),
    "lambda": ContinuousParameter(0.1, 10, scaling_type="Logarithmic"),
    "alpha": ContinuousParameter(0.001, 10, scaling_type="Logarithmic"),
}

# The tuner's objective. The script-mode container is AWS's built-in XGBoost image, so the tuning
# service only accepts built-in metric names; "validation:aucpr" is read from the LAST
# "[round]<TAB>validation-aucpr:<number>" log line, which train.py prints with the BEST score.
OBJECTIVE = "validation:aucpr"
HERE = os.path.dirname(os.path.abspath(__file__))     # the pipeline/ folder (train.py lives here)


# ============================================================================ printing helpers
def log(msg=""):
    """Print a line with the current time; flush=True writes it to the log immediately."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def section(title):
    """Print a visible section header in the log."""
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90, flush=True)


def savefig(ctx, name):
    """Save the current matplotlib figure as a PNG in the output folder and close it."""
    path = os.path.join(ctx["out"], name)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    log(f"chart saved: {path}")


def save_table(ctx, df, name, title):
    """Save a DataFrame as CSV and print it in full in the log."""
    path = os.path.join(ctx["out"], name)
    df.to_csv(path, index=False)
    print(f"\n{title}\n{df.to_string(index=False)}\n", flush=True)
    log(f"table saved: {path}")


# ============================================================================ S0: setup and data
def setup():
    """Connect to SageMaker and create a timestamped output folder.

    Returns "ctx" (context), a small dictionary passed to every function instead of global variables.
    No access keys: inside Studio, boto3 and the SDK use the space's IAM role automatically.
    """
    sess = sagemaker.Session()
    ctx = {
        "sess": sess,
        "role": get_execution_role(),
        "region": sess.boto_region_name,
        "bucket": sess.default_bucket(),                # sagemaker-<region>-<account>
        "sm": boto3.client("sagemaker"),
        "out": os.path.join("outputs", time.strftime("%Y%m%d-%H%M%S")),
    }
    os.makedirs(ctx["out"], exist_ok=True)
    log(f"SDK {sagemaker.__version__} | XGBoost {xgb.__version__} | region {ctx['region']} | outputs in {ctx['out']}")
    return ctx


def load_data():
    """Download the dataset and split it 70 / 15 / 15 (train / validation / test), stratified, seed 42.

    Also flags test rows identical to a training row (the dataset has exact duplicates), for the
    deduplicated AUC-PR.
    """
    log("Downloading OpenML dataset 1597 (1 to 3 minutes)...")
    df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
    df["Class"] = df["Class"].astype(str).astype(int)        # labels arrive as text categories
    feats = [c for c in df.columns if c != "Class"]          # V1..V28 and Amount (29 features)
    df = df[["Class"] + feats]                               # label first: the CSV format train.py expects
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df["Class"], random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["Class"], random_state=42)
    for name, part in [("train", tr), ("validation", va), ("test", te)]:
        log(f"{name:<11} rows={len(part):>7,}  frauds={part['Class'].sum():>4}")
    h_tr = pd.util.hash_pandas_object(tr[feats], index=False)
    in_train = pd.util.hash_pandas_object(te[feats], index=False).isin(h_tr).values
    log(f"test rows identical to a training row: {in_train.sum():,} ({in_train.mean():.2%})")
    return tr, va, te, feats, in_train


def upload_channels(ctx, tr, va, configs):
    """Write and upload each configuration's training file, plus the shared validation file.

    All full-data configurations share ONE upload (only their weight differs). Each sampled
    configuration keeps every fraud and `ratio` goods per fraud. The sample uses the same pandas call
    and seed as the A-vs-B run, so r50 reproduces challenger A's training data exactly. Each ratio is
    its own random draw (the samples are not nested).
    Records each configuration's number of goods in c["goods"].
    Returns {"validation": s3 path, "<config name>": s3 path, ...}.
    """
    p, sess, b = CFG["prefix"], ctx["sess"], ctx["bucket"]
    os.makedirs("data", exist_ok=True)
    va.to_csv("data/validation.csv", header=False, index=False)
    ch = {"validation": sess.upload_data("data/validation.csv", bucket=b, key_prefix=f"{p}/data/validation")}
    full_path = None
    for c in configs:
        if c["ratio"] is None:
            if full_path is None:                   # upload the full training file once
                tr.to_csv("data/train_full.csv", header=False, index=False)
                full_path = sess.upload_data("data/train_full.csv", bucket=b, key_prefix=f"{p}/data/train_full")
            ch[c["name"]] = full_path
            part = tr
        else:
            fraud = tr[tr["Class"] == 1]
            goods = tr[tr["Class"] == 0].sample(n=len(fraud) * c["ratio"], random_state=CFG["sample_seed"])
            part = pd.concat([fraud, goods]).sample(frac=1, random_state=CFG["sample_seed"])   # shuffle rows
            path = f"data/train_{c['name']}.csv"
            part.to_csv(path, header=False, index=False)
            ch[c["name"]] = sess.upload_data(path, bucket=b, key_prefix=f"{p}/data/train_{c['name']}")
        goods_n, fraud_n = int((part["Class"] == 0).sum()), int(part["Class"].sum())
        c["goods"] = goods_n
        log(f"{c['name']:<10} {len(part):>8,} rows, {goods_n:>8,} goods ({goods_n / fraud_n:.0f} per fraud), "
            f"weight {c['weight']:>3} -> effective {goods_n / fraud_n / c['weight']:.1f}:1")
    return ch


# ============================================================================ SageMaker jobs
def make_estimator(ctx, name, base, hp):
    """A script-mode estimator: "run pipeline/train.py in AWS's XGBoost 1.7-1 image on one ml.m5.xlarge"."""
    return XGBoost(
        entry_point=os.path.join(HERE, "train.py"),   # absolute path: works from any working directory
        framework_version=CFG["framework_version"], py_version="py3",
        role=ctx["role"], instance_type=CFG["instance_type"], instance_count=1,
        output_path=f"s3://{ctx['bucket']}/{CFG['prefix']}/output/{name}",
        base_job_name=base, max_run=CFG["max_run"],
        hyperparameters=hp,                          # become --name value arguments to train.py
        sagemaker_session=ctx["sess"])


def train_inputs(ch, name):
    """The two data channels: this configuration's training file and the shared validation file."""
    return {"train": TrainingInput(ch[name], content_type="text/csv"),
            "validation": TrainingInput(ch["validation"], content_type="text/csv")}


def job_exists(ctx, kind, name):
    """True if a training job (kind="training") or tuning job (kind="tuning") exists in this account."""
    try:
        if kind == "training":
            ctx["sm"].describe_training_job(TrainingJobName=name)
        else:
            ctx["sm"].describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        return True
    except ctx["sm"].exceptions.ClientError:
        return False


def reusable(ctx, name, allow):
    """The earlier jobs to reuse for this configuration, or None (then it is trained like the others)."""
    r = CFG["reuse"].get(name)
    if not allow or r is None:
        return None
    if job_exists(ctx, "tuning", r["tuning_job"]) and all(job_exists(ctx, "training", j) for j in r["seed_jobs"]):
        return r
    log(f"{name}: earlier jobs not found in this account, so it will be trained")
    return None


def run_tuning(ctx, ch, configs):
    """Run one random search per configuration that still needs one, at most CFG["tuners_at_once"] at a time.

    A new search starts as soon as a running one finishes (a rolling wave), so the machines stay busy
    without exceeding the quota. Sets c["tuning_job"] for each configuration it tunes.
    """
    queue = [c for c in configs if not c.get("tuning_job")]
    running = {}                                     # config name -> (config, tuning job name)
    while queue or running:
        while queue and len(running) < CFG["tuners_at_once"]:
            c = queue.pop(0)
            hp = {**CFG["fixed"], "scale_pos_weight": c["weight"], "seed": 1}
            est = make_estimator(ctx, c["name"], f"fraud-{c['name']}-trial", hp)
            t = HyperparameterTuner(
                est, objective_metric_name=OBJECTIVE, objective_type="Maximize",
                hyperparameter_ranges=RANGES, strategy="Random", random_seed=CFG["tuner_seed"],
                max_jobs=CFG["max_jobs"], max_parallel_jobs=CFG["max_parallel"],
                base_tuning_job_name=f"fraud-{c['name']}")   # tuning job names: at most 32 characters
            t.fit(train_inputs(ch, c["name"]), wait=False)
            running[c["name"]] = (c, t.latest_tuning_job.name)
            log(f"{c['name']}: tuning job {t.latest_tuning_job.name} started")
        time.sleep(90)
        for name, (c, job) in list(running.items()):
            dsc = ctx["sm"].describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job)
            st, cnt = dsc["HyperParameterTuningJobStatus"], dsc["TrainingJobStatusCounters"]
            log(f"{name:<10} {st:<11} completed {cnt['Completed']:>2}, running {cnt['InProgress']}, "
                f"failed {cnt['NonRetryableError'] + cnt['RetryableError']}")
            if st in ("Completed", "Failed", "Stopped"):
                c["tuning_job"] = job
                del running[name]
                if st != "Completed":
                    log(f"WARNING: {name} tuning ended as {st}; its best completed trial is used if there is one")


def best_hyperparameters(ctx, tuning_job):
    """The best trial's job name and its settings as a clean dictionary of numbers.

    describe_training_job returns every hyperparameter as TEXT, plus SageMaker's own bookkeeping keys
    (starting with "_" or "sagemaker"); those are dropped, JSON quotes removed, values converted.
    """
    d = ctx["sm"].describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job)
    job = d["BestTrainingJob"]["TrainingJobName"]
    raw = ctx["sm"].describe_training_job(TrainingJobName=job)["HyperParameters"]
    hp = {}
    for k_, v in raw.items():
        if k_.startswith("_") or k_.startswith("sagemaker"):
            continue
        v = v.strip('"')
        try:
            hp[k_] = int(v)
        except ValueError:
            hp[k_] = float(v)
    return job, hp


def wait_training_jobs(ctx, jobs):
    """Wait until every training job in the list has finished; print progress every 60 seconds."""
    pending = set(jobs)
    while pending:
        for j in list(pending):
            st = ctx["sm"].describe_training_job(TrainingJobName=j)["TrainingJobStatus"]
            if st in ("Completed", "Failed", "Stopped"):
                pending.discard(j)
                if st != "Completed":
                    log(f"WARNING: training job {j} ended as {st}")
        if pending:
            log(f"seed jobs: {len(jobs) - len(pending)} of {len(jobs)} finished")
            time.sleep(60)


def run_seeds(ctx, ch, configs):
    """Retrain each configuration's best settings with every seed, at most CFG["machine_quota"] jobs at a time."""
    todo = [(c, s) for c in configs if not c.get("seed_jobs") for s in CFG["seeds"]]
    for c, _ in todo:
        c["seed_jobs"] = []
    for i in range(0, len(todo), CFG["machine_quota"]):
        wave = todo[i:i + CFG["machine_quota"]]
        names = []
        for c, s in wave:
            hp = {**c["best_hp"], "seed": s}            # same settings, only the seed changes
            e = make_estimator(ctx, c["name"], f"fraud-{c['name']}-s{s}", hp)
            e.fit(train_inputs(ch, c["name"]), wait=False)
            job = e.latest_training_job.name
            c["seed_jobs"].append(job)
            names.append(job)
            log(f"{c['name']} seed {s}: {job} started")
        wait_training_jobs(ctx, names)


# ============================================================================ models and metrics
def load_booster(url, folder):
    """Download a model.tar.gz from S3, unpack it, and load the model with the local XGBoost.

    train.py saves "xgboost-model.json" (best trees only); registered packages hold "xgboost-model".
    """
    S3Downloader.download(url, folder)
    with tarfile.open(os.path.join(folder, "model.tar.gz")) as tar:
        tar.extractall(folder, filter="data")          # "data" filter = safe extraction
    for fname in ("xgboost-model.json", "xgboost-model"):
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            b = xgb.Booster()
            b.load_model(path)
            best = b.attr("best_iteration")
            return b[: int(best) + 1] if best is not None else b
    raise FileNotFoundError(f"no model file in {folder}")


def model_of_job(ctx, job):
    """Load the model a training job produced."""
    url = ctx["sm"].describe_training_job(TrainingJobName=job)["ModelArtifacts"]["S3ModelArtifacts"]
    return load_booster(url, os.path.join("models", job))


def ratio_threshold(y, s, r):
    """Lowest score cutoff where false positives per fraud caught stay at or below r."""
    order = np.argsort(-s)
    tp, fp = np.cumsum(y[order]), np.cumsum(1 - y[order])
    return s[order][np.where(fp <= r * tp)[0].max()]


def logit(s):
    """Probability -> log-odds; clipped to avoid infinity at exactly 0 or 1."""
    s = np.clip(np.asarray(s, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(s / (1 - s))


def score_seed(b, d):
    """Every metric for one seed model. d holds the validation/test arrays.

    Ranking: AUC-PR (validation, test, test without duplicated rows), AUC-ROC, and false positives per
    fraud at 80% detection on test. Operations: detection rate (DR), value detection rate (VDR) and the
    achieved false-positive ratio on test, with each cutoff chosen on VALIDATION.
    Calibration on test: mean predicted fraud probability vs the actual rate, log loss and Brier score,
    BEFORE Platt scaling (raw scores, pushed up by the fraud weight) and AFTER it (Platt fitted on
    validation). Returns (row, validation scores, test scores).
    """
    s_val, s_te = b.predict(xgb.DMatrix(d["X_val"])), b.predict(xgb.DMatrix(d["X_te"]))
    y_val, y_te, amt, keep = d["y_val"], d["y_te"], d["amt"], d["keep"]
    row = {"trees": b.num_boosted_rounds(),
           "val_auc_pr": average_precision_score(y_val, s_val),
           "test_auc_pr": average_precision_score(y_te, s_te),
           "test_auc_pr_dedup": average_precision_score(y_te[keep], s_te[keep]),
           "test_auc_roc": roc_auc_score(y_te, s_te)}
    prec, rec, _ = precision_recall_curve(y_te, s_te)
    p_at = prec[rec >= CFG["detection_for_fp"]].max()          # best precision that still reaches 80% detection
    row["test_fp_per_fraud_at_80pct"] = (1 - p_at) / p_at
    for r in CFG["ratios"]:
        thr = ratio_threshold(y_val, s_val, r)
        flag = s_te >= thr
        tp, fp = (flag & (y_te == 1)).sum(), (flag & (y_te == 0)).sum()
        row[f"DR@{r}:1"] = tp / y_te.sum()
        row[f"VDR@{r}:1"] = amt[flag & (y_te == 1)].sum() / amt[y_te == 1].sum()
        row[f"FP_ratio@{r}:1"] = fp / max(tp, 1)
    # Calibration before Platt: the raw score read as a probability
    row["raw_mean_pred_%"] = 100 * s_te.mean()
    row["raw_log_loss"] = log_loss(y_te, np.clip(s_te, 1e-7, 1 - 1e-7))
    row["raw_brier"] = brier_score_loss(y_te, s_te)
    # Calibration after Platt: a + b * logit(score), fitted on validation, applied to test
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(s_val).reshape(-1, 1), y_val)
    p_te = lr.predict_proba(logit(s_te).reshape(-1, 1))[:, 1]
    row["platt_mean_pred_%"] = 100 * p_te.mean()
    row["platt_log_loss"] = log_loss(y_te, p_te)
    row["platt_brier"] = brier_score_loss(y_te, p_te)
    row["actual_rate_%"] = 100 * y_te.mean()
    row["platt_a"], row["platt_b"] = float(lr.intercept_[0]), float(lr.coef_[0][0])
    row["thr_5to1"] = float(ratio_threshold(y_val, s_val, 5))
    return row, s_val, s_te


# ============================================================================ charts and summary
VOLUME_LABELS = {5848: "17:1", 17200: "50:1", 34400: "100:1", 68800: "200:1", 172000: "500:1", 199020: "full"}


def draw_charts(out, summary, seeds_tbl):
    """The two charts. Also callable on its own, from saved CSVs, to redraw without rerunning.

    volume_curve.png  Story 1: AUC-PR vs legitimate transactions kept, validation and test
                      (3-seed means as lines, each seed model as a faint dot)
    balance_curve.png Story 2: left, AUC-PR vs fraud weight (ranking); right, test log loss before
                      and after Platt scaling vs fraud weight (calibration: where the weight shows)
    """
    blue, orange, ink = "#2a78d6", "#eb6834", "#444441"
    series = [("val_auc_pr", "validation", blue), ("test_auc_pr", "test", orange)]
    vol = summary[summary["story"].isin(["volume", "both"])].sort_values("goods")
    bal = summary[summary["story"].isin(["balance", "both"])].sort_values("weight")

    def style(ax):
        ax.grid(axis="y", color="#e4e2da", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Story 1
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, label, color in series:
        for n in vol.index:
            pts = seeds_tbl.loc[seeds_tbl["config"] == n, col]
            ax.scatter([vol.loc[n, "goods"]] * len(pts), pts, color=color, alpha=0.3, s=22, zorder=2)
        ax.plot(vol["goods"], vol[col], "o-", color=color, lw=2.2, ms=7, zorder=3,
                label=f"{label} (3-seed mean)")
    goods_list = list(vol["goods"])
    for k, n in enumerate(vol.index):                 # ratio labels: under the lowest point, or beside the
        g = vol.loc[n, "goods"]                       # validation mean where the points sit high and close
        low = seeds_tbl.loc[seeds_tbl["config"] == n, ["val_auc_pr", "test_auc_pr"]].min().min()
        high_pt = vol.loc[n, "test_auc_pr"] - vol.loc[n, "val_auc_pr"] < 0.02 and low > 0.8
        crowded_right = k + 1 < len(goods_list) and goods_list[k + 1] / g < 1.3
        if not high_pt:
            xy, off, ha = (g, low), (0, -16), "center"
        elif crowded_right:                           # a close neighbour on the right: label on the left
            xy, off, ha = (g, vol.loc[n, "val_auc_pr"]), (-12, -12), "right"
        else:
            xy, off, ha = (g, vol.loc[n, "val_auc_pr"]), (12, -12), "left"
        ax.annotate(VOLUME_LABELS.get(int(g), n), xy, textcoords="offset points", xytext=off,
                    ha=ha, fontsize=9, color=ink)
    # Flag degenerate seed models (early stopping kept fewer than 10 trees)
    for _, r_ in seeds_tbl[(seeds_tbl["config"].isin(vol.index)) & (seeds_tbl["trees"] < 10)].iterrows():
        ax.annotate(f"seed {int(r_['seed'])}: early stopping kept {int(r_['trees'])} trees",
                    (r_["goods"], r_["val_auc_pr"]), textcoords="offset points", xytext=(10, 0),
                    va="center", fontsize=8, color="#888780")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.margins(y=0.12)
    ax.set_title("AUC-PR vs legitimate transactions kept (all frauds kept; effective balance about 17:1)",
                 loc="left", fontsize=12)
    ax.set(xlabel="Legitimate transactions in training (log scale)", ylabel="AUC-PR")
    style(ax)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "volume_curve.png"), dpi=150)
    plt.close()

    # Story 2
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for col, label, color in series:
        for n in bal.index:
            pts = seeds_tbl.loc[seeds_tbl["config"] == n, col]
            ax[0].scatter([bal.loc[n, "weight"]] * len(pts), pts, color=color, alpha=0.3, s=22, zorder=2)
        ax[0].plot(bal["weight"], bal[col], "o-", color=color, lw=2.2, ms=7, zorder=3, label=f"{label} (3-seed mean)")
    ax[0].set_ylim(0.80, 0.87)                        # same scale idea as Story 1's top end: differences look as small as they are
    ax[0].set_title("Ranking: AUC-PR vs fraud weight (full data)", loc="left", fontsize=12)
    ax[0].set(xlabel="Fraud weight, scale_pos_weight (log scale)", ylabel="AUC-PR")
    ax[0].legend(frameon=False, loc="lower right")
    ax[1].plot(bal["weight"], bal["raw_log_loss"], "o-", color=orange, lw=2.2, ms=7, label="raw score (before Platt)")
    ax[1].plot(bal["weight"], bal["platt_log_loss"], "o-", color=blue, lw=2.2, ms=7, label="after Platt scaling")
    ax[1].set_title("Calibration: test log loss vs fraud weight (lower is better)", loc="left", fontsize=12)
    ax[1].set(xlabel="Fraud weight, scale_pos_weight (log scale)", ylabel="Log loss")
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
    ax[1].legend(frameon=False, loc="upper left")
    for a in ax:
        a.set_xscale("log")
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        style(a)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "balance_curve.png"), dpi=150)
    plt.close()
    log(f"charts saved: {out}/volume_curve.png, {out}/balance_curve.png")


def write_summary_md(out, summary):
    """sweep_summary.md: both stories as markdown tables."""
    vol = summary[summary["story"].isin(["volume", "both"])].sort_values("goods")
    bal = summary[summary["story"].isin(["balance", "both"])].sort_values("weight")

    def md_table(df, cols, heads):
        lines = ["| " + " | ".join(heads) + " |", "| " + " | ".join(["---"] * len(heads)) + " |"]
        for n, r_ in df.iterrows():
            cells = [f"{r_[c_]:,.0f}" if c_ in ("goods", "weight") else f"{r_[c_]:.4f}" for c_ in cols]
            lines.append("| " + " | ".join([n] + cells) + " |")
        return lines
    md = [f"# Sweep summary ({time.strftime('%Y-%m-%d')})", "",
          f"3-seed means. CI = 95% bootstrap interval of test AUC-PR minus {REFERENCE}.", "",
          "## Story 1: volume (effective balance about 17:1)", ""]
    md += md_table(vol, ["goods", "val_auc_pr", "test_auc_pr", "test_auc_pr_dedup", "val_range", "ci95_low", "ci95_high"],
                   ["Config", "Goods", "Val AUC-PR", "Test AUC-PR", "Test dedup", "Val seed range", "CI low", "CI high"])
    md += ["", "## Story 2: balance (full data)", ""]
    md += md_table(bal, ["weight", "val_auc_pr", "test_auc_pr", "ci95_low", "ci95_high",
                         "raw_log_loss", "platt_log_loss", "platt_a", "platt_b"],
                   ["Config", "Weight", "Val AUC-PR", "Test AUC-PR", "CI low", "CI high",
                    "Raw log loss", "Platt log loss", "Platt a", "Platt b"])
    md += ["", f"Actual test fraud rate: {summary['actual_rate_%'].iloc[0]:.3f}%"]
    with open(os.path.join(out, "sweep_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


# ============================================================================ the pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reuse", action="store_true", help="retrain r50 and full-w35 instead of reusing earlier jobs")
    ap.add_argument("--register", action="store_true", help="also register the best configuration (after the gate)")
    ap.add_argument("--register-only", metavar="PACKAGE_DIR", help="register a candidate saved by an earlier run")
    ap.add_argument("--repackage-version", type=int, metavar="N",
                    help="no training: register version N's trees again, packaged with the reason-code serving code")
    args = ap.parse_args()
    if args.register_only:
        section("Registering a saved candidate")
        register(setup(), args.register_only)
        return
    if args.repackage_version:
        repackage(args.repackage_version)
        return

    configs = [dict(c) for c in CONFIGS]            # working copies (fields get added below)
    by_name = {c["name"]: c for c in configs}

    # ------------------------------------------------------------------------ S0
    section("S0. Setup, data, and one training file per configuration")
    ctx = setup()
    tr, va, te, feats, in_train = load_data()
    ch = upload_channels(ctx, tr, va, configs)
    d = {"X_val": va[feats].values, "y_val": va["Class"].values,
         "X_te": te[feats].values, "y_te": te["Class"].values,
         "amt": te["Amount"].values, "keep": ~in_train, "feats": feats}
    for c in configs:
        r = reusable(ctx, c["name"], allow=not args.no_reuse)
        if r:
            c["tuning_job"], c["seed_jobs"], c["reused"] = r["tuning_job"], list(r["seed_jobs"]), True
            log(f"{c['name']}: reusing tuning job {r['tuning_job']} and its 3 seed jobs")

    # ------------------------------------------------------------------------ S1
    n_new = sum(not c.get("reused") for c in configs)
    section(f"S1. Random searches: {n_new} configurations x {CFG['max_jobs']} trials, "
            f"{CFG['tuners_at_once']} searches at a time")
    run_tuning(ctx, ch, configs)
    for c in configs:
        c["best_job"], c["best_hp"] = best_hyperparameters(ctx, c["tuning_job"])
        res = HyperparameterTuningJobAnalytics(c["tuning_job"], sagemaker_session=ctx["sess"]).dataframe()
        res = res.sort_values("FinalObjectiveValue", ascending=False)
        cols = ["FinalObjectiveValue"] + list(RANGES) + ["TrainingJobName"]
        res[cols].head(10).round(4).to_csv(os.path.join(ctx["out"], f"tuning_top10_{c['name']}.csv"), index=False)
        log(f"{c['name']:<10} best trial {res['FinalObjectiveValue'].iloc[0]:.4f} -> {c['best_hp']}")

    # ------------------------------------------------------------------------ S2
    section("S2. Best settings of each configuration retrained with 3 seeds")
    run_seeds(ctx, ch, configs)

    # ------------------------------------------------------------------------ S3
    section("S3. Every seed model scored on validation and test")
    rows, scores = [], {}
    for c in configs:
        for s, job in zip(CFG["seeds"], c["seed_jobs"]):
            row, s_val, s_te = score_seed(model_of_job(ctx, job), d)
            rows.append({"config": c["name"], "story": c["story"], "goods": c["goods"],
                         "weight": c["weight"], "seed": s, "job": job, **row})
            scores[(c["name"], s)] = s_te
            log(f"{c['name']:<10} seed {s}: validation {row['val_auc_pr']:.4f}, test {row['test_auc_pr']:.4f}")
    seeds_tbl = pd.DataFrame(rows)
    seeds_tbl.round(6).to_csv(os.path.join(ctx["out"], "sweep_seeds.csv"), index=False)
    log("table saved: sweep_seeds.csv (every metric, every seed model)")

    # 3-seed means and ranges per configuration, in the CONFIGS order
    grp = seeds_tbl.drop(columns=["story", "seed", "job"]).groupby("config", sort=False)
    summary = grp.mean()
    summary["val_range"] = grp["val_auc_pr"].max() - grp["val_auc_pr"].min()
    summary["test_range"] = grp["test_auc_pr"].max() - grp["test_auc_pr"].min()
    summary.insert(0, "story", [by_name[n]["story"] for n in summary.index])

    # Bootstrap: each configuration's representative seed (closest to its mean) vs the reference's,
    # on the SAME resampled test rows each time -> 95% interval of the test AUC-PR difference
    rep = {}
    for n in summary.index:
        v = seeds_tbl[seeds_tbl["config"] == n].set_index("seed")["test_auc_pr"]
        rep[n] = int((v - v.mean()).abs().idxmin())
    y_te = d["y_te"]
    rng = np.random.default_rng(42)
    draws = [rng.integers(0, len(y_te), len(y_te)) for _ in range(CFG["bootstrap"])]
    draws = [i for i in draws if y_te[i].sum()]
    ref_s = scores[(REFERENCE, rep[REFERENCE])]
    ref_ap = np.array([average_precision_score(y_te[i], ref_s[i]) for i in draws])
    lo, hi = {}, {}
    for n in summary.index:
        if n == REFERENCE:
            lo[n], hi[n] = 0.0, 0.0
            continue
        s_te = scores[(n, rep[n])]
        diff = np.array([average_precision_score(y_te[i], s_te[i]) for i in draws]) - ref_ap
        lo[n], hi[n] = np.percentile(diff, [2.5, 97.5])
        log(f"{n:<10} test AUC-PR minus {REFERENCE}: 95% interval {lo[n]:+.4f} to {hi[n]:+.4f}")
    summary["ci95_low"] = pd.Series(lo)
    summary["ci95_high"] = pd.Series(hi)
    save_table(ctx, summary.reset_index().round(4), "sweep_summary.csv",
               f"Sweep summary (3-seed means; ci95 = test AUC-PR minus {REFERENCE})")

    # ------------------------------------------------------------------------ S4
    section("S4. Charts and summary")
    draw_charts(ctx["out"], summary, seeds_tbl)
    write_summary_md(ctx["out"], summary)

    # ------------------------------------------------------------------------ S5
    # The best configuration is chosen on VALIDATION; the test set then gates it against the champion.
    best = summary["val_auc_pr"].idxmax()
    section(f"S5. Best configuration on validation: {best}")
    if by_name[best].get("reused"):
        log(f"{best} is the earlier challenger already in the registry: nothing new to register.")
        return
    prepare_candidate(ctx, by_name[best], seeds_tbl, d, register_now=args.register)


# ============================================================================ registration
def champion(ctx):
    """The latest Approved version in the registry: (label, model URL), or (None, None) if there is none."""
    sm = ctx["sm"]
    try:
        approved = sm.list_model_packages(ModelPackageGroupName=CFG["group"], ModelApprovalStatus="Approved",
                                          SortBy="CreationTime", SortOrder="Descending")["ModelPackageSummaryList"]
    except sm.exceptions.ClientError:
        approved = []
    if not approved:
        return None, None
    arn = approved[0]["ModelPackageArn"]
    url = sm.describe_model_package(ModelPackageName=arn)["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    return f"champion v{arn.split('/')[-1]}", url


def prepare_candidate(ctx, c, seeds_tbl, d, register_now):
    """Gate the best configuration against the champion on test and save a registration candidate.

    Gate: the 3-seed mean test AUC-PR must beat the champion's (or reach CFG["min_auc_pr_first"]
    when the registry is empty). The representative seed (closest to the mean) is packaged.
    Thresholds and Platt values are stored at FULL precision (a rounded cutoff changes alerts).
    """
    mine = seeds_tbl[seeds_tbl["config"] == c["name"]].set_index("seed")
    mean_ap = mine["test_auc_pr"].mean()
    rep = int((mine["test_auc_pr"] - mean_ap).abs().idxmin())
    label, url = champion(ctx)
    if label:
        y_te = d["y_te"]
        champ_scores = load_booster(url, os.path.join("models", "champion")).predict(xgb.DMatrix(d["X_te"]))
        mine_scores = model_of_job(ctx, c["seed_jobs"][CFG["seeds"].index(rep)]).predict(xgb.DMatrix(d["X_te"]))
        champ_ap = average_precision_score(y_te, champ_scores)
        # Bootstrap: models fixed, test rows resampled -> how much the gain moves by chance
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(CFG["bootstrap"]):
            i = rng.integers(0, len(y_te), len(y_te))
            if y_te[i].sum():
                diffs.append(average_precision_score(y_te[i], mine_scores[i]) - average_precision_score(y_te[i], champ_scores[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # Two conditions: a material lift AND an interval that excludes zero
        passes = bool(mean_ap - champ_ap >= CFG["min_lift"] and lo > 0)
        reason = (f"Sweep best ({c['name']}): mean test AUC-PR {mean_ap:.4f} vs {label} {champ_ap:.4f}"
                  f" (lift {mean_ap - champ_ap:+.4f}; 95% CI {lo:+.4f} to {hi:+.4f}; needs >= {CFG['min_lift']} and CI above 0)")
    else:
        passes = bool(mean_ap >= CFG["min_auc_pr_first"])
        reason = f"First model ({c['name']}): mean test AUC-PR {mean_ap:.4f} vs minimum {CFG['min_auc_pr_first']}"
    status = "Approved" if passes else "Rejected"
    log(f"GATE: {status}. {reason}")

    pkg_dir = os.path.join(ctx["out"], "register", c["name"])
    r_ = mine.loc[rep]
    build_package(model_of_job(ctx, c["seed_jobs"][CFG["seeds"].index(rep)]), pkg_dir,
                  scoring_settings(r_["platt_a"], r_["platt_b"], r_["thr_5to1"], d["feats"]))
    meta = {"config": c["name"], "goods_in_training": c["goods"], "scale_pos_weight": c["weight"],
            "seed_used": rep, "val_auc_pr_mean": round(mine["val_auc_pr"].mean(), 4),
            "test_auc_pr": round(r_["test_auc_pr"], 4), "mean_auc_pr_3_seeds": round(mean_ap, 4),
            "dr_5to1": round(r_["DR@5:1"], 4), "vdr_5to1": round(r_["VDR@5:1"], 4),
            "thr_5to1": repr(float(r_["thr_5to1"])),                      # full precision
            "platt_a": repr(float(r_["platt_a"])), "platt_b": repr(float(r_["platt_b"])),
            "tuning_job": c["tuning_job"], "best_training_job": c["best_job"],
            "training_code": "train.py script mode XGBoost 1.7-1",
            "serving_code": "inference.py: score + top 3 TreeSHAP reason codes",
            "score_scale": "0-100 where 50 = fraud odds 1:1000 and +3 points doubles the odds"}
    cand = {"config": c["name"], "status": status, "reason": reason, "seed": rep,
            "description": f"Sweep configuration {c['name']} (train.py script mode) seed {rep}",
            "meta": {k_: str(v_).replace(",", "") for k_, v_ in meta.items()}}   # no commas in metadata
    with open(os.path.join(pkg_dir, "candidate.json"), "w") as f:
        json.dump(cand, f, indent=2)
    log(f"Candidate saved in {pkg_dir}")
    if register_now:
        register(ctx, pkg_dir)
    else:
        log("DRY RUN: nothing written to the registry. To register this candidate:")
        log(f"    python pipeline/train_pipeline.py --register-only {pkg_dir}")


def scoring_settings(platt_a, platt_b, thr_5to1, feats):
    """What inference.py needs besides the trees, stored as code/scoring.json (FULL precision)."""
    sc = CFG_SCORE
    return {"platt_a": float(platt_a), "platt_b": float(platt_b), "thr_5to1": float(thr_5to1),
            "features": list(feats), "pdo": sc["pdo"], "anchor_score": sc["anchor_score"],
            "anchor_odds": sc["anchor_odds"], "explain_alerts_only": CFG["explain_alerts_only"]}


def build_package(booster, pkg_dir, scoring):
    """Build model.tar.gz = xgboost-model + code/inference.py + code/scoring.json in pkg_dir.

    The trees are re-saved with the local XGBoost 2.1 as "xgboost-model", the file name the 3.0-5
    serving image looks for; inference.py is copied from the pipeline/ folder next to this file.
    """
    os.makedirs(os.path.join(pkg_dir, "code"), exist_ok=True)
    booster.save_model(os.path.join(pkg_dir, "xgboost-model"))
    with open(os.path.join(HERE, "inference.py")) as f_in, \
         open(os.path.join(pkg_dir, "code", "inference.py"), "w") as f_out:
        f_out.write(f_in.read())
    with open(os.path.join(pkg_dir, "code", "scoring.json"), "w") as f:
        json.dump(scoring, f, indent=2)
    with tarfile.open(os.path.join(pkg_dir, "model.tar.gz"), "w:gz") as tar:
        tar.add(os.path.join(pkg_dir, "xgboost-model"), arcname="xgboost-model")
        tar.add(os.path.join(pkg_dir, "code", "inference.py"), arcname="code/inference.py")
        tar.add(os.path.join(pkg_dir, "code", "scoring.json"), arcname="code/scoring.json")
    log(f"Package built: {pkg_dir}/model.tar.gz (trees + inference.py + scoring.json)")


def repackage(version):
    """Register version N's trees again as a new version, now with the reason-code serving code.

    No training: the trees are copied exactly, so every score is identical (checked below on test).
    Calibration and the 5:1 cutoff are recomputed from validation at FULL precision (version 4 stored
    them rounded to 4 decimals). The new version gets version N's status and metadata, plus a note.
    """
    section(f"Repackaging version {version} with the reason-code serving code (no training)")
    ctx = setup()
    sm = ctx["sm"]
    arn = next(p["ModelPackageArn"] for p in
               sm.list_model_packages(ModelPackageGroupName=CFG["group"], MaxResults=100)["ModelPackageSummaryList"]
               if p.get("ModelPackageVersion") == version)
    desc = sm.describe_model_package(ModelPackageName=arn)
    url = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    old_meta = desc.get("CustomerMetadataProperties", {})
    booster = load_booster(url, os.path.join("models", f"v{version}"))

    _, va, te, feats, _ = load_data()
    s_val = booster.predict(xgb.DMatrix(va[feats].values))
    s_te = booster.predict(xgb.DMatrix(te[feats].values))
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(s_val).reshape(-1, 1), va["Class"].values)
    thr = ratio_threshold(va["Class"].values, s_val, 5)
    test_ap = average_precision_score(te["Class"].values, s_te)
    log(f"Version {version} test AUC-PR recomputed: {test_ap:.4f} (registry says {old_meta.get('test_auc_pr', 'n/a')})")
    log(f"Platt a = {lr.intercept_[0]!r}, b = {lr.coef_[0][0]!r}; 5:1 cutoff = {thr!r} (full precision)")

    pkg_dir = os.path.join(ctx["out"], "register", f"v{version}-reasons")
    build_package(booster, pkg_dir, scoring_settings(lr.intercept_[0], lr.coef_[0][0], thr, feats))
    meta = {**old_meta, "source_version": version,
            "serving_code": "inference.py: score + top 3 TreeSHAP reason codes",
            "platt_a": repr(float(lr.intercept_[0])), "platt_b": repr(float(lr.coef_[0][0])),
            "thr_5to1": repr(float(thr)), "test_auc_pr": round(test_ap, 4)}
    cand = {"config": f"v{version}-reasons", "status": desc["ModelApprovalStatus"],
            "reason": f"Same trees as version {version} (test AUC-PR {test_ap:.4f}); adds reason codes",
            "seed": old_meta.get("seed_used", ""),
            "description": f"Version {version} trees + reason-code serving code (inference.py)",
            "meta": {k_: str(v_).replace(",", "") for k_, v_ in meta.items()}}
    with open(os.path.join(pkg_dir, "candidate.json"), "w") as f:
        json.dump(cand, f, indent=2)
    register(ctx, pkg_dir)


def register(ctx, pkg_dir):
    """Upload one saved package, create the next version in the registry, then set its status."""
    with open(os.path.join(pkg_dir, "candidate.json")) as f:
        cand = json.load(f)
    sm = ctx["sm"]
    try:                                          # first run in a new account: create the group once
        sm.describe_model_package_group(ModelPackageGroupName=CFG["group"])
    except sm.exceptions.ClientError:
        sm.create_model_package_group(ModelPackageGroupName=CFG["group"],
                                      ModelPackageGroupDescription="XGBoost credit card fraud models")
        log(f"Created model package group {CFG['group']}")
    url = ctx["sess"].upload_data(os.path.join(pkg_dir, "model.tar.gz"), bucket=ctx["bucket"],
                                  key_prefix=f"{CFG['prefix']}/registered/{time.strftime('%Y%m%d-%H%M%S')}")
    image = sagemaker.image_uris.retrieve("xgboost", ctx["region"], version=CFG["serve_version"])
    resp = sm.create_model_package(
        ModelPackageGroupName=CFG["group"], ModelPackageDescription=cand["description"],
        InferenceSpecification={
            # The recipe: this image + this archive + "run OUR inference.py from the archive's code/ folder"
            "Containers": [{"Image": image, "ModelDataUrl": url,
                            "Environment": {"SAGEMAKER_PROGRAM": "inference.py",
                                            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                                            "SAGEMAKER_REGION": ctx["region"]}}],
            "SupportedContentTypes": ["text/csv"], "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge"],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"]},
        ModelApprovalStatus="PendingManualApproval", CustomerMetadataProperties=cand["meta"])
    arn = resp["ModelPackageArn"]
    sm.update_model_package(ModelPackageArn=arn, ModelApprovalStatus=cand["status"],
                            ApprovalDescription=cand["reason"])
    log(f"Registered {cand['config']} as {CFG['group']}/{arn.split('/')[-1]} with status {cand['status']}")


# Run main() only when the file is executed (python pipeline/train_pipeline.py), not when imported.
if __name__ == "__main__":
    main()
