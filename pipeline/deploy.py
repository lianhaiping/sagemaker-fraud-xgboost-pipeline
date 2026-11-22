"""
deploy.py - the deployment pipeline (the counterpart of train_pipeline.py).

WHAT IT DOES
    Deploys the model the registry APPROVED, proves the live endpoint behaves exactly like the
    offline evaluation, measures real-time latency, and deletes the endpoint again.
    Steps (the section headers it prints):
        F0  setup: SageMaker session, the same data split as train_pipeline.py (same test rows)
        F1  find the model package to deploy: latest Approved version, or --version N
        F2  deploy that package as a real-time endpoint (billing starts here)
        F3  smoke test: 5 frauds + 5 legitimate test transactions (and, for packages with the
            reason-code serving code, a check that the served score and the reason codes come back)
        F4  whole test set through the endpoint: must reproduce the registry's numbers
        F5  real-time latency: one transaction per call, p50 / p95 / p99
        F6  CloudWatch: model time vs SageMaker overhead, per minute
        F7  delete the endpoint, its configuration and the model object
        then: a short deploy_report.md with every check and its PASS / FAIL

WHERE IT RUNS
    In a Studio terminal (File -> New -> Terminal), from the REPOSITORY ROOT, like
    train_pipeline.py (outputs/ is created there). It only sends requests to AWS: the endpoint itself runs on a
    separate ml.m5.xlarge machine that SageMaker starts and stops.

HOW TO RUN
    python pipeline/deploy.py --all                 # deploy -> test -> ALWAYS delete (the safe default)
    python pipeline/deploy.py --all --keep          # same, but leave the endpoint running (remember to delete!)
    python pipeline/deploy.py --all --version 4     # deploy a specific approved version instead of the latest
    python pipeline/deploy.py --deploy              # only F0-F2: deploy and stop (endpoint keeps billing)
    python pipeline/deploy.py --test                # only F3-F6, against the endpoint from the last --deploy
    python pipeline/deploy.py --delete              # only F7, for the endpoint from the last --deploy
    python pipeline/deploy.py --delete --endpoint fraud-xgb-v4-20261122-023358    # delete a named endpoint

WHY --all DELETES IN A "finally:" BLOCK
    An endpoint bills by the hour from the moment it exists. If a test crashes halfway, a plain
    script would exit and leave it running. With try/finally, the delete runs whatever happens:
    a failed check, a Python error, or Ctrl+C. That is the main reason a deploy script is safer
    than deploying by hand.

OUTPUTS
    outputs/deploy-<timestamp>/ : smoke_test.csv, reproduction.csv, latency.csv, cloudwatch.csv,
    deploy_report.md, and deploy_state.json (which endpoint is live, so --test and --delete can
    find it later). The exit code is 0 when every check passed and 1 otherwise, so a CI/CD
    pipeline (for example GitHub Actions) can stop a release on a failed check.
"""
# ============================================================================ imports
import argparse            # command-line options (--all, --deploy, --test, --delete, --version ...)
import glob                # find the most recent deploy_state.json
import json                # save and read deploy_state.json
import os                  # folders and file paths
import time                # timestamps, timing calls, waiting between status checks
from datetime import datetime, timedelta, timezone   # CloudWatch needs a time window in UTC

import boto3               # low-level AWS clients: sagemaker (registry, endpoints),
                           # sagemaker-runtime (send transactions), cloudwatch (metrics)
import numpy as np
import pandas as pd
import sagemaker           # SageMaker Python SDK v2: used here only for ModelPackage.deploy
from sagemaker import ModelPackage, get_execution_role
from sklearn.datasets import fetch_openml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

# ============================================================================ settings (edit here)
# Everything that defines a deployment lives in one dictionary, the same idea as train_pipeline.py's CFG.
CFG = {
    "group": "fraud-xgb-models",         # model package group in the registry (train_pipeline.py fills it)
    "instance_type": "ml.m5.xlarge",     # endpoint machine (about $0.23 an hour in us-east-2)
    "instance_count": 1,                 # machines behind the endpoint; production uses 2+ for
                                         # high availability (one per Availability Zone)
    "endpoint_prefix": "fraud-xgb",      # endpoint name = fraud-xgb-v<version>-<timestamp>
    "batch_rows": 1000,                  # rows per request in F4 (a request body must stay < 6 MB;
                                         # 1,000 rows of 29 numbers is about 0.3 MB)
    "smoke_per_class": 5,                # F3: frauds and legitimate rows to send
    "latency_calls": 200,                # F5: single-transaction calls to time
    "latency_warmup": 10,                # F5: first calls ignored (opening the HTTPS connection)
    "latency_budget_ms": 50,             # F5: p99 must stay under this (card networks typically
                                         # allow fraud scoring roughly 50-100 ms)
    "tolerance": 0.001,                  # F4: endpoint vs registry numbers may differ by this much
    "cloudwatch_wait_s": 90,             # F6: metrics reach CloudWatch about a minute after the calls
    # The 0-100 score (same scale as train_pipeline.py):
    # 50 = fraud odds 1:1000 (about 0.1%), and every +3 points doubles the odds.
    "score": {"pdo": 3, "anchor_score": 50, "anchor_odds": 1 / 1000},
}


# ============================================================================ printing helpers
def log(msg=""):
    """Print a line with the current time; flush=True writes it to the log immediately."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def section(title):
    """Print a visible section header, so the log reads step by step."""
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90, flush=True)


def save_table(ctx, df, name, title):
    """Save a DataFrame as CSV in the run folder and print it in full."""
    path = os.path.join(ctx["out"], name)
    df.to_csv(path, index=False)
    print(f"\n{title}\n{df.to_string(index=False)}\n", flush=True)
    log(f"table saved: {path}")


def check(ctx, name, passed, detail):
    """Record one PASS / FAIL check; all of them go into deploy_report.md and the exit code."""
    ctx["checks"].append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})
    log(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")


# ============================================================================ F0: setup and data
def setup():
    """Connect to AWS and create a timestamped output folder for this run.

    Returns "ctx" (context), a small dictionary handed to every step instead of global variables,
    so each function shows what it uses. No access keys anywhere: inside Studio, boto3 and the SDK
    pick up the space's IAM role automatically (the same role the training pipeline uses).
    """
    sess = sagemaker.Session()
    ctx = {
        "sess": sess,                                        # SDK session (used by ModelPackage)
        "role": get_execution_role(),                        # the IAM role the endpoint model uses
        "region": sess.boto_region_name,                     # us-east-2
        "sm": boto3.client("sagemaker"),                     # registry + endpoint management API
        "runtime": boto3.client("sagemaker-runtime"),        # the API that SCORES transactions
        "cw": boto3.client("cloudwatch"),                    # endpoint metrics
        "out": os.path.join("outputs", "deploy-" + time.strftime("%Y%m%d-%H%M%S")),
        "checks": [],                                        # filled by check()
    }
    os.makedirs(ctx["out"], exist_ok=True)
    log(f"SDK {sagemaker.__version__} | region {ctx['region']} | outputs in {ctx['out']}")
    return ctx


def load_test_data():
    """Download the dataset and recreate the training pipeline's split exactly; return only the TEST part.

    Same code as train_pipeline.py's load_data: 70% train, then the remaining 30%
    split in half (15% validation, 15% test), stratified by Class, random_state=42. Same seed and
    same order of calls -> the same 42,722 test rows the model was graded on, so F4 can compare the
    endpoint with the registry's numbers row for row.
    """
    log("Downloading OpenML dataset 1597 (1 to 3 minutes)...")
    df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
    df["Class"] = df["Class"].astype(str).astype(int)        # labels arrive as text categories
    feats = [c for c in df.columns if c != "Class"]          # V1..V28 and Amount: the 29 features
    _, tmp = train_test_split(df, test_size=0.30, stratify=df["Class"], random_state=42)
    _, te = train_test_split(tmp, test_size=0.50, stratify=tmp["Class"], random_state=42)
    # .values: plain numbers in the trained column order, no column names (what the endpoint expects)
    X = te[feats].values
    y = te["Class"].values
    amount = te["Amount"].values                             # for the value detection rate (VDR)
    log(f"test rows={len(y):,}  frauds={int(y.sum())}")
    return X, y, amount


# ============================================================================ F1: find the package
def find_package(ctx, version=None):
    """Return the registry entry to deploy: the latest APPROVED version, or a given version number.

    This is the "deploy what was approved" rule: the script never builds a model by hand. It reads
    the serving image, the model file and the training pipeline's metadata (calibration, thresholds, expected
    results) from the registry, so the deployment and its tests use exactly what the gate approved.
    """
    sm = ctx["sm"]
    if version is None:
        # Newest first, only Approved ones: the first entry is the current champion
        pkgs = sm.list_model_packages(ModelPackageGroupName=CFG["group"], ModelApprovalStatus="Approved",
                                      SortBy="CreationTime", SortOrder="Descending")
        if not pkgs["ModelPackageSummaryList"]:
            raise SystemExit(f"No Approved version in {CFG['group']}: nothing to deploy.")
        arn = pkgs["ModelPackageSummaryList"][0]["ModelPackageArn"]
    else:
        # A versioned package's ARN is the group's ARN path plus "/<version>"; list and pick it
        pkgs = sm.list_model_packages(ModelPackageGroupName=CFG["group"], MaxResults=100)
        match = [p for p in pkgs["ModelPackageSummaryList"] if p.get("ModelPackageVersion") == version]
        if not match:
            raise SystemExit(f"Version {version} not found in {CFG['group']}.")
        arn = match[0]["ModelPackageArn"]

    desc = sm.describe_model_package(ModelPackageName=arn)
    # Safety rule: never deploy something the gate rejected, even when asked by version number
    if desc["ModelApprovalStatus"] != "Approved":
        raise SystemExit(f"Version {desc['ModelPackageVersion']} is {desc['ModelApprovalStatus']}; "
                         "only Approved versions may be deployed.")

    container = desc["InferenceSpecification"]["Containers"][0]   # serving image + model file in S3
    meta = desc.get("CustomerMetadataProperties", {})             # training pipeline metadata (all text)
    pkg = {
        "arn": arn,
        "version": desc["ModelPackageVersion"],
        "image": container["Image"].split("/")[-1],               # name only: the full URI has the account
        "reason": desc.get("ApprovalDescription", ""),
        "meta": meta,
        # True when the package carries inference.py (reason codes): its answers are JSON
        "reasons": "SAGEMAKER_PROGRAM" in container.get("Environment", {}),
    }
    log(f"Reason-code serving code in the package: {'yes' if pkg['reasons'] else 'no (scores only)'}")
    log(f"Deploying version {pkg['version']} ({desc['ModelApprovalStatus']}) | image {pkg['image']}")
    log(f"Approval reason: {pkg['reason']}")
    for k, v in sorted(meta.items()):
        print(f"    {k}: {v}", flush=True)
    return pkg


# ============================================================================ F2: deploy
def deploy(ctx, pkg):
    """Create a real-time endpoint from the approved package and wait until it is InService.

    ModelPackage(...).deploy() does three AWS calls for us:
        1. create_model            - "this image + this model.tar.gz, run with this role"
        2. create_endpoint_config  - "one ml.m5.xlarge machine, all traffic to that model"
        3. create_endpoint         - start the machine, pull the image, load the model, and call
                                     the container's /ping until it answers -> InService
    It takes 5 to 10 minutes. Billing starts when the endpoint is created, not when it is used.
    The SDK prints a v2 deprecation notice here (v3 uses ModelBuilder instead); not an error.
    """
    name = f"{CFG['endpoint_prefix']}-v{pkg['version']}-{time.strftime('%Y%m%d-%H%M%S')}"
    log(f"Creating endpoint {name} on {CFG['instance_count']} x {CFG['instance_type']} (5 to 10 minutes)...")
    model = ModelPackage(role=ctx["role"], model_package_arn=pkg["arn"], sagemaker_session=ctx["sess"])
    # wait=True blocks until InService (or raises if the deployment fails)
    model.deploy(initial_instance_count=CFG["instance_count"], instance_type=CFG["instance_type"],
                 endpoint_name=name, wait=True)
    save_state(ctx, name, pkg)                # remember the endpoint for --test / --delete later
    log(f"Endpoint in service: {name}  (billing until F7 deletes it)")
    return name


def save_state(ctx, endpoint, pkg):
    """Write deploy_state.json: which endpoint is live and which package it serves.

    A later "python deploy.py --test" or "--delete" (a new process, with no memory of this one)
    reads the newest state file to find the endpoint and the package metadata it needs.
    """
    state = {"endpoint": endpoint, "package_arn": pkg["arn"], "version": pkg["version"],
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(ctx["out"], "deploy_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    """Read the newest outputs/deploy-*/deploy_state.json (folder names sort by timestamp)."""
    files = sorted(glob.glob(os.path.join("outputs", "deploy-*", "deploy_state.json")))
    if not files:
        raise SystemExit("No deploy_state.json found: run --deploy first, or pass --endpoint NAME.")
    with open(files[-1]) as f:
        return json.load(f)


# ============================================================================ scoring helpers
def call_endpoint(ctx, endpoint, X):
    """Send rows to the endpoint and return one raw fraud score per row.

    This is the raw API call a bank's authorization service would make (from Java, Go, ...):
        1. CSV body: one line per transaction, 29 numbers, no label, no header
        2. invoke_endpoint over HTTPS, signed with the role's credentials (no keys in code)
        3. the XGBoost container receives it on /invocations and scores each row
        4. the answer depends on the package:
           - with the reason-code serving code (inference.py in the package): JSON, one object per
             transaction with raw_score, fraud_probability, score_0_100, alert_5to1 and reasons.
             The full objects are kept in ctx["last_predictions"] for the smoke test.
           - with AWS's default serving code (older packages): CSV, one number per row
    """
    body = "\n".join(",".join(repr(float(v)) for v in row) for row in np.atleast_2d(X))
    if ctx.get("json_mode"):
        resp = ctx["runtime"].invoke_endpoint(EndpointName=endpoint, ContentType="text/csv",
                                              Accept="application/json", Body=body)
        preds = json.loads(resp["Body"].read().decode())["predictions"]
        ctx["last_predictions"] = preds
        return np.array([p["raw_score"] for p in preds])
    resp = ctx["runtime"].invoke_endpoint(EndpointName=endpoint, ContentType="text/csv", Body=body)
    text = resp["Body"].read().decode().replace("\n", ",")
    return np.array([float(v) for v in text.split(",") if v.strip() != ""])


def calibrate(raw, meta):
    """Raw endpoint output -> calibrated fraud probability and the 0-100 score.

    1. Platt calibration (fitted on validation by train_pipeline.py, stored as platt_a / platt_b):
           calibrated log-odds = a + b * logit(raw)
       This undoes the effect of training with weight 35 (raw scores are not true probabilities).
    2. Scorecard scaling: score = 50 + 3 * log2(odds / 0.001), capped to 0-100, so 50 means odds
       1:1000 and every 3 points doubles the odds. The same cutoff means the same risk for any model.
    """
    a, b = float(meta["platt_a"]), float(meta["platt_b"])
    raw = np.clip(raw.astype(float), 1e-9, 1 - 1e-9)
    log_odds = a + b * np.log(raw / (1 - raw))
    prob = 1 / (1 + np.exp(-log_odds))
    sc = CFG["score"]
    score = sc["anchor_score"] + sc["pdo"] * (log_odds - np.log(sc["anchor_odds"])) / np.log(2)
    return prob, np.clip(score, 0, 100)


# ============================================================================ F3: smoke test
def smoke_test(ctx, endpoint, meta, X, y, amount):
    """Send 5 known frauds and 5 legitimate transactions; show raw score, probability, score, alert.

    A smoke test answers "is it alive and sensible?" before the bigger checks: the endpoint answers,
    returns one number per row, frauds mostly score high and legitimate rows sit in the 35-55 band.
    The rows are drawn with a fixed seed, so every run sends the same 10 transactions (the first live run
    alerted 4 of 5 frauds; the missed one was a $276 "hidden" fraud scoring 52.8).
    """
    thr = float(meta["thr_5to1"])               # raw cutoff for a 5:1 false-positive ratio (validation)
    rng = np.random.default_rng(0)
    n = CFG["smoke_per_class"]
    idx = np.concatenate([rng.choice(np.where(y == 1)[0], n, replace=False),
                          rng.choice(np.where(y == 0)[0], n, replace=False)])
    raw = call_endpoint(ctx, endpoint, X[idx])
    prob, score = calibrate(raw, meta)
    table = pd.DataFrame({
        "actual": np.where(y[idx] == 1, "fraud", "legit"),
        "amount": amount[idx].round(2),
        "raw_score": raw.round(5),
        "fraud_prob_%": (prob * 100).round(2),
        "score_0_100": score.round(1),
        "alert_at_5to1": np.where(raw >= thr, "ALERT", "-"),
    })
    if ctx.get("json_mode"):
        # Reason codes from the endpoint: the features pushing each transaction towards fraud
        preds = ctx["last_predictions"]
        table["reasons"] = [", ".join(r["feature"] for r in p["reasons"]) for p in preds]
        served = np.array([p["score_0_100"] for p in preds])
        check(ctx, "served 0-100 score matches the registry calibration",
              np.allclose(served, score.round(1), atol=0.15),
              f"largest gap {np.abs(served - score.round(1)).max():.2f} points")
        alerted = [p for p, f in zip(preds, raw >= thr) if f]
        check(ctx, "every alert comes with reason codes", all(len(p["reasons"]) > 0 for p in alerted),
              f"{sum(len(p['reasons']) > 0 for p in alerted)} of {len(alerted)} alerts have reasons")
    save_table(ctx, table, "smoke_test.csv", "Smoke test: 5 frauds and 5 legitimate test transactions")
    check(ctx, "smoke test: one score per row", len(raw) == len(idx), f"{len(raw)} scores for {len(idx)} rows")
    caught = int(((raw >= thr) & (y[idx] == 1)).sum())
    false_alarms = int(((raw >= thr) & (y[idx] == 0)).sum())
    check(ctx, "smoke test: frauds alerted, no false alarms", caught >= n - 1 and false_alarms == 0,
          f"{caught} of {n} frauds alerted, {false_alarms} false alarms")


# ============================================================================ F4: reproduction
def reproduce(ctx, endpoint, meta, X, y, amount):
    """Score all 42,722 test rows through the endpoint and compare with the registry's numbers.

    If the endpoint's AUC-PR, detection rate and value detection rate equal what the training pipeline measured
    offline, then nothing changed between evaluation and production: same model, same serving
    behavior (the pipeline scored with XGBoost 2.1 locally; the endpoint runs the 3.0-5 image).

    Note on the FP ratio: it is reported, not checked. In the first live run the 5:1 cutoff was
    stored rounded to 4 decimals (0.0023), so the endpoint raised one false alarm fewer than the
    offline evaluation (5.16 vs 5.17): a metadata lesson, not a model difference. train_pipeline.py
    now stores thresholds and Platt values at full precision.
    """
    thr = float(meta["thr_5to1"])
    t0 = time.time()
    k = CFG["batch_rows"]
    raw = np.concatenate([call_endpoint(ctx, endpoint, X[i:i + k]) for i in range(0, len(X), k)])
    secs = time.time() - t0
    log(f"Scored {len(raw):,} rows in {secs:.1f} s ({1000 * secs / len(raw):.2f} ms per row in batches of {k})")
    check(ctx, "full test: one score per row", len(raw) == len(y), f"{len(raw):,} scores for {len(y):,} rows")

    alerts, fraud = raw >= thr, y == 1
    tp, fp = int((alerts & fraud).sum()), int((alerts & ~fraud).sum())
    live = {
        "test_auc_pr": average_precision_score(y, raw),
        "dr_5to1": tp / fraud.sum(),                                   # detection rate = recall
        "vdr_5to1": amount[alerts & fraud].sum() / amount[fraud].sum(),  # share of fraud DOLLARS caught
        "fp_ratio_5to1": fp / max(tp, 1),                              # false alarms per fraud caught
    }
    rows = []
    for key, value in live.items():
        expected = float(meta[key]) if key in meta else np.nan        # offline value from the training pipeline (registry)
        diff = value - expected
        rows.append({"metric": key, "endpoint": round(value, 4), "registry (offline)": expected,
                     "difference": round(diff, 4) if not np.isnan(diff) else np.nan})
        if key in meta:   # the three model checks; the FP ratio is not in the metadata (see note)
            check(ctx, f"reproduces {key}", abs(diff) <= CFG["tolerance"],
                  f"endpoint {value:.4f} vs registry {expected:.4f}")
    save_table(ctx, pd.DataFrame(rows), "reproduction.csv", "Endpoint vs registry (offline evaluation)")
    log(f"Alerts at 5:1: {int(alerts.sum())} (frauds caught {tp} of {int(fraud.sum())}, false alarms {fp})")


# ============================================================================ F5: latency
def latency(ctx, endpoint, X):
    """Time single-transaction calls, the way card authorizations arrive (one at a time).

    Batch throughput (F4) is not the number that matters for authorizations: each one is a separate
    request with its own fixed costs. We send 200 one-row requests, drop the first 10 (the HTTPS
    connection is opened and warmed up then), and report percentiles. In payments the tail (p99)
    matters more than the average: the slowest authorizations are the ones that time out.
    The first live run measured p50 12.0 ms and p99 14.7 ms from Studio, network included.
    """
    lat_ms = []
    for i in range(CFG["latency_calls"]):
        t0 = time.perf_counter()                 # perf_counter: the most precise clock for short timings
        call_endpoint(ctx, endpoint, X[i:i + 1])
        lat_ms.append(1000 * (time.perf_counter() - t0))
    lat = np.array(lat_ms[CFG["latency_warmup"]:])
    pct = {p: float(np.percentile(lat, p)) for p in (50, 95, 99)}
    table = pd.DataFrame([{"calls": len(lat), "p50_ms": round(pct[50], 1), "p95_ms": round(pct[95], 1),
                           "p99_ms": round(pct[99], 1), "max_ms": round(float(lat.max()), 1)}])
    save_table(ctx, table, "latency.csv", "Single-transaction latency (ms, round trip from Studio)")
    check(ctx, "latency p99 within budget", pct[99] <= CFG["latency_budget_ms"],
          f"p99 {pct[99]:.1f} ms vs budget {CFG['latency_budget_ms']} ms")


# ============================================================================ F6: CloudWatch
def cloudwatch(ctx, endpoint):
    """Read the endpoint's metrics: calls per minute, model time and SageMaker overhead.

    SageMaker publishes, per minute: Invocations (requests), ModelLatency (time inside the container,
    scoring) and OverheadLatency (SageMaker's own routing). Both latencies are in MICROseconds.
    Round trip = model + overhead + network. The first live run found about 3 + 6 + 3 ms for single calls:
    fixed per-request costs dominate, so a bigger model hardly changes latency. Metrics arrive
    about a minute late, hence the wait before reading them.
    """
    log(f"Waiting {CFG['cloudwatch_wait_s']} s for CloudWatch to receive the metrics...")
    time.sleep(CFG["cloudwatch_wait_s"])
    now = datetime.now(timezone.utc)
    rows = {}
    for metric, stat in [("Invocations", "Sum"), ("ModelLatency", "Average"), ("OverheadLatency", "Average")]:
        r = ctx["cw"].get_metric_statistics(
            Namespace="AWS/SageMaker", MetricName=metric,
            Dimensions=[{"Name": "EndpointName", "Value": endpoint},
                        {"Name": "VariantName", "Value": "AllTraffic"}],   # the SDK's default variant
            StartTime=now - timedelta(minutes=30), EndTime=now, Period=60, Statistics=[stat])
        for p in r["Datapoints"]:
            rows.setdefault(p["Timestamp"], {})[metric] = p[stat]
    if not rows:
        log("No CloudWatch datapoints yet (they can take a few minutes); skipping this table.")
        return
    table = pd.DataFrame(rows).T.sort_index()
    table.index.name = "minute_utc"
    for col in ["ModelLatency", "OverheadLatency"]:
        if col in table:
            table[col + "_ms"] = (table.pop(col) / 1000).round(2)    # microseconds -> milliseconds
    save_table(ctx, table.reset_index(), "cloudwatch.csv", "CloudWatch per minute: calls, model and overhead time")


# ============================================================================ F7: delete
def delete(ctx, endpoint):
    """Delete the endpoint, its configuration and the model object, then wait until it is gone.

    Only the endpoint bills; the configuration and model object are free but are cleared too, so
    nothing is left behind. The registry version and its model.tar.gz in S3 are NOT touched:
    the same version can be redeployed any time in about 8 minutes.
    """
    sm = ctx["sm"]
    try:
        ep = sm.describe_endpoint(EndpointName=endpoint)
    except sm.exceptions.ClientError:
        log(f"Endpoint {endpoint} not found: nothing to delete.")
        return
    cfg = ep["EndpointConfigName"]
    models = [v["ModelName"] for v in sm.describe_endpoint_config(EndpointConfigName=cfg)["ProductionVariants"]]
    sm.delete_endpoint(EndpointName=endpoint)                 # this is the part that bills
    sm.delete_endpoint_config(EndpointConfigName=cfg)
    for m in models:
        sm.delete_model(ModelName=m)
    log(f"Deleted: endpoint {endpoint} | config {cfg} | models {models}")

    # Deletion is asynchronous (status "Deleting" first); poll until the endpoint no longer exists
    while True:
        names = [e["EndpointName"] for e in sm.list_endpoints(NameContains=endpoint)["Endpoints"]]
        if endpoint not in names:
            break
        log("Still deleting...")
        time.sleep(15)
    # Final safety net: show ANY endpoint still running in the account (forgotten ones bill too)
    left = [(e["EndpointName"], e["EndpointStatus"]) for e in sm.list_endpoints()["Endpoints"]]
    log("No endpoints left: billing stopped." if not left else f"WARNING, other endpoints still exist: {left}")


# ============================================================================ report
def write_report(ctx, pkg, endpoint):
    """Write deploy_report.md: what was deployed and every check with PASS / FAIL."""
    lines = [f"# Deployment report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"- Registry version: {pkg['version']} ({pkg['image']})",
             f"- Approval reason: {pkg['reason']}",
             f"- Endpoint: {endpoint} on {CFG['instance_count']} x {CFG['instance_type']}", "",
             "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['check']} | {c['result']} | {c['detail']} |" for c in ctx["checks"]]
    path = os.path.join(ctx["out"], "deploy_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    log(f"report saved: {path}")


def run_tests(ctx, endpoint, pkg):
    """F3-F6 in order, on the same test rows."""
    ctx["json_mode"] = pkg.get("reasons", False)     # JSON answers with reason codes, or plain CSV scores
    X, y, amount = load_test_data()
    section("F3. Smoke test: 10 real test transactions")
    smoke_test(ctx, endpoint, pkg["meta"], X, y, amount)
    section("F4. Whole test set through the endpoint: must reproduce the offline results")
    reproduce(ctx, endpoint, pkg["meta"], X, y, amount)
    section("F5. Real-time latency: one transaction per call")
    latency(ctx, endpoint, X)
    section("F6. CloudWatch: model time vs SageMaker overhead")
    cloudwatch(ctx, endpoint)


# ============================================================================ main
def main():
    p = argparse.ArgumentParser(description="Deploy, test and delete the approved fraud model.")
    p.add_argument("--all", action="store_true", help="deploy, test, then delete (always, unless --keep)")
    p.add_argument("--deploy", action="store_true", help="F1-F2 only: deploy and leave it running")
    p.add_argument("--test", action="store_true", help="F3-F6 only, on the last deployed endpoint")
    p.add_argument("--delete", action="store_true", help="F7 only, on the last deployed endpoint")
    p.add_argument("--keep", action="store_true", help="with --all: do not delete at the end")
    p.add_argument("--version", type=int, help="registry version to deploy (default: latest Approved)")
    p.add_argument("--endpoint", help="endpoint name for --test / --delete (default: from deploy_state.json)")
    a = p.parse_args()
    if not (a.all or a.deploy or a.test or a.delete):
        p.error("choose one of --all, --deploy, --test, --delete")

    section("F0. Setup")
    ctx = setup()

    if a.all:
        # The safe path: whatever happens after the endpoint exists, the finally block deletes it
        section("F1. Find the approved model package")
        pkg = find_package(ctx, a.version)
        endpoint = None
        try:
            section("F2. Deploy (billing starts)")
            endpoint = deploy(ctx, pkg)
            run_tests(ctx, endpoint, pkg)
        finally:
            # Runs on success, on a failed check, on a Python error, and on Ctrl+C
            if endpoint and not a.keep:
                section("F7. Delete the endpoint (billing stops)")
                delete(ctx, endpoint)
            elif endpoint:
                log(f"--keep: endpoint {endpoint} is STILL RUNNING and billing. "
                    f"Delete it with: python pipeline/deploy.py --delete")
            if endpoint:
                write_report(ctx, pkg, endpoint)

    elif a.deploy:
        section("F1. Find the approved model package")
        pkg = find_package(ctx, a.version)
        section("F2. Deploy (billing starts)")
        endpoint = deploy(ctx, pkg)
        log("Deployed and left running. Next: python pipeline/deploy.py --test, then python pipeline/deploy.py --delete")

    elif a.test:
        # A new process: find the endpoint and its package from the state file (or --endpoint)
        state = load_state()
        endpoint = a.endpoint or state["endpoint"]
        desc = ctx["sm"].describe_model_package(ModelPackageName=state["package_arn"])
        cont = desc["InferenceSpecification"]["Containers"][0]
        pkg = {"arn": state["package_arn"], "version": state["version"],
               "image": cont["Image"].split("/")[-1],
               "reason": desc.get("ApprovalDescription", ""),
               "meta": desc.get("CustomerMetadataProperties", {}),
               "reasons": "SAGEMAKER_PROGRAM" in cont.get("Environment", {})}
        run_tests(ctx, endpoint, pkg)
        write_report(ctx, pkg, endpoint)

    elif a.delete:
        endpoint = a.endpoint or load_state()["endpoint"]
        section("F7. Delete the endpoint (billing stops)")
        delete(ctx, endpoint)

    # Exit code for automation: 1 if any check failed (a CI/CD pipeline stops the release on it)
    failed = [c for c in ctx["checks"] if c["result"] == "FAIL"]
    if failed:
        log(f"{len(failed)} check(s) FAILED: {[c['check'] for c in failed]}")
        raise SystemExit(1)
    if ctx["checks"]:
        log(f"All {len(ctx['checks'])} checks passed.")


# Run main() only when the file is executed (python deploy.py), not when it is imported.
if __name__ == "__main__":
    main()
