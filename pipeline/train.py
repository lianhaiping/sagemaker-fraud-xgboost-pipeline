"""
train.py - the training code we own, run by SageMaker in "script mode".

WHERE THIS RUNS
    Not on your machine. train_pipeline.py asks SageMaker to start a training job; SageMaker starts an
    ml.m5.xlarge machine, pulls AWS's XGBoost container image (framework version 1.7-1), copies this
    file into it, and runs:   python train.py --max_depth 4 --min_child_weight 7.3 ...
    One training job = one run of this file = one trained model.

WHAT SAGEMAKER PROVIDES (the "/opt/ml" container contract)
    Hyperparameters  -> command-line arguments (--name value), read with argparse below
    SM_CHANNEL_TRAIN       -> folder with the training CSV      (/opt/ml/input/data/train)
    SM_CHANNEL_VALIDATION  -> folder with the validation CSV    (/opt/ml/input/data/validation)
    SM_MODEL_DIR           -> folder to save the model into     (/opt/ml/model)
    After this script ends, SageMaker tars SM_MODEL_DIR into model.tar.gz and uploads it to S3.

WHAT WE PROVIDE BACK
    The model file, plus log lines. Because this container image is the one AWS's built-in XGBoost
    algorithm uses, the tuner only accepts built-in metric names such as "validation:aucpr", read
    from the log in XGBoost's standard "[round]<TAB>validation-aucpr:<number>" format; the last such
    line wins. train.py therefore ends with one line in that format holding the BEST score.

DATA FORMAT: CSV, no header, label (0/1) in column 0, the 29 features after it.

RUNNING ON A LAPTOP (no SageMaker): put CSVs in data/train and data/validation, then
    python train.py --max_depth 4
The os.environ.get(..., default) calls below fall back to those local folders.
"""
import argparse
import os

import pandas as pd
import xgboost as xgb


def parse_args():
    """Read hyperparameters from the command line.

    SageMaker turns the estimator's hyperparameters={"max_depth": 4, ...} (and the tuner's
    random draws) into arguments like "--max_depth 4". Each add_argument says: the name, the
    type to convert the text to, and a default if the argument is missing.
    """
    p = argparse.ArgumentParser()

    # ---- Fixed settings: the same value for every trial (set in CFG["fixed"] in train_pipeline.py)
    p.add_argument("--eta", type=float, default=0.05)                 # learning rate: each tree's correction x 0.05
    p.add_argument("--num_round", type=int, default=2000)             # cap on the number of trees
    p.add_argument("--early_stopping_rounds", type=int, default=100)  # patience on validation AUC-PR
    p.add_argument("--scale_pos_weight", type=float, default=1.0)     # each fraud counts this many times
    p.add_argument("--seed", type=int, default=1)                     # fixes row/column sampling -> reproducible

    # ---- Searched settings: the tuner draws a random value for each trial
    p.add_argument("--max_depth", type=int, default=5)                # questions in a row per tree
    p.add_argument("--min_child_weight", type=float, default=1.0)     # minimum evidence per leaf
    p.add_argument("--subsample", type=float, default=0.8)            # share of rows each tree sees
    p.add_argument("--colsample_bytree", type=float, default=0.8)     # share of features each tree may use
    p.add_argument("--gamma", type=float, default=0.0)                # minimum loss improvement to split
    # "lambda" is a reserved word in Python, so store it under another name (dest="reg_lambda")
    p.add_argument("--lambda", dest="reg_lambda", type=float, default=1.0)   # L2 penalty on leaf values
    p.add_argument("--alpha", type=float, default=0.0)                # L1 penalty on leaf values

    # parse_known_args (not parse_args) ignores any extra arguments the container may add,
    # instead of crashing on them. The "_" receives the ignored ones.
    args, _ = p.parse_known_args()
    return args


def load_channel(path):
    """Read every CSV file in a channel folder into one XGBoost DMatrix.

    SageMaker copies everything under the channel's S3 prefix into this folder, so there may be
    one file or several; we read them all and stack them (pd.concat).
    Returns: the DMatrix, the number of rows, and the number of frauds (for a sanity printout).
    """
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)
    # .values -> plain NumPy arrays with NO column names, matching what the endpoint will receive
    # (a model trained WITH column names refuses to score data without them)
    X = df.iloc[:, 1:].values      # all rows, columns 1..29 = features
    y = df.iloc[:, 0].values       # all rows, column 0 = label
    return xgb.DMatrix(X, label=y), len(df), int(y.sum())


def main():
    a = parse_args()

    # Where SageMaker put the data and where it wants the model; local defaults for laptop runs
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "data/train")
    val_dir = os.environ.get("SM_CHANNEL_VALIDATION", "data/validation")
    model_dir = os.environ.get("SM_MODEL_DIR", "model")

    dtrain, n_tr, f_tr = load_channel(train_dir)
    dval, n_va, f_va = load_channel(val_dir)
    # flush=True pushes the line to the log immediately (otherwise Python may buffer it)
    print(f"train rows={n_tr} frauds={f_tr} | validation rows={n_va} frauds={f_va}", flush=True)

    # XGBoost's native parameter dictionary
    params = {
        "objective": "binary:logistic",     # output a fraud probability from 0 to 1
        "eval_metric": "aucpr",             # score each round by area under the precision-recall curve
        "tree_method": "hist",              # bin feature values first: much faster split search
        "eta": a.eta,
        "max_depth": a.max_depth,
        "min_child_weight": a.min_child_weight,
        "subsample": a.subsample,
        "colsample_bytree": a.colsample_bytree,
        "gamma": a.gamma,
        "lambda": a.reg_lambda,
        "alpha": a.alpha,
        "scale_pos_weight": a.scale_pos_weight,
        "seed": a.seed,
        "nthread": os.cpu_count(),          # use every CPU core on the training machine
    }
    print("params:", params, flush=True)

    # Train with early stopping.
    # evals: after every round, score the model on both sets and print them every 50 rounds.
    # early_stopping_rounds watches the LAST set in evals (validation, at the real 0.17% fraud rate):
    # if validation AUC-PR has not improved for this many rounds, training stops.
    booster = xgb.train(params, dtrain, num_boost_round=a.num_round,
                        evals=[(dtrain, "train"), (dval, "validation")],
                        early_stopping_rounds=a.early_stopping_rounds, verbose_eval=50)

    best = booster.best_iteration          # the best round, counted from 0
    best_score = booster.best_score        # validation AUC-PR at that round
    # Training ran up to `early_stopping_rounds` past the best round; slicing keeps trees 0..best,
    # so the saved model is exactly the best one (no extra trees reach the endpoint).
    booster = booster[: best + 1]

    # Readable summary lines for the log
    print(f"best_iteration={best}", flush=True)
    print(f"trees_kept={best + 1}", flush=True)
    print(f"best_validation_aucpr={best_score:.6f}", flush=True)

    # The line the TUNER reads. This container image is the same one AWS uses for its built-in
    # XGBoost algorithm, so the tuning service only accepts the built-in metric names (such as
    # "validation:aucpr") and finds them in the log with AWS's own pattern: a line that starts
    # with "[round]" and contains a TAB followed by "validation-aucpr:<number>".
    # XGBoost's per-round log lines already look like that, and the tuner keeps the LAST value
    # it sees. So we print one final line in the same format with the BEST score: that becomes
    # the trial's objective value. Do not change its format.
    print(f"[{best}]\tvalidation-aucpr:{best_score:.6f}", flush=True)

    # Save into SM_MODEL_DIR; SageMaker packs this folder into model.tar.gz and uploads it to S3.
    # JSON format: a newer XGBoost (2.1 locally, 3.0 in the serving image) can read it; the reverse is not guaranteed.
    os.makedirs(model_dir, exist_ok=True)
    booster.save_model(os.path.join(model_dir, "xgboost-model.json"))


# Run main() only when the file is executed (python train.py), not when it is imported.
if __name__ == "__main__":
    main()
